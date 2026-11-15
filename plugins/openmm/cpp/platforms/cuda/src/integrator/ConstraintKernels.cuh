// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// -----------------------------------------------------------------------------
// SETTLE and SHAKE: the distance-constraint solvers of the Langevin middle step.
//
// Included by Constraints.cuh after its storage and arithmetic helpers (CV, CVD,
// loadC4/storeC4, loadD4/storeD4, invW, loadPos4, nvsqrt, nvrsqrt, nvrecip, nvfabs,
// nvmk3). Every function is templated on the compute type Real: float (single
// precision), double, or matx::fltflt (a pair of floats, the mixed precision).
//
// Two kinds of constraint cluster are solved, each by one thread:
//
// * SETTLE, for rigid three-site water: S. Miyamoto and P. A. Kollman, J. Comput.
//   Chem. 13, 952-962 (1992). The positions are corrected in closed form: the
//   correct triangle is built in a frame tied to the old molecular plane, and three
//   rotations (about the frame's x', y' and z' axes) bring it onto the displaced
//   triangle while keeping the centre of mass. The velocities are corrected in closed
//   form too: the three impulses along the three bonds that cancel every relative
//   velocity along those bonds solve a 3x3 linear system, written out by Cramer's
//   rule.
//
// * SHAKE, for a central atom bonded to one, two or three peripheral atoms of equal
//   mass with equal bond lengths: J.-P. Ryckaert, G. Ciccotti and H. J. C. Berendsen,
//   J. Comput. Phys. 23, 327-341 (1977). Each bond in turn is corrected along its old
//   direction, the correction taking effect at once for the next bond (Gauss-Seidel),
//   and the sweeps repeat until no bond needs a correction or 15 sweeps have been
//   made. The velocity form removes the relative velocity along each bond in the same
//   way, the second half of H. C. Andersen's RATTLE, J. Comput. Phys. 52, 24-34 (1983).
//
// Order of operations. In mixed precision the separate kernels and the fused step's
// lane-parallel SETTLE (SettleLanes.cuh) must give the same results bit for bit, and
// float-float arithmetic is neither associative nor, for products, commutative in its
// last bit; an operation between a float-float value and a plain float is also a
// different algorithm from the one between two float-float values. So the operand
// order and operand types of every expression below are fixed, and changing them
// changes the results: sums are evaluated left to right, a vector times a scalar has
// the vector component on the left, and plain-float constants and cluster parameters
// stay plain floats.
// -----------------------------------------------------------------------------

namespace NvOpenMM {

// Small three-vector helpers, generic over float3, double3 and ff3. They keep the
// operand order fixed: the left operand of each product is the first argument's
// component.
namespace cvec {

template<class V> static __device__ __forceinline__ V sub(const V& a, const V& b) {
    return nvmk3(a.x - b.x, a.y - b.y, a.z - b.z);
}
template<class V> static __device__ __forceinline__ V add(const V& a, const V& b) {
    return nvmk3(a.x + b.x, a.y + b.y, a.z + b.z);
}
// a times the scalar s, component on the left. s may be a plain float.
template<class V, class S> static __device__ __forceinline__ V scale(const V& a, S s) {
    return nvmk3(a.x * s, a.y * s, a.z * s);
}
// Dot product summed x, y, z left to right, a's component on the left of each product.
template<class V> static __device__ __forceinline__ auto dot(const V& a, const V& b) -> decltype(a.x * b.x) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template<class V> static __device__ __forceinline__ V cross(const V& a, const V& b) {
    return nvmk3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
// a divided, component by component, by its length.
template<class V> static __device__ __forceinline__ V unit(const V& a) {
    const auto len = nvsqrt(dot(a, a));
    return nvmk3(a.x / len, a.y / len, a.z / len);
}
template<class V4> static __device__ __forceinline__ auto xyz(const V4& v) -> decltype(nvmk3(v.x, v.y, v.z)) {
    return nvmk3(v.x, v.y, v.z);
}
// v4 with x, y, z replaced and w kept.
template<class V4, class V> static __device__ __forceinline__ V4 withXyz(V4 v, const V& p) {
    v.x = p.x; v.y = p.y; v.z = p.z;
    return v;
}

}  // namespace cvec

// ================================ SETTLE ======================================

// Result of the SETTLE frame solve: the corrected triangle in the working frame,
// before its last rotation about z', and that rotation.
//   aY:         y' of atom A (A lies on the y' axis, x' = 0)
//   bX:         x' of atom B; atom C is at x' = -bX
//   bY, cY:     y' of atoms B and C
//   rotSin, rotCos: sine and cosine of the rotation angle theta about z'
template<class Real> struct SettleFit {
    Real aY, bX, bY, cY, rotSin, rotCos;
};

// The SETTLE frame solve (Miyamoto and Kollman 1992, the position section).
//
// The working frame has its z' axis normal to the old molecular plane and its x'
// axis normal to both z' and the displaced position of A (relative to the displaced
// centre of mass), so A's displaced position has x' = 0. The correct (canonical)
// triangle is described by ra (centre of mass to A along the height), rb (centre of
// mass to the B-C midpoint) and rc (half the B-C distance). Three rotations bring it
// onto the displaced one:
//   phi about x', fixed by A's displaced height zA above the old plane;
//   psi about y', fixed by the height difference zB - zC of B and C;
//   theta about z', fixed by requiring that the displacement from the old triangle
//   to the new one has no net angular momentum about z' (the paper's alpha, beta,
//   gamma equation, solved for sin theta).
// After phi and psi, B's x' coordinate is corrected so that the B-C distance equals
// sqrt(q) exactly, which removes the error the small-angle form would leave.
//
// The solve has two stages, kept as separate functions so that the lane-parallel
// version can fetch the second stage's inputs only when it needs them:
//   settleTilt: phi, psi and the B-C correction, from ra, rb, rc, rc2 = 2 rc,
//     q = (B-C length)^2 and the displaced z' of A, B, C; it fills aY, bX, bY, cY;
//   settleTurn: theta, from those and the in-plane coordinates (x', y') of B and C,
//     old (relative to A) and displaced (relative to the displaced centre of mass);
//     it fills rotSin and rotCos.
template<class Real>
static __device__ __forceinline__ SettleFit<Real> settleTilt(Real ra, Real rb, Real rc, Real rc2, Real q,
                                                            Real zA, Real zB, Real zC) {
    const Real one = Real(1.0f);
    // Rotation about x' (phi) and about y' (psi).
    const Real sinPhi = zA / ra;
    const Real cosPhi = nvsqrt(one - sinPhi * sinPhi);
    const Real zBC = zB - zC;
    const Real sinPsi = zBC / (rc2 * cosPhi);
    const Real cosPsi = nvsqrt(one - sinPsi * sinPsi);

    // The canonical triangle after the two rotations, projected on the x'y' plane.
    SettleFit<Real> f;
    f.aY = ra * cosPhi;
    Real bX = -(rc * cosPsi);
    const Real tilt = rc * sinPsi * sinPhi;
    const Real heightB = rb * cosPhi;
    f.bY = -heightB - tilt;
    f.cY = -heightB + tilt;

    // Correct bX so that (2 bX)^2 + dy^2 + dz^2 = q, i.e. |B - C| = sqrt(q):
    // with H the current squared distance, the corrected 2|bX| is sqrt(4 bX^2 - H + q).
    const Real bX2 = bX * bX;
    const Real yBC = f.bY - f.cY;
    const Real distSq = 4.0f * bX2 + yBC * yBC + zBC * zBC;
    const Real shift = 2.0f * bX + nvsqrt(4.0f * bX2 - distSq + q);
    f.bX = bX - shift * 0.5f;
    return f;
}

// Rotation about z' (theta). Requiring that the move from the old triangle to the
// rotated one carries no angular momentum about z' gives
// alpha sin(theta) + beta cos(theta) = gamma; with n = alpha^2 + beta^2 its root is
// sin(theta) = (alpha gamma - beta sqrt(n - gamma^2)) / n.
template<class Real>
static __device__ __forceinline__ void settleTurn(SettleFit<Real>& f, Real xB0, Real yB0, Real xC0, Real yC0,
                                                  Real xB1, Real yB1, Real xC1, Real yC1) {
    const Real one = Real(1.0f);
    const Real alpha = f.bX * (xB0 - xC0) + yB0 * f.bY + yC0 * f.cY;
    const Real beta = f.bX * (yC0 - yB0) + xB0 * f.bY + xC0 * f.cY;
    const Real gamma = xB0 * yB1 - xB1 * yB0 + xC0 * yC1 - xC1 * yC0;
    const Real norm = alpha * alpha + beta * beta;
    f.rotSin = (alpha * gamma - beta * nvsqrt(norm - gamma * gamma)) / norm;
    f.rotCos = nvsqrt(one - f.rotSin * f.rotSin);
}

// Both stages.
template<class Real>
static __device__ __forceinline__ SettleFit<Real> settleFitFrame(
        Real ra, Real rb, Real rc, Real rc2, Real q, Real zA, Real zB, Real zC,
        Real xB0, Real yB0, Real xC0, Real yC0, Real xB1, Real yB1, Real xC1, Real yC1) {
    SettleFit<Real> f = settleTilt(ra, rb, rc, rc2, q, zA, zB, zC);
    settleTurn(f, xB0, yB0, xC0, yC0, xB1, yB1, xC1, yC1);
    return f;
}

// The canonical triangle's dimensions from the masses (mA the atom with two equal
// bonds) and the float lengths d1 (A-B, A-C) and d2 (B-C), with invTotal =
// 1/(mA + mB + mC). The height h of the isosceles triangle is split by the centre of
// mass into ra (towards A) and rb (towards the B-C midpoint); rc is half of d2.
template<class Real> struct SettleShape {
    Real ra, rb, rc, rc2, q;
};
// The terms that depend only on the lengths: rc, 2 rc and q = d2^2.
template<class Real>
static __device__ __forceinline__ SettleShape<Real> settleShapeLengths(float d2) {
    SettleShape<Real> s;
    s.rc = Real(0.5f * d2);
    s.rc2 = 2.0f * s.rc;
    const Real len2 = Real(d2);
    s.q = len2 * len2;
    return s;
}
template<class Real>
static __device__ __forceinline__ SettleShape<Real> settleShape(Real mB, Real mC, Real invTotal, float d1, float d2) {
    SettleShape<Real> s = settleShapeLengths<Real>(d2);
    const Real len1 = Real(d1);
    const Real h = nvsqrt(len1 * len1 - s.rc * s.rc);
    s.ra = h * (mB + mC) * invTotal;
    s.rb = h - s.ra;
    return s;
}

// The SETTLE working frame as three unit axes, from b0, c0 (old B and C relative to
// old A) and a1 (displaced A relative to the displaced centre of mass).
template<class V3> struct SettleAxes {
    V3 ex, ey, ez;
};
template<class V3>
static __device__ __forceinline__ SettleAxes<V3> settleAxes(const V3& b0, const V3& c0, const V3& a1) {
    // z' normal to the old plane; x' normal to z' and to A's displaced position, so A
    // sits on the y'z' plane; y' completes the right-handed frame.
    const V3 z = cvec::cross(b0, c0);
    const V3 x = cvec::cross(a1, z);
    const V3 y = cvec::cross(z, x);
    return SettleAxes<V3>{cvec::unit(x), cvec::unit(y), cvec::unit(z)};
}

// A frame vector (x', y', z') back in the lab frame: sum over the axes of axis
// component times frame coordinate.
template<class V3, class Real>
static __device__ __forceinline__ V3 settleToLab(const SettleAxes<V3>& f, Real px, Real py, Real pz) {
    return nvmk3(f.ex.x * px + f.ey.x * py + f.ez.x * pz,
                 f.ex.y * px + f.ey.y * py + f.ez.y * pz,
                 f.ex.z * px + f.ey.z * py + f.ez.z * pz);
}

// One water's position constraint, all three atoms on one thread.
// pA, pB, pC: old positions; dA, dB, dC: this step's displacements (updated in
// place); mA, mB, mC: masses; d1, d2: the float bond lengths.
template<class Real>
static __device__ __forceinline__ void settleWaterPositions(
        const typename CV<Real>::v3& pA, const typename CV<Real>::v3& pB, const typename CV<Real>::v3& pC,
        typename CV<Real>::v3& dA, typename CV<Real>::v3& dB, typename CV<Real>::v3& dC,
        Real mA, Real mB, Real mC, float d1, float d2) {
    using V3 = typename CV<Real>::v3;
    // Old B and C relative to old A.
    const V3 b0 = cvec::sub(pB, pA);
    const V3 c0 = cvec::sub(pC, pA);
    const Real invTotal = nvrecip(mA + mB + mC);

    // Displaced positions relative to old A, and their centre of mass.
    const V3 movedB = cvec::add(b0, dB);
    const V3 movedC = cvec::add(c0, dC);
    const V3 com = nvmk3((dA.x * mA + movedB.x * mB + movedC.x * mC) * invTotal,
                         (dA.y * mA + movedB.y * mB + movedC.y * mC) * invTotal,
                         (dA.z * mA + movedB.z * mB + movedC.z * mC) * invTotal);
    const V3 a1 = cvec::sub(dA, com);
    const V3 b1 = cvec::sub(movedB, com);
    const V3 c1 = cvec::sub(movedC, com);

    const SettleAxes<V3> ax = settleAxes(b0, c0, a1);
    const SettleShape<Real> sh = settleShape<Real>(mB, mC, invTotal, d1, d2);
    const Real zA = cvec::dot(ax.ez, a1);
    const Real zB = cvec::dot(ax.ez, b1);
    const Real zC = cvec::dot(ax.ez, c1);
    const SettleFit<Real> f = settleFitFrame<Real>(sh.ra, sh.rb, sh.rc, sh.rc2, sh.q, zA, zB, zC,
            cvec::dot(ax.ex, b0), cvec::dot(ax.ey, b0), cvec::dot(ax.ex, c0), cvec::dot(ax.ey, c0),
            cvec::dot(ax.ex, b1), cvec::dot(ax.ey, b1), cvec::dot(ax.ex, c1), cvec::dot(ax.ey, c1));

    // Rotate the fitted triangle by theta about z' and return it to the lab frame;
    // the new displacements are the corrected positions minus the old ones.
    const V3 a3 = settleToLab(ax, -(f.aY * f.rotSin), f.aY * f.rotCos, zA);
    const V3 b3 = settleToLab(ax, f.bX * f.rotCos - f.bY * f.rotSin, f.bX * f.rotSin + f.bY * f.rotCos, zB);
    const V3 c3 = settleToLab(ax, -(f.bX * f.rotCos) - f.cY * f.rotSin, -(f.bX * f.rotSin) + f.cY * f.rotCos, zC);
    dA = cvec::add(com, a3);
    dB = cvec::sub(cvec::add(com, b3), b0);
    dC = cvec::sub(cvec::add(com, c3), c0);
}

template<class Real>
__global__ void settleToPositions(int numClusters, const float4* __restrict__ oldPos,
        typename CVD<Real>::s4* __restrict__ posDelta,
        const typename CV<Real>::s4* __restrict__ velm,
        const int4* __restrict__ clusterAtoms, const float2* __restrict__ clusterParams,
        const float4* __restrict__ posqCorrection) {
    using V4 = typename CV<Real>::v4;
    Real* tag = nullptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numClusters; i += gridDim.x * blockDim.x) {
        const int4 at = clusterAtoms[i];
        const float2 len = clusterParams[i];
        const V4 pA = loadPos4<Real>(oldPos, posqCorrection, at.x);
        const V4 pB = loadPos4<Real>(oldPos, posqCorrection, at.y);
        const V4 pC = loadPos4<Real>(oldPos, posqCorrection, at.z);
        V4 dA = loadD4(posDelta[at.x], tag);
        V4 dB = loadD4(posDelta[at.y], tag);
        V4 dC = loadD4(posDelta[at.z], tag);
        // The three stored velocities (of which invW uses the inverse mass) are read
        // before the first conversion, so their reads are issued with the others
        // instead of each after the previous mass's conversion.
        const auto sA = velm[at.x], sB = velm[at.y], sC = velm[at.z];
        const Real mA = invW(sA, tag);
        const Real mB = invW(sB, tag);
        const Real mC = invW(sC, tag);
        auto a = cvec::xyz(dA), b = cvec::xyz(dB), c = cvec::xyz(dC);
        settleWaterPositions<Real>(cvec::xyz(pA), cvec::xyz(pB), cvec::xyz(pC), a, b, c, mA, mB, mC, len.x, len.y);
        posDelta[at.x] = storeD4(cvec::withXyz(dA, a));
        posDelta[at.y] = storeD4(cvec::withXyz(dB, b));
        posDelta[at.z] = storeD4(cvec::withXyz(dC, c));
    }
}

// The SETTLE velocity impulses (Miyamoto and Kollman 1992, the velocity section).
//
// The three bonds AB, BC and CA, with unit vectors eAB, eBC, eCA along the current
// geometry, each carry an impulse tau along themselves. Requiring that every bond's
// relative velocity (uAB, uBC, uCA, each the far atom's velocity minus the near
// atom's, along the bond) vanishes after the impulses gives a 3x3 linear system whose
// coefficients are the masses and the cosines cA, cB, cC of the triangle's angles at
// A, B and C. Its determinant is D below, and the three taus are written out by
// Cramer's rule, every term multiplied through by mA mB mC (mu is the inverse of
// that product, total the sum of the masses).
// The three pieces of that solution, shared with the lane-parallel version:
//   settleDeterminant: D, from the sine-squared terms sA, sB, sC (1 - c^2), the
//     cosines and the masses;
//   settleCofactor: an off-diagonal term p q x - t u - t v, in this operand order;
//   settleDiagonal: a diagonal term s m1 m1 m2 m2 mu + total.
template<class Real>
static __device__ __forceinline__ Real settleDeterminant(Real sA, Real sB, Real sC, Real cA, Real cB, Real cC,
                                                         Real mA, Real mB, Real mC, Real mu) {
    const Real one = Real(1.0f);
    return (((sA * mB + sB * mA) * mC
             + (sA * mB * mB + 2.0f * (cA * cB * cC + one) * mA * mB + sB * mA * mA)) * mC
            + sC * mA * mB * (mA + mB)) * mu;
}
template<class Real>
static __device__ __forceinline__ Real settleCofactor(Real p, Real q, Real x, Real t, Real u, Real v) {
    return p * q * x - t * u - t * v;
}
template<class Real>
static __device__ __forceinline__ Real settleDiagonal(Real s, Real m1, Real m2, Real mu, Real total) {
    return s * m1 * m1 * m2 * m2 * mu + total;
}

template<class Real> struct SettleImpulse {
    Real ab, bc, ca;
};
template<class V3, class Real>
static __device__ __forceinline__ SettleImpulse<Real> settleImpulses(
        const V3& eAB, const V3& eBC, const V3& eCA, Real uAB, Real uBC, Real uCA,
        Real mA, Real mB, Real mC, Real mu, Real total) {
    const Real one = Real(1.0f);
    const Real cA = -cvec::dot(eAB, eCA);
    const Real cB = -cvec::dot(eAB, eBC);
    const Real cC = -cvec::dot(eBC, eCA);
    const Real sA = one - cA * cA;   // sine squared of each angle
    const Real sB = one - cB * cB;
    const Real sC = one - cC * cC;

    const Real det = settleDeterminant(sA, sB, sC, cA, cB, cC, mA, mB, mC, mu);
    const Real kAB = settleCofactor(cB, cC, mA, cA, mB, mC);
    const Real kBC = settleCofactor(cA, cC, mB, cB, mC, mA);
    const Real kCA = settleCofactor(cA, cB, mC, cC, mB, mA);
    const Real kBCswap = settleCofactor(cA, cC, mB, cB, mA, mC);

    SettleImpulse<Real> t;
    t.ab = (kAB * uCA + kBC * uBC + settleDiagonal(sC, mA, mB, mu, total) * uAB) / det;
    t.bc = (kCA * uCA + settleDiagonal(sA, mB, mC, mu, total) * uBC + kBCswap * uAB) / det;
    t.ca = (settleDiagonal(sB, mA, mC, mu, total) * uCA + kCA * uBC + kAB * uAB) / det;
    return t;
}

// The velocity change of one atom: the impulse of the bond leaving it along that
// bond, minus the impulse of the bond arriving at it along that bond, times the
// atom's inverse mass, per component.
template<class V3, class Real>
static __device__ __forceinline__ V3 settleKick(const V3& v, Real tOut, const V3& eOut, Real tIn, const V3& eIn, Real w) {
    return nvmk3(v.x + (tOut * eOut.x - tIn * eIn.x) * w,
                 v.y + (tOut * eOut.y - tIn * eIn.y) * w,
                 v.z + (tOut * eOut.z - tIn * eIn.z) * w);
}

// Unit vector from a to b, scaled by the reciprocal square root of its squared length.
template<class V3>
static __device__ __forceinline__ V3 settleBondUnit(const V3& a, const V3& b) {
    const V3 r = cvec::sub(b, a);
    return cvec::scale(r, nvrsqrt(cvec::dot(r, r)));
}

template<class Real>
__global__ void settleToVelocities(int numClusters, const float4* __restrict__ oldPos,
        typename CV<Real>::s4* __restrict__ velm,
        const int4* __restrict__ clusterAtoms, const float4* __restrict__ posqCorrection) {
    using V3 = typename CV<Real>::v3;
    using V4 = typename CV<Real>::v4;
    Real* tag = nullptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numClusters; i += gridDim.x * blockDim.x) {
        const int4 at = clusterAtoms[i];
        const V3 pA = cvec::xyz(loadPos4<Real>(oldPos, posqCorrection, at.x));
        const V3 pB = cvec::xyz(loadPos4<Real>(oldPos, posqCorrection, at.y));
        const V3 pC = cvec::xyz(loadPos4<Real>(oldPos, posqCorrection, at.z));
        const V4 vA = loadC4(velm[at.x], tag);
        const V4 vB = loadC4(velm[at.y], tag);
        const V4 vC = loadC4(velm[at.z], tag);
        const Real mA = nvrecip(vA.w), mB = nvrecip(vB.w), mC = nvrecip(vC.w);

        const V3 eAB = settleBondUnit(pA, pB);
        const V3 eBC = settleBondUnit(pB, pC);
        const V3 eCA = settleBondUnit(pC, pA);
        const V3 xA = cvec::xyz(vA), xB = cvec::xyz(vB), xC = cvec::xyz(vC);
        const Real uAB = cvec::dot(cvec::sub(xB, xA), eAB);
        const Real uBC = cvec::dot(cvec::sub(xC, xB), eBC);
        const Real uCA = cvec::dot(cvec::sub(xA, xC), eCA);
        const SettleImpulse<Real> t = settleImpulses(eAB, eBC, eCA, uAB, uBC, uCA, mA, mB, mC,
                                                     nvrecip(mA * mB * mC), mA + mB + mC);
        velm[at.x] = storeC4(cvec::withXyz(vA, settleKick(xA, t.ab, eAB, t.ca, eCA, vA.w)));
        velm[at.y] = storeC4(cvec::withXyz(vB, settleKick(xB, t.bc, eBC, t.ab, eAB, vB.w)));
        velm[at.z] = storeC4(cvec::withXyz(vC, settleKick(xC, t.ca, eCA, t.bc, eBC, vC.w)));
    }
}

// ================================= SHAKE ======================================
//
// A cluster is a central atom (slot 0) and peripheral atoms in slots 1, 2, 3; slots 2
// and 3 are present when their atom index is not -1. Parameters, all plain floats:
// params.x = central inverse mass, params.y = 1 / (2 (central + peripheral inverse
// mass)), params.z = squared bond length, params.w = peripheral inverse mass.
//
// The bond vectors r_k = (central - peripheral) are taken from the old positions and
// stay fixed during the iteration; each correction moves the two atoms of bond k along
// r_k in inverse proportion to their masses (Ryckaert, Ciccotti and Berendsen 1977).

// The old geometry of a cluster. bond(k) gives r_k = (central - peripheral k) from the
// old positions and its squared length, for a present slot k (1..3), in one of two
// forms that give the same values:
// - KEEP: every r_k and |r_k|^2 is formed once, before the sweeps, and kept.
// - otherwise only the central position is kept, and r_k and |r_k|^2 are formed
//   again, by the same operations, each time a sweep visits bond k. Holding all three
//   r_k and r_k^2 costs 24 registers in mixed precision, which inside the fused kernel
//   (capped at 96) spill to local memory; re-forming them costs one position load and
//   four float-float operations per bond and sweep.
// The fused kernel keeps them in single precision only, where it has registers to spare
// (72 of its 96 on sm_89, without spills); forming them again in every sweep measured
// about 1 us per step slower on DHFR there, on the RTX 6000 Ada and the RTX 5090. The
// separate SHAKE kernels have no register cap to stay under, and keep them in every
// precision.
template<class Real, bool KEEP> struct ShakeBonds;
template<class Real> struct ShakeBonds<Real, false> {
    using V3 = typename CV<Real>::v3;
    const float4* __restrict__ oldPos;
    const float4* __restrict__ corr;
    int4 atoms;
    V3 centre;
    __device__ __forceinline__ int atom(int k) const {
        return k == 0 ? atoms.x : k == 1 ? atoms.y : k == 2 ? atoms.z : atoms.w;
    }
    __device__ __forceinline__ bool present(int k) const { return k <= 1 || atom(k) != -1; }
    __device__ __forceinline__ void bond(int k, V3& r, Real& r2) const {
        r = cvec::sub(centre, cvec::xyz(loadPos4<Real>(oldPos, corr, atom(k))));
        r2 = cvec::dot(r, r);
    }
};
template<class Real> struct ShakeBonds<Real, true> {
    using V3 = typename CV<Real>::v3;
    V3 r[4];
    Real r2[4];
    int4 atoms;
    __device__ __forceinline__ int atom(int k) const {
        return k == 0 ? atoms.x : k == 1 ? atoms.y : k == 2 ? atoms.z : atoms.w;
    }
    __device__ __forceinline__ bool present(int k) const { return k <= 1 || atom(k) != -1; }
    // k is a compile-time constant at every call (the sweeps' loops over k are unrolled).
    __device__ __forceinline__ void bond(int k, V3& rk, Real& rk2) const { rk = r[k]; rk2 = r2[k]; }
};
// Whether the fused kernel keeps the bond vectors (see ShakeBonds).
template<class Real> static constexpr bool shakeKeepsBonds = std::is_same<Real, float>::value;
template<class Real, bool KEEP>
static __device__ __forceinline__ ShakeBonds<Real, KEEP> shakeBonds(int4 atoms, const float4* __restrict__ oldPos,
                                                                   const float4* __restrict__ posqCorrection) {
    ShakeBonds<Real, KEEP> b;
    b.atoms = atoms;
    const auto centre = cvec::xyz(loadPos4<Real>(oldPos, posqCorrection, atoms.x));
    if constexpr (KEEP) {
#pragma unroll
        for (int k = 1; k < 4; k++)
            if (b.present(k)) {
                b.r[k] = cvec::sub(centre, cvec::xyz(loadPos4<Real>(oldPos, posqCorrection, b.atom(k))));
                b.r2[k] = cvec::dot(b.r[k], b.r[k]);
            }
    } else {
        b.oldPos = oldPos;
        b.corr = posqCorrection;
        b.centre = centre;
    }
    return b;
}

// Writes x, y, z back to a slot. The fourth component is read again through the
// accessor rather than carried through the iteration (the per-cluster functions run
// inside the fused kernel at a cap of 96 registers); it is the value loaded at the
// start, since nothing stores to the slot in between.
template<class Real, class Access, class V3>
static __device__ __forceinline__ void shakeStore(const Access& acc, int atom, int slot, const V3& p) {
    acc.store(atom, slot, cvec::withXyz(acc.load(atom, slot), p));
}

// SHAKE on the displacements. For bond k, with s the current central-minus-peripheral
// displacement difference, the new squared bond length is |r + s|^2 = r^2 + 2 r.s +
// s^2, and eps = d^2 - that is the error. A first-order correction along r of size
// g = eps / (2 (wc + wp) (r.s + r^2)) moves the central atom by r g wc and the
// peripheral atom by -r g wp. A bond counts as satisfied when |eps| < d^2 tol.
template<class Real, bool KEEP = shakeKeepsBonds<Real>, class Access>
static __device__ __forceinline__ void shakeClusterPositions(int4 atoms, float4 params, Real tol,
        const float4* __restrict__ oldPos, const float4* __restrict__ posqCorrection, const Access& acc) {
    using V3 = typename CV<Real>::v3;
    const int idx[4] = {atoms.x, atoms.y, atoms.z, atoms.w};
    const float wc = params.x, kappa = params.y, d2 = params.z, wp = params.w;
    const ShakeBonds<Real, KEEP> b = shakeBonds<Real, KEEP>(atoms, oldPos, posqCorrection);
    V3 d[4];
#pragma unroll
    for (int k = 0; k < 4; k++)
        if (b.present(k)) d[k] = cvec::xyz(acc.load(idx[k], k));
    const Real limit = d2 * tol;

    bool corrected = true;
    for (int sweep = 0; sweep < 15 && corrected; sweep++) {
        corrected = false;
#pragma unroll
        for (int k = 1; k < 4; k++) {
            if (!b.present(k)) continue;
            V3 rk; Real rk2;
            b.bond(k, rk, rk2);
            const V3 s = cvec::sub(d[0], d[k]);
            const Real s2 = cvec::dot(s, s);
            const Real rs = cvec::dot(rk, s);
            // d^2 - r^2 is the error before any displacement.
            const Real eps = (d2 - rk2) - 2.0f * rs - s2;
            // A NaN ratio compares false and applies nothing.
            if (nvfabs(eps) / limit >= 1.0f) {
                const Real g = eps * kappa / (rs + rk2);
                const V3 step = cvec::scale(rk, g);
                d[0] = cvec::add(d[0], cvec::scale(step, wc));
                d[k] = cvec::sub(d[k], cvec::scale(step, wp));
                corrected = true;
            }
        }
    }
#pragma unroll
    for (int k = 0; k < 4; k++)
        if (b.present(k)) shakeStore<Real>(acc, idx[k], k, d[k]);
}

// SHAKE on the velocities (the velocity half of RATTLE). For bond k, with s the
// central-minus-peripheral velocity difference, the impulse that removes the
// relative velocity along r is delta = -2 kappa (s.r) / r^2 (kappa = 1/(2(wc + wp)));
// it changes the central velocity by r delta wc and the peripheral one by -r delta wp.
// Every bond is corrected on every sweep; the sweeps stop once every |delta| is at
// most tol.
template<class Real, bool KEEP = shakeKeepsBonds<Real>, class Access>
static __device__ __forceinline__ void shakeClusterVelocities(int4 atoms, float4 params, Real tol,
        const float4* __restrict__ oldPos, const float4* __restrict__ posqCorrection, const Access& acc) {
    using V3 = typename CV<Real>::v3;
    const int idx[4] = {atoms.x, atoms.y, atoms.z, atoms.w};
    const float wc = params.x, kappa = params.y, wp = params.w;
    const float gain = -2.0f * kappa;
    const ShakeBonds<Real, KEEP> b = shakeBonds<Real, KEEP>(atoms, oldPos, posqCorrection);
    V3 v[4];
#pragma unroll
    for (int k = 0; k < 4; k++)
        if (b.present(k)) v[k] = cvec::xyz(acc.load(idx[k], k));

    bool moving = true;
    for (int sweep = 0; sweep < 15 && moving; sweep++) {
        moving = false;
#pragma unroll
        for (int k = 1; k < 4; k++) {
            if (!b.present(k)) continue;
            V3 rk; Real rk2;
            b.bond(k, rk, rk2);
            const V3 s = cvec::sub(v[0], v[k]);
            const Real rs = cvec::dot(s, rk);
            const Real delta = gain * rs / rk2;
            const V3 dv = cvec::scale(rk, delta);
            v[0] = cvec::add(v[0], cvec::scale(dv, wc));
            v[k] = cvec::sub(v[k], cvec::scale(dv, wp));
            if (nvfabs(delta) > tol) moving = true;
        }
    }
#pragma unroll
    for (int k = 0; k < 4; k++)
        if (b.present(k)) shakeStore<Real>(acc, idx[k], k, v[k]);
}

// Accessors over the global arrays for the separate kernels, addressed by atom.
template<class Real> struct ShakeGlobalDisplacements {
    typename CVD<Real>::s4* buf;
    __device__ __forceinline__ typename CV<Real>::v4 load(int atom, int) const {
        Real* tag = nullptr;
        return loadD4(buf[atom], tag);
    }
    __device__ __forceinline__ void store(int atom, int, const typename CV<Real>::v4& v) const {
        buf[atom] = storeD4(v);
    }
};
template<class Real> struct ShakeGlobalVelocities {
    typename CV<Real>::s4* buf;
    __device__ __forceinline__ typename CV<Real>::v4 load(int atom, int) const {
        Real* tag = nullptr;
        return loadC4(buf[atom], tag);
    }
    __device__ __forceinline__ void store(int atom, int, const typename CV<Real>::v4& v) const {
        buf[atom] = storeC4(v);
    }
};

template<class Real>
__global__ void shakeToPositions(int numClusters, Real tol, const float4* __restrict__ oldPos,
        typename CVD<Real>::s4* __restrict__ posDelta, const int4* __restrict__ clusterAtoms,
        const float4* __restrict__ clusterParams, const float4* __restrict__ posqCorrection) {
    const ShakeGlobalDisplacements<Real> acc{posDelta};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numClusters; i += gridDim.x * blockDim.x)
        shakeClusterPositions<Real, true>(clusterAtoms[i], clusterParams[i], tol, oldPos, posqCorrection, acc);
}

// The plugin passes the velocity array (velm) in the parameter named posDelta.
template<class Real>
__global__ void shakeToVelocities(int numClusters, Real tol, const float4* __restrict__ oldPos,
        typename CV<Real>::s4* __restrict__ posDelta, const int4* __restrict__ clusterAtoms,
        const float4* __restrict__ clusterParams, const float4* __restrict__ posqCorrection) {
    const ShakeGlobalVelocities<Real> acc{posDelta};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numClusters; i += gridDim.x * blockDim.x)
        shakeClusterVelocities<Real, true>(clusterAtoms[i], clusterParams[i], tol, oldPos, posqCorrection, acc);
}

}  // namespace NvOpenMM
