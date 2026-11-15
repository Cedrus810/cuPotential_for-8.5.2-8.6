// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// -----------------------------------------------------------------------------
// SETTLE spread over three lanes of a warp, for the fused one-launch step.
//
// Included by FusedChains.cuh after Constraints.cuh (with ConstraintKernels.cuh)
// and after FusedChains.cuh's shuffle, round_velocity and same_bits helpers.
//
// Lane layout, fixed by the caller: lanes 3g, 3g+1, 3g+2 of a warp hold atoms A, B,
// C of water g (A is the atom with the two equal bond lengths). On each lane the
// caller sets base = 3g, k = 0, 1 or 2 for A, B or C, k1 = (k+1) mod 3 and
// k2 = (k+2) mod 3. Lanes that carry no water (lanes 30 and 31, and slots past the
// last water) are given water 0's atoms; they run the same code and their results
// are discarded.
//
// The arithmetic is that of settleToVelocities and settleToPositions in
// ConstraintKernels.cuh (Miyamoto and Kollman, J. Comput. Chem. 13, 952 (1992)),
// with the same helpers and the same operation order for every value, so in mixed
// precision each lane's result equals the separate kernel's bit for bit. Values move
// between lanes only by shuffles, which copy bits exactly.
//
// The work is divided so that each lane computes about a third of it and keeps few
// values live (the fused kernel runs at a cap of 96 registers):
//
// * Velocities. Lane k owns bond k, from its atom to the next one (A-B, B-C, C-A):
//   its unit vector, the relative velocity along it, and the cosine of the angle at
//   its own atom. These scalars are exchanged; the determinant of the impulse system
//   is computed on every lane, and each lane solves only for its own bond's impulse.
//   The impulse of the previous bond is then taken from the previous lane.
//
// * Positions, up to the frame axes. Lane k holds Cartesian component k (x, y or z)
//   of every vector: the old positions relative to A, the centre of mass, the
//   displaced positions and the three frame axes. A component of a cross product
//   needs the other two components of its inputs, which are taken from the
//   neighbouring lanes.
// * Positions, the frame. Lane k then owns frame axis k (x', y', z'): it gathers that
//   axis, normalises it, and computes the frame coordinates along it. The frame
//   solve's inputs are gathered onto every lane, and the solve runs on every lane.
// * Positions, the result. Each lane places its own atom: it fetches the unit axes
//   and the centre of mass, rotates its atom's fitted coordinates into the lab frame
//   and forms its new displacement.
//
// Every shuffle is executed by all 32 lanes: the code has no branch around one, and
// per-lane roles are chosen with conditional expressions on values.
// -----------------------------------------------------------------------------

namespace NvOpenMM {
namespace settle_lanes {

// The caller fills base, k, k1 and k2 (see the header).
template<class Real> struct Lanes {
    int base, k, k1, k2;
};

// Terms that depend only on the masses and the two bond lengths, when every water
// has the same ones: masses, 1/(mA mB mC), mA + mB + mC, 1/(mA + mB + mC), and the
// parts of the canonical triangle that need a square root (ra, rb). Built by
// consts_kernel. rc, 2 rc and d2^2 are not in the table: they cost a few
// multiplications from the water's lengths, and every table entry stays live in
// registers from the start of the fused step to its use.
template<class Real> struct Consts {
    Real mass[3];
    Real invProduct, total, invTotal;
    Real ra, rb;
};

// The value x of the given lane, per component.
template<class V3> static __device__ __forceinline__ V3 lane3(const V3& x, int lane) {
    return nvmk3(shuffle(x.x, lane), shuffle(x.y, lane), shuffle(x.z, lane));
}

// Per-lane choices. They are written on the plain floats of each value so that the
// compiler emits select instructions; a choice between whole float-float structs
// can become a branch, which makes the lanes of different roles run one after the
// other.
static __device__ __forceinline__ float choose(bool p, float a, float b) { return p ? a : b; }
static __device__ __forceinline__ double choose(bool p, double a, double b) { return p ? a : b; }
static __device__ __forceinline__ fltflt choose(bool p, fltflt a, fltflt b) {
    return fltflt(p ? a.hi : b.hi, p ? a.lo : b.lo);
}
template<class V3> static __device__ __forceinline__ V3 choose(bool p, const V3& a, const V3& b) {
    return nvmk3(choose(p, a.x, b.x), choose(p, a.y, b.y), choose(p, a.z, b.z));
}

// a, b or c for i = 0, 1, 2.
template<class T> static __device__ __forceinline__ T pick(int i, const T& a, const T& b, const T& c) {
    return choose(i == 0, a, choose(i == 1, b, c));
}

// Component i of v.
template<class V3> static __device__ __forceinline__ auto comp(const V3& v, int i) -> decltype(v.x) {
    return pick(i, v.x, v.y, v.z);
}

// A vector spread over the water's lanes, lane j holding component j, gathered whole.
template<class Real> static __device__ __forceinline__ typename CV<Real>::v3 gather(const Lanes<Real>& L, Real c) {
    return nvmk3(shuffle(c, L.base), shuffle(c, L.base + 1), shuffle(c, L.base + 2));
}

// Exchange between the three lanes of a water: on entry each lane holds a vector
// (one row of a 3x3 matrix); on return own, next and prev are component k of the
// vectors of lanes k, k1 and k2. Two shuffles per component: lane i sends its
// component i+1 to lane i+1 and its component i+2 to lane i+2.
template<class Real> struct Transposed {
    Real own, next, prev;
    // The value that came from lane j (0, 1 or 2 of the water).
    __device__ __forceinline__ Real from(const Lanes<Real>& L, int j) const {
        return choose(j == L.k, own, choose(j == L.k1, next, prev));
    }
};
template<class Real>
static __device__ __forceinline__ Transposed<Real> transpose(const Lanes<Real>& L, Real r0, Real r1, Real r2) {
    Transposed<Real> t;
    t.own = pick(L.k, r0, r1, r2);
    t.prev = shuffle(pick(L.k1, r0, r1, r2), L.base + L.k2);
    t.next = shuffle(pick(L.k2, r0, r1, r2), L.base + L.k1);
    return t;
}

// Component k of the cross product u x v, from the lanes' components of u and v:
// u[k1] v[k2] - u[k2] v[k1].
template<class Real>
static __device__ __forceinline__ Real crossComp(const Lanes<Real>& L, Real u, Real v) {
    const Real uNext = shuffle(u, L.base + L.k1), uPrev = shuffle(u, L.base + L.k2);
    const Real vNext = shuffle(v, L.base + L.k1), vPrev = shuffle(v, L.base + L.k2);
    return uNext * vPrev - uPrev * vNext;
}

// The velocity constraint. pos: this lane's atom position; v: its velocity, with the
// inverse mass in v.w. On return v.x, v.y, v.z are the constrained velocity.
template<class Real, bool CONSTS>
__device__ __forceinline__ void vel(const Lanes<Real>& L, const Consts<Real>& K,
                                    const typename CV<Real>::v4& pos, typename CV<Real>::v4& v) {
    using V3 = typename CV<Real>::v3;
    const int lA = L.base, lB = L.base + 1, lC = L.base + 2;
    const Real one = Real(1.0f);

    // This lane's bond (to the next atom): unit vector and relative velocity along it.
    const V3 vk = cvec::xyz(v);
    const V3 e = settleBondUnit(cvec::xyz(pos), lane3(cvec::xyz(pos), L.base + L.k1));
    const Real u = cvec::dot(cvec::sub(lane3(vk, L.base + L.k1), vk), e);
    // The angle at this lane's atom lies between its bond and the previous lane's:
    // cA = -(eAB.eCA), cB = -(eAB.eBC), cC = -(eBC.eCA).
    const bool isA = L.k == 0;
    Real c;
    {
        const V3 ePrev = lane3(e, L.base + L.k2);
        c = -cvec::dot(choose(isA, e, ePrev), choose(isA, ePrev, e));
    }
    const Real s = one - c * c;

    const Real cA = shuffle(c, lA), cB = shuffle(c, lB), cC = shuffle(c, lC);
    const Real sA = shuffle(s, lA), sB = shuffle(s, lB), sC = shuffle(s, lC);
    Real mA, mB, mC, invProduct, total;
    if constexpr (CONSTS) {
        mA = K.mass[0]; mB = K.mass[1]; mC = K.mass[2];
        invProduct = K.invProduct; total = K.total;
    } else {
        const Real m = nvrecip(v.w);
        mA = shuffle(m, lA); mB = shuffle(m, lB); mC = shuffle(m, lC);
        invProduct = nvrecip(mA * mB * mC);
        total = mA + mB + mC;
    }
    const Real det = settleDeterminant(sA, sB, sC, cA, cB, cC, mA, mB, mC, invProduct);

    // This lane's impulse, (T1 uCA + T2 uBC + T3 uAB) / det, with the terms of
    // settleImpulses for its bond:
    //   A-B: T1 = kAB,   T2 = kBC,   T3 = diag(sC; mA, mB)
    //   B-C: T1 = kCA,   T2 = diag(sA; mB, mC), T3 = kBC' (kBC with mA, mC swapped)
    //   C-A: T1 = diag(sB; mA, mC), T2 = kCA, T3 = kAB
    // Two cofactors per lane: first = kAB, kCA, kCA; second = kBC, kBC', kAB.
    const int k = L.k;
    const Real first = settleCofactor(pick(k, cB, cA, cA), pick(k, cC, cB, cB), pick(k, mA, mC, mC),
                                      pick(k, cA, cC, cC), mB, pick(k, mC, mA, mA));
    const Real second = settleCofactor(pick(k, cA, cA, cB), pick(k, cC, cC, cC), pick(k, mB, mB, mA),
                                       pick(k, cB, cB, cA), pick(k, mC, mA, mB), pick(k, mA, mC, mC));
    const Real diag = settleDiagonal(pick(k, sC, sA, sB), pick(k, mA, mB, mA), pick(k, mB, mC, mC), invProduct, total);
    const Real t1 = choose(k == 2, diag, first);
    const Real t2 = pick(k, second, diag, first);
    const Real t3 = choose(k == 0, diag, second);
    const Real tOwn = (t1 * shuffle(u, lC) + t2 * shuffle(u, lB) + t3 * shuffle(u, lA)) / det;
    const Real tPrev = shuffle(tOwn, L.base + L.k2);

    const V3 nv = settleKick(vk, tOwn, e, tPrev, lane3(e, L.base + L.k2), v.w);
    v.x = nv.x; v.y = nv.y; v.z = nv.z;
}

// The position constraint. apos: this lane's atom position; xp: its displacement of
// this step, updated in place; w: its inverse mass; params: the water's lengths
// (x: A-B and A-C, y: B-C).
template<class Real, bool CONSTS>
__device__ __forceinline__ void pos(const Lanes<Real>& L, const Consts<Real>& K,
                                    const typename CV<Real>::v4& apos, typename CV<Real>::v4& xp, Real w, float2 params) {
    using V3 = typename CV<Real>::v3;
    const int lA = L.base, lB = L.base + 1, lC = L.base + 2;
    const int k = L.k;

    Real mA, mB, mC, invTotal;
    if constexpr (CONSTS) {
        mA = K.mass[0]; mB = K.mass[1]; mC = K.mass[2];
        invTotal = K.invTotal;
    } else {
        const Real m = nvrecip(w);
        mA = shuffle(m, lA); mB = shuffle(m, lB); mC = shuffle(m, lC);
        invTotal = nvrecip(mA + mB + mC);
    }

    // Component k of the three atoms' old positions and displacements.
    Real b0, c0, a1, b1, c1, com;
    {
        const Transposed<Real> p = transpose(L, apos.x, apos.y, apos.z);
        const Transposed<Real> d = transpose(L, xp.x, xp.y, xp.z);
        const Real pA = p.from(L, 0);
        b0 = p.from(L, 1) - pA;               // old B and C relative to old A
        c0 = p.from(L, 2) - pA;
        const Real dA = d.from(L, 0);
        const Real movedB = b0 + d.from(L, 1);   // displaced, relative to old A
        const Real movedC = c0 + d.from(L, 2);
        com = (dA * mA + movedB * mB + movedC * mC) * invTotal;
        a1 = dA - com;                        // displaced, relative to the centre of mass
        b1 = movedB - com;
        c1 = movedC - com;
    }

    // Component k of the frame axes: z' = b0 x c0, x' = a1 x z', y' = z' x x'. Lane k
    // then gathers axis k whole (its component j is on lane j) and normalises it.
    V3 axis;
    {
        const Real ez = crossComp(L, b0, c0);
        const Real ex = crossComp(L, a1, ez);
        const Real ey = crossComp(L, ez, ex);
        const Transposed<Real> t = transpose(L, ex, ey, ez);
        axis = cvec::unit(nvmk3(t.from(L, 0), t.from(L, 1), t.from(L, 2)));
    }

    // Frame coordinates along this lane's axis: on the x' and y' lanes those of b0, c0,
    // b1, c1; on the z' lane those of a1, b1, c1.
    Real f1, f2, f3, f4;
    {
        const bool isZ = k == 2;
        f1 = cvec::dot(axis, choose(isZ, gather(L, a1), gather(L, b0)));
        f2 = cvec::dot(axis, choose(isZ, gather(L, b1), gather(L, c0)));
        f3 = cvec::dot(axis, choose(isZ, gather(L, c1), gather(L, b1)));
        f4 = cvec::dot(axis, gather(L, c1));
    }

    SettleShape<Real> sh;
    if constexpr (CONSTS) {
        sh = settleShapeLengths<Real>(params.y);
        sh.ra = K.ra; sh.rb = K.rb;
    } else {
        sh = settleShape<Real>(mB, mC, invTotal, params.x, params.y);
    }
    // The frame solve, on every lane. Its first stage needs only the z' coordinates;
    // the in-plane coordinates are fetched after it.
    const Real zA = shuffle(f1, lC), zB = shuffle(f2, lC), zC = shuffle(f3, lC);
    const Real fz = pick(k, zA, zB, zC);   // this atom's z', which the fit keeps
    SettleFit<Real> f = settleTilt<Real>(sh.ra, sh.rb, sh.rc, sh.rc2, sh.q, zA, zB, zC);
    settleTurn<Real>(f, shuffle(f1, lA), shuffle(f1, lB), shuffle(f2, lA), shuffle(f2, lB),
                     shuffle(f3, lA), shuffle(f3, lB), shuffle(f4, lA), shuffle(f4, lB));

    // This atom of the fitted triangle, rotated by theta about z':
    //   A: (-(aY sin), aY cos);  B: (bX cos - bY sin, bX sin + bY cos);
    //   C: (-(bX cos) - cY sin, -(bX sin) + cY cos).
    const bool isA = k == 0;
    const Real yOwn = pick(k, f.aY, f.bY, f.cY);
    const Real ys = yOwn * f.rotSin, yc = yOwn * f.rotCos;
    const Real bc = f.bX * f.rotCos, bs = f.bX * f.rotSin;
    const Real fx = choose(isA, -ys, choose(k == 1, bc, -bc) - ys);
    const Real fy = choose(isA, yc, choose(k == 1, bs, -bs) + yc);
    // Lab component c: x'_c fx + y'_c fy + z'_c fz, the axes' components fetched from
    // their lanes one component at a time.
    auto labComp = [&](Real e) { return shuffle(e, lA) * fx + shuffle(e, lB) * fy + shuffle(e, lC) * fz; };
    const V3 lab = nvmk3(labComp(axis.x), labComp(axis.y), labComp(axis.z));

    // New displacement: centre of mass plus the constrained position, minus the old
    // position relative to A (nothing for A itself).
    const V3 p = cvec::xyz(apos);
    const V3 rel = cvec::sub(p, lane3(p, lA));
    const V3 placed = cvec::add(gather(L, com), lab);
    const V3 nd = choose(isA, placed, cvec::sub(placed, rel));
    xp.x = nd.x; xp.y = nd.y; xp.z = nd.z;
}

// Builds the constants table from water 0 and checks that every water shares its
// masses and lengths; *ok is cleared if any does not, or if the table would not
// reproduce what the lanes compute (see below).
template<class Real>
__global__ void consts_kernel(Consts<Real>* out, int* ok, const typename CV<Real>::s4* velm,
                              const int4* atoms, const float2* params, int n) {
    const int4 a0 = atoms[0];
    const float2 len0 = params[0];
    const auto w0 = velm[a0.x].w, w1 = velm[a0.y].w, w2 = velm[a0.z].w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int4 a = atoms[i];
        const float2 len = params[i];
        if (!same_bits(velm[a.x].w, w0) || !same_bits(velm[a.y].w, w1) || !same_bits(velm[a.z].w, w2) ||
            !same_bits(len.x, len0.x) || !same_bits(len.y, len0.y))
            *ok = 0;
    }
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // The velocity lanes see each inverse mass after one storage round trip, the
    // position lanes after two. The table holds one mass per atom for both, so it is
    // valid only when 1/w is the same bits for both forms (and w is not zero).
    const decltype(w0) stored[3] = {w0, w1, w2};
    Consts<Real> K;
    bool valid = true;
#pragma unroll
    for (int j = 0; j < 3; j++) {
        const Real once = round_velocity(Real(stored[j]));
        const Real twice = round_velocity(once);
        K.mass[j] = nvrecip(once);
        if (!same_bits(nvrecip(twice), K.mass[j]) || stored[j] == 0) valid = false;
    }
    K.invProduct = nvrecip(K.mass[0] * K.mass[1] * K.mass[2]);
    K.total = K.mass[0] + K.mass[1] + K.mass[2];
    K.invTotal = nvrecip(K.mass[0] + K.mass[1] + K.mass[2]);
    const SettleShape<Real> sh = settleShape<Real>(K.mass[1], K.mass[2], K.invTotal, len0.x, len0.y);
    K.ra = sh.ra; K.rb = sh.rb;
    *out = K;
    if (!valid) *ok = 0;
}

}  // namespace settle_lanes
}  // namespace NvOpenMM
