// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// -----------------------------------------------------------------------------
// Templated SETTLE + SHAKE constraint kernels.
//
// One body per kernel, instantiated on the compute scalar Real:
//   - float                (single precision)
//   - double               (mixed precision, native fp64)
//   - matx::fltflt (df64)  (mixed precision, two-float ~48-bit; fp32 ops only)
//
// The df64 path exists because the constraint solvers are fp64-ALU-bound on
// consumer GPUs (fp64 at ~1/64 fp32): fltflt does the geometry with fp32 ops at
// near-double accuracy. Storage is unchanged (velm/posDelta are float4 in single,
// double4 in mixed; posq is float4 + optional float4 correction), so the ABI and
// host side are identical -- only the in-kernel arithmetic type changes. A
// mixed-precision position (posq + posqCorrection) is already a normalized
// float-float pair, so it loads into fltflt for ~free.
//
// This file holds the storage and arithmetic helpers and the BAOAB part
// kernels. The SETTLE and SHAKE kernels themselves are in ConstraintKernels.cuh,
// which is included below the helpers it calls.
// -----------------------------------------------------------------------------

#ifndef __MATX_INLINE__
#define __MATX_INLINE__ inline
#endif
#ifndef __MATX_HOST__
#define __MATX_HOST__ __host__
#endif
#ifndef __MATX_DEVICE__
#define __MATX_DEVICE__ __device__
#endif
#include <cuda/std/cmath>   // fltflt.h uses cuda::std::isfinite but doesn't include it
#include <cub/block/block_reduce.cuh>   // the CoM block reductions
#include "fltflt.hpp"
#include "common.cuh"   // MM_FORCE_SCALE + the float3/double3/float4 operators

// Shares the plugin's namespace: these kernels are part of NvOpenMM, and a
// separate nv_constraints namespace only made their call sites longer.
namespace NvOpenMM {

using matx::fltflt;

// --- df64 vector types (float3/float4/double3/double4 are CUDA builtins) -------
struct ff3 { fltflt x, y, z; };
struct ff4 { fltflt x, y, z, w; };
static __device__ __forceinline__ ff3  operator*(ff3 a, fltflt s) { return ff3{a.x*s, a.y*s, a.z*s}; }
static __device__ __forceinline__ ff3& operator*=(ff3& a, fltflt s) { a.x=a.x*s; a.y=a.y*s; a.z=a.z*s; return a; }

// --- compute-type Real -> its vector + storage types -----------------------------
// One trait keyed on the compute scalar. `v3/v4` are the in-register COMPUTE
// vectors (ff3/ff4 for fltflt); `s/s2/s4` are the in-memory STORAGE types
// (velm/posDelta/dt: float* in single, double* in mixed). The storage is a pure
// function of Real -- float->float4, {double,fltflt}->double4 -- which is why the
// kernels need only Real, never a separate storage-vec or `MIXED` template arg.
template<class Real> struct CV;
template<> struct CV<float>  { using v3 = float3;  using v4 = float4;  using s = float;  using s2 = float2;  using s4 = float4;  };
template<> struct CV<double> { using v3 = double3; using v4 = double4; using s = double; using s2 = double2; using s4 = double4; };
template<> struct CV<fltflt> { using v3 = ff3;     using v4 = ff4;     using s = double; using s2 = double2; using s4 = double4; };

// --- scalar math, overloaded on Real ---------------------------------------------
static __device__ __forceinline__ float  nvsqrt(float x)  { return sqrtf(x); }
static __device__ __forceinline__ double nvsqrt(double x) { return sqrt(x); }
static __device__ __forceinline__ fltflt nvsqrt(fltflt x) { return matx::fltflt_sqrt(x); }
static __device__ __forceinline__ float  nvrsqrt(float x)  { return rsqrtf(x); }
static __device__ __forceinline__ double nvrsqrt(double x) { return rsqrt(x); }
// The numerator stays a BARE float -- do not wrap it in fltflt(). A float
// numerator selects fltflt_div(float, fltflt), which is the same algorithm with
// the a.lo terms dropped because a.lo is known to be zero (fltflt.hpp:575).
// fltflt(1.0f) would force the general fltflt/fltflt path and pay for a limb
// that is always 0.
//
// This holds for * and / ONLY. Do NOT extend it to + and -: fltflt_add(fltflt,
// fltflt) uses the FPAN restructuring (Zhang & Aiken SC'25) while the float
// overload stays a Thall-style chain, so they are DIFFERENT algorithms, not the
// same one with zero terms dropped. Baring a float addend is a little faster and
// just as accurate, but it changes the result -- measured at ~6.7k differing
// results per 4.2M random inputs. That is why the `Real(1.0f)-x` forms below
// still wrap their literal.
static __device__ __forceinline__ fltflt nvrsqrt(fltflt x) { return 1.0f/matx::fltflt_sqrt(x); }
static __device__ __forceinline__ float  nvrecip(float x)  { return 1.0f/x; }
static __device__ __forceinline__ double nvrecip(double x) { return 1.0/x; }
static __device__ __forceinline__ fltflt nvrecip(fltflt x) { return 1.0f/x; }   // bare 1.0f: see nvrsqrt
static __device__ __forceinline__ float  nvfabs(float x)  { return fabsf(x); }
static __device__ __forceinline__ double nvfabs(double x) { return fabs(x); }
static __device__ __forceinline__ fltflt nvfabs(fltflt x) { return matx::fltflt_abs(x); }

// --- vector constructors, overloaded on Real -------------------------------------
static __device__ __forceinline__ float3  nvmk3(float x, float y, float z)    { return make_float3(x,y,z); }
static __device__ __forceinline__ double3 nvmk3(double x, double y, double z) { return make_double3(x,y,z); }
static __device__ __forceinline__ ff3     nvmk3(fltflt x, fltflt y, fltflt z) { return ff3{x,y,z}; }

// --- zero vec4 -----------------------------------------------------------------
template<class C4> static __device__ __forceinline__ C4 nvzero4();
template<> __device__ __forceinline__ float4  nvzero4<float4>()  { return make_float4(0.f,0.f,0.f,0.f); }
template<> __device__ __forceinline__ double4 nvzero4<double4>() { return make_double4(0.0,0.0,0.0,0.0); }
template<> __device__ __forceinline__ ff4     nvzero4<ff4>()     { fltflt z(0.0f); return ff4{z,z,z,z}; }

// --- load velm/posDelta storage (S4) into calc vec4 (C4), and store back -------
// Single/double: identity. fltflt: convert double4 components (bit-extraction, no fp64).
static __device__ __forceinline__ float4  loadC4(const float4& s,  float*)  { return s; }
static __device__ __forceinline__ double4 loadC4(const double4& s, double*) { return s; }
static __device__ __forceinline__ ff4     loadC4(const double4& s, fltflt*) {
    return ff4{ fltflt(s.x), fltflt(s.y), fltflt(s.z), fltflt(s.w) };
}
static __device__ __forceinline__ float4  storeC4(float4 v)  { return v; }
static __device__ __forceinline__ double4 storeC4(double4 v) { return v; }
static __device__ __forceinline__ double4 storeC4(ff4 v) {
    return make_double4((double)v.x, (double)v.y, (double)v.z, (double)v.w);
}
// --- delta-buffer storage (posDelta / plannedDisplacement) ---------------------
// These two are the ONLY buffers in the step that nothing outside this file reads.
// plannedDisplacement is ours outright; posDelta is OpenMM-allocated but every OpenMM path that
// reads it clears or overwrites it first (CommonKernels.cpp:446 clears before
// applyConstraints; IntegrationUtilities.cpp:952 raw-copies velm over it), State
// retrieval never touches it, and on our side part2 writes it before part3 reads it
// over the identical `v.w != 0` atom set -- part2's two launches partition ALL atoms
// (CudaLangevinMiddleKernel.cu:463).
//
// So for fltflt they hold the hi/lo pair DIRECTLY (ff4a below) instead of a double,
// and the double round-trip -- 2x F2F.F64.F32 + a DADD per component, on a 1:64-rate
// fp64 pipe -- disappears. ff4a is 32 bytes, the same footprint as the double4 OpenMM
// allocated, so this reinterprets the existing storage rather than resizing it.
// velm CANNOT join them: State retrieval, the barostat, CoM removal and OpenMM's
// velm<->posDelta byte copy all read it as double4.
//
// Measured on an RTX 5090 at bace/dhfr/water_100k NPT mixed: ~2% end-to-end, together
// with the __fdividef divide seed in fltflt.hpp. Per-kernel nsys durations are a poor
// instrument here -- these kernels run ~92% overlapped under the captured graph, so a
// kernel that gets cheaper can show a LONGER duration once the stream it overlapped
// with finishes sooner. Time a whole step instead.
template<class Real> struct CVD { using s4 = typename CV<Real>::s4; };
// ff4a is ff4 at 16-byte alignment. ff4 inherits fltflt's alignas(8), and that alone
// costs part2 6x STG.E.128 -> 2x STG.E.128 + 8x STG.E.64 -- ten narrow stores where the
// double4 it replaces got six wide ones, which shows up as a REGRESSION under the
// co-scheduled graph even though the isolated kernel is faster. Same 32 bytes; every
// element of a 32-byte-strided array off a cudaMalloc base is already 16-aligned, so
// this constrains nothing it did not already satisfy.
struct alignas(16) ff4a { fltflt x, y, z, w; };
template<> struct CVD<fltflt>  { using s4 = ff4a; };

static __device__ __forceinline__ float4  loadD4(const float4& s,  float*)  { return s; }
static __device__ __forceinline__ double4 loadD4(const double4& s, double*) { return s; }
static __device__ __forceinline__ ff4     loadD4(const ff4a& s,    fltflt*) { return ff4{s.x,s.y,s.z,s.w}; }
static __device__ __forceinline__ float4  storeD4(float4 v)  { return v; }
static __device__ __forceinline__ double4 storeD4(double4 v)  { return v; }
static __device__ __forceinline__ ff4a    storeD4(ff4 v)     { return ff4a{v.x,v.y,v.z,v.w}; }

// A read-only read of one element of the displacement or velocity storage through the
// read-only data path (__ldg), for arrays the kernel does not write. The bits are
// copied as they are.
static __device__ __forceinline__ float4 ldgS4(const float4* p) { return __ldg(p); }
static __device__ __forceinline__ double4 ldgS4(const double4* p) {
    const double2 a = __ldg(reinterpret_cast<const double2*>(p)), b = __ldg(reinterpret_cast<const double2*>(p) + 1);
    return make_double4(a.x, a.y, b.x, b.y);
}
static __device__ __forceinline__ ff4a ldgS4(const ff4a* p) {
    ff4a r;
    float4* w = reinterpret_cast<float4*>(&r);
    w[0] = __ldg(reinterpret_cast<const float4*>(p));
    w[1] = __ldg(reinterpret_cast<const float4*>(p) + 1);
    return r;
}

// Inverse mass = 1 / stored velocity .w, computed in Real.
static __device__ __forceinline__ float  invW(const float4& v,  float*)  { return 1.0f/v.w; }
static __device__ __forceinline__ double invW(const double4& v, double*) { return 1.0/v.w; }
static __device__ __forceinline__ fltflt invW(const double4& v, fltflt*) { return 1.0f/fltflt(v.w); }   // bare 1.0f: see nvrsqrt

// --- load a position (posq + optional compensated correction) into calc vec4 ---
template<class Real> static __device__ __forceinline__ typename CV<Real>::v4
loadPos4(const float4* posq, const float4* corr, int i);
template<> __device__ __forceinline__ float4 loadPos4<float>(const float4* posq, const float4*, int i) {
    return posq[i];
}
template<> __device__ __forceinline__ double4 loadPos4<double>(const float4* posq, const float4* corr, int i) {
    float4 p = posq[i], c = corr[i];
    return make_double4(p.x+(double)c.x, p.y+(double)c.y, p.z+(double)c.z, p.w);
}
template<> __device__ __forceinline__ ff4 loadPos4<fltflt>(const float4* posq, const float4* corr, int i) {
    float4 p = posq[i], c = corr[i];
    return ff4{ fltflt(p.x,c.x), fltflt(p.y,c.y), fltflt(p.z,c.z), fltflt(p.w,0.0f) };
}

} // namespace NvOpenMM

#include "ConstraintKernels.cuh"

namespace NvOpenMM {

// ---- Helpers for the Langevin stages ------------------------------------------
// The fixed-point force decode, a vec4 constructor and the position commit, in the
// compute type. The stages themselves follow them.

// Fixed-point force accumulator (mm_long) -> compute type.
template<class Real> static __device__ __forceinline__ Real loadForce(long long f);
template<> __device__ __forceinline__ float  loadForce<float>(long long f)  { return (float) f; }
template<> __device__ __forceinline__ double loadForce<double>(long long f) { return (double) f; }
template<> __device__ __forceinline__ fltflt loadForce<fltflt>(long long f) { return fltflt((double) f); }

// make vec4 in the compute type.
static __device__ __forceinline__ float4  mkC4(float x, float y, float z, float w)     { return make_float4(x,y,z,w); }
static __device__ __forceinline__ double4 mkC4(double x, double y, double z, double w)  { return make_double4(x,y,z,w); }
static __device__ __forceinline__ ff4     mkC4(fltflt x, fltflt y, fltflt z, fltflt w)  { return ff4{x,y,z,w}; }

// Commit a computed position to (posq[, posqCorrection]). For df64 the fltflt
// hi/lo ARE the compensated storage; for double it's the usual Dekker split.
template<class Real> static __device__ __forceinline__ void
storePos(float4* posq, float4* corr, int i, typename CV<Real>::v4 pos);
template<> __device__ __forceinline__ void storePos<float>(float4* posq, float4*, int i, float4 pos) {
    posq[i] = pos;
}
template<> __device__ __forceinline__ void storePos<double>(float4* posq, float4* corr, int i, double4 pos) {
    float4 hi = make_float4((float)pos.x, (float)pos.y, (float)pos.z, (float)pos.w);
    posq[i] = hi;
    corr[i] = make_float4((float)(pos.x-(double)hi.x), (float)(pos.y-(double)hi.y), (float)(pos.z-(double)hi.z), 0.0f);
}
template<> __device__ __forceinline__ void storePos<fltflt>(float4* posq, float4* corr, int i, ff4 pos) {
    posq[i] = make_float4(pos.x.hi, pos.y.hi, pos.z.hi, (float)pos.w);
    corr[i] = make_float4(pos.x.lo, pos.y.lo, pos.z.lo, 0.0f);
}

// ==== Langevin integration stages ====
// One step of the BAOAB splitting of Langevin dynamics (Leimkuhler and Matthews,
// Appl. Math. Res. Express 2013; J. Chem. Phys. 138, 174102), in the arrangement
// where velocities and positions are stored at the same time:
//   stage 1 (B):     v += dt F / m, a full kick; then the velocity constraints.
//   stage 2 (A O A): half a drift, the exact Ornstein-Uhlenbeck update of the
//                    velocity, the second half drift. The drift is collected as a
//                    displacement instead of being added to the position; then the
//                    position constraints correct that displacement.
//   stage 3:         the velocity absorbs the constraint's change of the
//                    displacement, divided by dt, and the position moves by the
//                    corrected displacement.
// Atoms with zero inverse mass (virtual sites, frozen atoms) are skipped by every
// stage and none of their entries is written.
//
// The order and grouping of the operations below is deliberate: the mixed-precision
// results are reproduced bit for bit, and in float-float arithmetic (and in double,
// where products may be fused into sums) the order and the operand types decide the
// rounding. stageFma marks each product that is fused with the sum after it.

// x*y + z. In float and double with a single rounding; in float-float as the product
// followed by the sum (the library rounds every operation explicitly).
static __device__ __forceinline__ float  stageFma(float x, float y, float z)    { return __fmaf_rn(x, y, z); }
static __device__ __forceinline__ double stageFma(double x, double y, double z) { return __fma_rn(x, y, z); }
static __device__ __forceinline__ fltflt stageFma(fltflt x, fltflt y, fltflt z) { return x * y + z; }

// Stage 1 for one atom with mass, given its three fixed-point force words and
// k = dt / 2^32 (the force scale folded into the step): v += (k / m) F.
template<class Real>
static __device__ __forceinline__ void kick_v(typename CV<Real>::v4 &v, long long fx, long long fy, long long fz,
                                              Real k) {
    const Real kOverM = k * v.w;
    v.x = stageFma(kOverM, loadForce<Real>(fx), v.x);
    v.y = stageFma(kOverM, loadForce<Real>(fy), v.y);
    v.z = stageFma(kOverM, loadForce<Real>(fz), v.z);
}

// The kick factor of stage 1: the step size divided by the fixed-point force scale.
template<class Real>
static __device__ __forceinline__ Real kickFactor(Real stepSize) { return stepSize / MM_FORCE_SCALE; }

// Stage 2 for one atom with mass. vel is updated in place by the Ornstein-Uhlenbeck
// step over dt, vel <- decay vel + noise sqrt(1/m) xi, where decay = exp(-gamma dt),
// noise = sqrt(kT (1 - decay^2)) and xi is three standard normal deviates keyed on
// (atom, step, seed). The returned displacement is half a step at the old velocity
// plus half a step at the new one; its w is 0.
template<class Real>
static __device__ __forceinline__ typename CV<Real>::v4
ouDrift(typename CV<Real>::v4 &vel, Real decay, Real noise, Real halfStep, unsigned atom, unsigned step, unsigned seed) {
    float xi[3];
    nvGaussian3(atom, step, seed, xi);
    const Real sigma = noise * nvsqrt(vel.w);
    const typename CV<Real>::v4 old = vel;
    vel.x = stageFma(decay, old.x, sigma * xi[0]);
    vel.y = stageFma(decay, old.y, sigma * xi[1]);
    vel.z = stageFma(decay, old.z, sigma * xi[2]);
    return mkC4(stageFma(halfStep, old.x, halfStep * vel.x),
                stageFma(halfStep, old.y, halfStep * vel.y),
                stageFma(halfStep, old.z, halfStep * vel.z),
                Real(0.0f));
}

// Stage 3 for one atom with mass: moved is the displacement after the position
// constraints, planned the one stage 2 produced, invStep = 1 / dt.
template<class Real>
static __device__ __forceinline__ void absorbConstraint(typename CV<Real>::v4 &vel, typename CV<Real>::v4 &pos,
        const typename CV<Real>::v4 &moved, const typename CV<Real>::v4 &planned, Real invStep) {
    vel.x = stageFma(moved.x - planned.x, invStep, vel.x);
    vel.y = stageFma(moved.y - planned.y, invStep, vel.y);
    vel.z = stageFma(moved.z - planned.z, invStep, vel.z);
    pos.x = pos.x + moved.x;
    pos.y = pos.y + moved.y;
    pos.z = pos.z + moved.z;
}

// The three stage kernels, one thread per atom. Each thread issues all of its
// atom's reads before the mass test, so they are in flight together; arrays the
// kernel does not write are read through the read-only data path.

// Stage 1 over atoms [0, numAtoms). Thread 0 of block 0 also advances the step
// counter, once per launch; stage 2 of the same step keys its noise on the new value.
template<class Real>
__global__ void langevinKick(int numAtoms, int paddedNumAtoms, typename CV<Real>::s4* __restrict__ velm,
                             const long long* __restrict__ force, const typename CV<Real>::s2* __restrict__ dt,
                             unsigned* __restrict__ stepCount) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        stepCount[0] = stepCount[0] + 1u;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms)
        return;
    Real* tag = nullptr;
    const typename CV<Real>::s4 stored = velm[i];
    const long long fx = __ldg(force + i), fy = __ldg(force + i + paddedNumAtoms),
                    fz = __ldg(force + i + 2 * paddedNumAtoms);
    const Real stepSize = Real(__ldg(&dt[0].y));
    typename CV<Real>::v4 v = loadC4(stored, tag);
    if (v.w != 0.0f) {
        kick_v<Real>(v, fx, fy, fz, kickFactor(stepSize));
        velm[i] = storeC4(v);
    }
}

// Stage 2 over atoms [0, numAtoms), or with LIST over atomList[0 .. numAtoms).
// langevinCoefficients holds decay and noise (see ouDrift). Both displacement buffers receive the
// result: posDelta is corrected by the position constraints, plannedDisplacement keeps the
// uncorrected one for stage 3.
template<class Real, bool LIST>
__global__ void langevinDrift(int numAtoms, const int* __restrict__ atomList, typename CV<Real>::s4* __restrict__ velm,
                              typename CVD<Real>::s4* __restrict__ posDelta, typename CVD<Real>::s4* __restrict__ plannedDisplacement,
                              const typename CV<Real>::s* __restrict__ langevinCoefficients,
                              const typename CV<Real>::s2* __restrict__ dt,
                              const unsigned* __restrict__ stepCount, unsigned seed) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= numAtoms)
        return;
    const int i = LIST ? __ldg(atomList + j) : j;
    Real* tag = nullptr;
    typename CV<Real>::v4 v = loadC4(velm[i], tag);
    if (v.w != 0.0f) {
        const Real decay = Real(__ldg(langevinCoefficients)), noise = Real(__ldg(langevinCoefficients + 1));
        const Real halfStep = 0.5f * Real(__ldg(&dt[0].y));
        const unsigned step = __ldg(stepCount);
        const typename CV<Real>::v4 d = ouDrift<Real>(v, decay, noise, halfStep, (unsigned) i, step, seed);
        velm[i] = storeC4(v);
        posDelta[i] = storeD4(d);
        plannedDisplacement[i] = storeD4(d);
    }
}

// Stage 3 over atoms [0, numAtoms).
template<class Real>
__global__ void langevinCommit(int numAtoms, float4* __restrict__ posq, typename CV<Real>::s4* __restrict__ velm,
                               const typename CVD<Real>::s4* __restrict__ posDelta,
                               const typename CVD<Real>::s4* __restrict__ plannedDisplacement,
                               const typename CV<Real>::s2* __restrict__ dt, float4* __restrict__ posqCorrection) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms)
        return;
    const Real invStep = nvrecip(Real(__ldg(&dt[0].y)));
    Real* tag = nullptr;
    const typename CV<Real>::s4 stored = velm[i];
    const typename CVD<Real>::s4 movedRaw = ldgS4(posDelta + i), plannedRaw = ldgS4(plannedDisplacement + i);
    typename CV<Real>::v4 p = loadPos4<Real>(posq, posqCorrection, i);
    typename CV<Real>::v4 v = loadC4(stored, tag);
    // Computed for every atom and stored only for one with mass: the double build
    // needs 40 registers this way, 48 with the arithmetic inside the test.
    absorbConstraint<Real>(v, p, loadD4(movedRaw, tag), loadD4(plannedRaw, tag), invStep);
    if (v.w != 0.0f) {
        velm[i] = storeC4(v);
        storePos<Real>(posq, posqCorrection, i, p);
    }
}

// ==== Centre-of-mass motion removal ====
// Removes the total momentum from the velocities: v_i -= P / M for every atom, with
// P the sum of m_i v_i and M the total mass. Two launches of the same grid (64
// threads per block): the first leaves one partial momentum per block, the second
// sums the partials in every block and subtracts. The sums are in float, and the
// fixed block size and grid fix their order. S is the velocity storage scalar
// (float or double); no compute type is involved. The loops are not unrolled: the
// launchers' grid gives each thread at most one atom, and unrolling the loops
// costs registers.
static constexpr int COM_BLOCK = 64;

// Block b's partial momentum to blockMomentum[b]; its w is 0.
template<class S>
__global__ void comPartialMomentum(int numAtoms,
        const typename CV<S>::s4* __restrict__ velm, float4* __restrict__ blockMomentum) {
    using Reduce = cub::BlockReduce<float4, COM_BLOCK>;
    __shared__ typename Reduce::TempStorage scratch;
    float4 mom = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int stride = COM_BLOCK * gridDim.x;
#pragma unroll 1
    for (int i = blockIdx.x * COM_BLOCK + threadIdx.x; i < numAtoms; i += stride) {
        const typename CV<S>::s4 v = velm[i];
        if (v.w != 0) {
            const S mass = S(1) / v.w;
            if constexpr (std::is_same<S, float>::value) {
                mom.x = __fmaf_rn(v.x, mass, mom.x);
                mom.y = __fmaf_rn(v.y, mass, mom.y);
                mom.z = __fmaf_rn(v.z, mass, mom.z);
            } else {
                mom.x = mom.x + (float) (v.x * mass);
                mom.y = mom.y + (float) (v.y * mass);
                mom.z = mom.z + (float) (v.z * mass);
            }
        }
    }
    const float4 total = Reduce(scratch).Sum(mom);
    if (threadIdx.x == 0)
        blockMomentum[blockIdx.x] = total;
}

// Sums the gridDim.x partial momenta and subtracts inverseTotalMass times the sum from
// every atom's velocity, massless atoms included. w is not written.
template<class S>
__global__ void comSubtract(int numAtoms, typename CV<S>::s4* __restrict__ velm,
        const float4* __restrict__ blockMomentum, float inverseTotalMass) {
    using Reduce = cub::BlockReduce<float4, COM_BLOCK>;
    __shared__ typename Reduce::TempStorage scratch;
    __shared__ float4 totalShared;
    float4 part = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll 1
    for (int b = threadIdx.x; b < (int) gridDim.x; b += COM_BLOCK)
        part += blockMomentum[b];
    const float4 sum = Reduce(scratch).Sum(part);
    if (threadIdx.x == 0)
        totalShared = sum;
    __syncthreads();
    const float4 total = totalShared;
    const int stride = COM_BLOCK * gridDim.x;
    if constexpr (std::is_same<S, float>::value) {
        const float nx = -total.x, ny = -total.y, nz = -total.z;
#pragma unroll 1
        for (int i = blockIdx.x * COM_BLOCK + threadIdx.x; i < numAtoms; i += stride) {
            float4 v = velm[i];
            v.x = __fmaf_rn(nx, inverseTotalMass, v.x);
            v.y = __fmaf_rn(ny, inverseTotalMass, v.y);
            v.z = __fmaf_rn(nz, inverseTotalMass, v.z);
            velm[i].x = v.x;
            velm[i].y = v.y;
            velm[i].z = v.z;
        }
    } else {
        const double dx = (double) __fmul_rn(inverseTotalMass, total.x), dy = (double) __fmul_rn(inverseTotalMass, total.y),
                     dz = (double) __fmul_rn(inverseTotalMass, total.z);
#pragma unroll 1
        for (int i = blockIdx.x * COM_BLOCK + threadIdx.x; i < numAtoms; i += stride) {
            double4 v = velm[i];
            velm[i].x = v.x - dx;
            velm[i].y = v.y - dy;
            velm[i].z = v.z - dz;
        }
    }
}

} // namespace NvOpenMM
