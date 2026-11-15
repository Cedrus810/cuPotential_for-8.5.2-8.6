// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// -----------------------------------------------------------------------------
// The single- and mixed-precision Langevin-middle step in one launch: part1, the
// velocity constraints, part2, the position constraints and part3, for every SETTLE
// cluster, SHAKE cluster and unconstrained atom.
//
// A cluster's chain touches only its own atoms and the noise is keyed on the atom,
// so one block can run a whole chain. The arithmetic is shared with the separate
// kernels in Constraints.cuh: mixed precision is bitwise identical to them, single
// precision may differ by FP32 contraction. Compared with the separate kernels:
//   - one launch, with stages synchronised only within a warp;
//   - intermediate values stay in registers or shared memory, and the rounding of
//     each velocity store/load is reproduced in registers (round_velocity);
//   - posDelta and plannedDisplacement are not written (see the note in Constraints.cuh).
//
// Layout (built by CudaIntegrateLangevinMiddleStepKernel::buildFusedLayout):
//   - SHAKE blocks, one atom per lane: clusters sorted by size so each warp holds
//     one size and no cluster straddles warps, letting stages sync by __syncwarp.
//     Each cluster is solved on its first atom's lane.
//   - SETTLE blocks: 10 waters per warp on lanes 0-29, one atom per lane, trading
//     values by shuffles.
//   - unconstrained atoms: lanes 30-31 of the SETTLE warps, or spare SHAKE lanes
//     and extra blocks when those run out.
// -----------------------------------------------------------------------------

#ifndef NV_FUSED_CHAINS_CUH_
#define NV_FUSED_CHAINS_CUH_

namespace NvOpenMM {

// fltflt(double(x)) -- a fltflt stored to a double and loaded back -- in fp32 and
// integer instructions, bit for bit. For a normal, non-power-of-two hi with a small
// lo, the round trip only rounds lo to a multiple of the double's ulp at hi, which
// adding and subtracting a scaled constant does exactly. Every other input takes
// the fp64 round trip.
static __device__ __forceinline__ float ff_add_rn(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
static __device__ __forceinline__ float ff_sub_rn(float a, float b) { float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
static __device__ __noinline__ fltflt ff_double_rt_slow(fltflt x) { return fltflt((double)x); }
static __device__ __forceinline__ fltflt ff_double_rt(fltflt x) {
    const unsigned hb = __float_as_uint(x.hi), lb = __float_as_uint(x.lo);
    const unsigned eb = hb & 0x7f800000u;
    const float two_m = __uint_as_float(eb - (29u << 23));
    const float C = __uint_as_float((eb - (29u << 23)) | (lb & 0x80000000u));
    float lo = fabsf(x.lo) < two_m ? ff_sub_rn(ff_add_rn(x.lo, C), C) : x.lo;
    const float half = __uint_as_float(eb - (24u << 23));   // ulp(hi) / 2
    if (eb < (60u << 23) || eb > (200u << 23) || (hb & 0x7fffffu) == 0 || !(fabsf(lo) < half))
        return ff_double_rt_slow(x);
    if (lo == 0.0f) lo = __uint_as_float(hb & 0x80000000u);
    return fltflt(x.hi, lo);
}
static __device__ __forceinline__ ff4 ff_double_rt4(ff4 v) {
    return ff4{ff_double_rt(v.x), ff_double_rt(v.y), ff_double_rt(v.z), ff_double_rt(v.w)};
}

// Preserve the velocity-storage boundary between fused stages. In single
// precision storage and computation are both float; mixed retains its exact
// float-float -> double -> float-float rounding. Mixed computed in double (on GPUs
// with fast FP64, see chooseMixedCompute in CudaLangevinMiddleKernel.cu) stores
// what it computes, so there is nothing to round.
static __device__ __forceinline__ float round_velocity(float x) { return x; }
static __device__ __forceinline__ double round_velocity(double x) { return x; }
static __device__ __forceinline__ fltflt round_velocity(fltflt x) { return ff_double_rt(x); }
static __device__ __forceinline__ float4 round_velocity(float4 x) { return x; }
static __device__ __forceinline__ double4 round_velocity(double4 x) { return x; }
static __device__ __forceinline__ ff4 round_velocity(ff4 x) { return ff_double_rt4(x); }
static __device__ __forceinline__ bool same_bits(float x, float y) { return __float_as_uint(x) == __float_as_uint(y); }
static __device__ __forceinline__ bool same_bits(double x, double y) { return __double_as_longlong(x) == __double_as_longlong(y); }
static __device__ __forceinline__ bool same_bits(fltflt x, fltflt y) { return same_bits(x.hi, y.hi) && same_bits(x.lo, y.lo); }
static __device__ __forceinline__ float shuffle(float x, int lane) { return __shfl_sync(0xffffffffu, x, lane); }
static __device__ __forceinline__ double shuffle(double x, int lane) { return __shfl_sync(0xffffffffu, x, lane); }
static __device__ __forceinline__ fltflt shuffle(fltflt x, int lane) {
    return fltflt(__shfl_sync(0xffffffffu, x.hi, lane), __shfl_sync(0xffffffffu, x.lo, lane));
}

// SHAKE's per-atom vectors for the fused kernel: the block's shared copies, addressed
// by lane (lanes l0.. hold the cluster's atoms x, y, z, w). The velocity store repeats
// the round trip through velm's storage that the separate kernels take.
template<class Real> struct ShakeVelocityShared {
    typename CVD<Real>::s4* sv; int l0;
    __device__ __forceinline__ typename CV<Real>::v4 load(int, int slot) const { Real* tag = nullptr; return loadD4(sv[l0 + slot], tag); }
    __device__ __forceinline__ void store(int, int slot, const typename CV<Real>::v4 &v) const { sv[l0 + slot] = storeD4(round_velocity(v)); }
};
template<class Real> struct ShakeDeltaShared {
    typename CVD<Real>::s4* sd; int l0;
    __device__ __forceinline__ typename CV<Real>::v4 load(int, int slot) const { Real* tag = nullptr; return loadD4(sd[l0 + slot], tag); }
    __device__ __forceinline__ void store(int, int slot, const typename CV<Real>::v4 &v) const { sd[l0 + slot] = storeD4(v); }
};

// namespace settle_lanes, the lane-parallel SETTLE of the fused step, is in
// SettleLanes.cuh. It calls the helpers above, so it is included here.
} // namespace NvOpenMM
#include "SettleLanes.cuh"
namespace NvOpenMM {

// Timing trace of the fused kernel, for layout studies; off unless the build
// defines NV_FUSED_TRACE=1. When on and ChainArgs::trace is set, thread 0 of each
// block writes, per work unit u, trace[u] = {start, end} in globaltimer
// nanoseconds, the end taken when the block has finished the unit, and per block
// b, trace[units + b] = {first unit's start, end of its last unit}, with .x's top
// 16 bits replaced by the SM id. The host side is in launchFusedT.
#ifndef NV_FUSED_TRACE
#define NV_FUSED_TRACE 0
#endif
#if NV_FUSED_TRACE
static __device__ __forceinline__ unsigned long long fused_trace_clock() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#endif

// Lanes per block. 10 waters per warp, and SHAKE clusters packed by warp.
static constexpr int LANES = 128;
static constexpr int SETTLE_PER_WARP = 10;
static constexpr int SETTLE_PER_BLOCK = LANES / 32 * SETTLE_PER_WARP;
static constexpr int SETTLE_FREE_PER_BLOCK = LANES / 32 * 2;
// The register cap, as a minimum of resident blocks per SM, tuned per architecture.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define NV_FUSED_MIN_BLOCKS 4
#else
#define NV_FUSED_MIN_BLOCKS 5
#endif
// The same for a launch with SETTLE units only (no SHAKE clusters and no packed free
// atoms). Without the SHAKE path the kernel needs fewer registers: at 6 blocks (a cap
// of 80) it spills nothing on sm_89, sm_90 and sm_120, in either precision and with or
// without the SETTLE table. On water_100k, 6 blocks measured faster than 7 and 8 (caps
// of 72 and 64) on both the RTX 6000 Ada and the RTX 5090.
#define NV_FUSED_MIN_BLOCKS_SETTLE 6

template<class Real> struct ChainArgs {
    int nSettle, paddedNumAtoms;
    Real tol;
    const long long *force;
    float4 *posq;
    typename CV<Real>::s4 *velm;
    const int4 *settleAtoms, *shakeAtoms;
    const float2 *settleParams;
    const float4 *shakeParams;
    float4 *posqCorrection;
    const typename CV<Real>::s *langevinCoefficients;
    const typename CV<Real>::s2 *dt;
    unsigned *stepCount;
    unsigned *doneBlocks;    // zero between launches; see the step counter below
    unsigned *nextUnit;      // work units handed out past the first wave; zero between launches too
    unsigned seed;
    const int *freeAtoms;    // atoms in no cluster, on the SETTLE warps' spare lanes
    int nFree;               // 0 when they are packed with the SHAKE clusters instead
    int packedBlocks;        // SHAKE (and packed free) blocks, before the SETTLE blocks
    const int *laneAtom;     // per packed lane: its atom, or -1
    const int *laneCluster;  // per packed lane: the SHAKE cluster it solves, or -1
    int settleChunks;        // SETTLE_PER_BLOCK clusters each
    const settle_lanes::Consts<Real> *settleConsts;   // when the kernel runs with CONSTS
    // SHAKE units 0 .. nHeavy-1 each get an SM of their own (see heavy_start); 0: off.
    int nHeavy;
    unsigned *heavyNext;     // heavy units claimed this launch; zero between launches
    unsigned *launchTag;     // heavy_start's tag for this launch; never 0
    unsigned long long *smWords;   // NV_FUSED_SM_SLOTS rank words, then as many state words
#if NV_FUSED_TRACE
    ulonglong2 *trace;       // see NV_FUSED_TRACE above; null on launches not traced
#endif
};

// ==== Langevin: the fused step's per-atom functions ====
// P2Args<Real> holds part 2's per-launch constants; the block marked "Langevin part 2
// constants" in fusedLangevinMiddleStep builds it. part1_v, part2_v and part3_v are
// parts 1, 2 and 3 for one atom, as settle_block and shake_block call them (kick_v,
// in Constraints.cuh, is part 1 for shake_block, which reads the force itself).
template<class Real> struct P2Args {
    Real decay;        // exp(-gamma dt): the velocity's share kept over the step
    Real noise;        // sqrt(kT (1 - decay^2)): scaled by sqrt(1/m) per atom
    Real halfStep;     // dt / 2
    unsigned step;     // the step counter value the noise is keyed on
    unsigned seed;
};

template<class Real>
static __device__ __forceinline__ void part1_v(const ChainArgs<Real> &c, int atom, typename CV<Real>::v4 &v) {
    if (v.w != 0.0f)
        kick_v<Real>(v, __ldg(c.force + atom), __ldg(c.force + atom + c.paddedNumAtoms),
                     __ldg(c.force + atom + 2 * c.paddedNumAtoms), kickFactor(Real(__ldg(&c.dt[0].y))));
}

template<class Real>
static __device__ __forceinline__ typename CV<Real>::v4 part2_v(const P2Args<Real> &a, int atom, typename CV<Real>::v4 &v) {
    return ouDrift<Real>(v, a.decay, a.noise, a.halfStep, (unsigned) atom, a.step, a.seed);
}

template<class Real>
static __device__ __forceinline__ void part3_v(const ChainArgs<Real> &c, int atom, typename CV<Real>::v4 v, const typename CV<Real>::v4 &delta,
                                               const typename CV<Real>::v4 &drift, const typename CV<Real>::v4 &pos0) {
    typename CV<Real>::v4 pos = pos0;
    absorbConstraint<Real>(v, pos, delta, drift, nvrecip(Real(__ldg(&c.dt[0].y))));
    c.velm[atom] = storeC4(v);
    storePos<Real>(c.posq, c.posqCorrection, atom, pos);
}

// A SETTLE warp's chain (see the header). Lanes 30 and 31 carry free atoms, whose
// constraint stages leave their values as they were.
template <class Real, bool CONSTS>
static __device__ __forceinline__ void settle_block(const ChainArgs<Real> &c, const P2Args<Real> &a, int b) {
    using namespace settle_lanes;
    Real *tag = nullptr;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane / 3;
    const int wg = b * (LANES / 32) + warp;
    const int cl = wg * SETTLE_PER_WARP + g;
    const bool active = lane < 30 && cl < c.nSettle;
    Lanes<Real> L;
    L.k = lane < 30 ? lane % 3 : 0;
    L.base = lane < 30 ? 3 * g : 0;
    L.k1 = L.k == 2 ? 0 : L.k + 1;
    L.k2 = L.k == 0 ? 2 : L.k - 1;
    const int clr = active ? cl : 0;
    const int4 at = c.settleAtoms[clr];
    const int fi = wg * 2 + (lane - 30);
    const bool free_atom = lane >= 30 && fi < c.nFree;
    // Lanes with no work run the shuffles on cluster 0's atom (in range, never stored).
    const int atom = free_atom ? c.freeAtoms[fi] : L.k == 0 ? at.x : L.k == 1 ? at.y : at.z;

    Consts<Real> K;
    if constexpr (CONSTS) K = *c.settleConsts;
    // part1
    typename CV<Real>::v4 v = loadC4(c.velm[atom], tag);
    part1_v(c, atom, v);
    if (v.w != 0.0f) v = round_velocity(v);   // stored by part1, loaded by settleToVelocities
    const typename CV<Real>::v4 apos = loadPos4<Real>(c.posq, c.posqCorrection, atom);
    // settleToVelocities
    const typename CV<Real>::v4 v_part1 = v;
    vel<Real, CONSTS>(L, K, apos, v);
    v = round_velocity(v);   // stored by settleToVelocities, loaded by part2
    if (free_atom) v = v_part1;   // no constraint: part2 loads what part1 stored
    // part2
    typename CV<Real>::v4 delta = nvzero4<typename CV<Real>::v4>();
    if (v.w != 0.0f) {
        delta = part2_v(a, atom, v);
        v = round_velocity(v);   // stored by part2, loaded by settleToPositions and part3
    }
    const typename CV<Real>::v4 drift = delta;
    // settleToPositions
    pos<Real, CONSTS>(L, K, apos, delta, v.w, c.settleParams[clr]);
    if (free_atom) delta = drift;   // no constraint: part3 reads what part2 stored
    // part3
    if ((active || free_atom) && v.w != 0.0f)
        part3_v(c, atom, v, delta, drift, apos);
}

// A SHAKE (and packed free) block's chain, each lane's velocity and delta in
// shared memory between the stages.
template<class Real>
static __device__ __forceinline__ void shake_block(const ChainArgs<Real> &c, const P2Args<Real> &a, int atom, int own) {
    __shared__ typename CVD<Real>::s4 sv[LANES], sd[LANES];
    Real *tag = nullptr;
    const int t = threadIdx.x;
    // part1
    typename CV<Real>::v4 v = nvzero4<typename CV<Real>::v4>();
    if (atom != -1) {
        // The velocity and the three force words are read together, before any
        // arithmetic; the force is read whether or not the atom has mass (the index is
        // in range), which keeps its reads out of the mass test, where each was issued
        // only after the velocity had arrived and been converted. The force and dt
        // are never written by the kernel: __ldg.
        const typename CV<Real>::s4 vRaw = c.velm[atom];
        const long long fx = __ldg(c.force + atom), fy = __ldg(c.force + atom + c.paddedNumAtoms),
                        fz = __ldg(c.force + atom + 2 * c.paddedNumAtoms);
        v = loadC4(vRaw, tag);
        if (v.w != 0.0f) kick_v<Real>(v, fx, fy, fz, Real(__ldg(&c.dt[0].y)) / MM_FORCE_SCALE);   // part1_v
        // Stored by part1 only when the atom has mass; loaded by shakeToVelocities
        // either way, through a double.
        if (v.w != 0.0f) v = round_velocity(v);
        sv[t] = storeD4(v);
    }
    __syncwarp();
    if (own != -1) shakeClusterVelocities<Real>(__ldg(c.shakeAtoms + own), __ldg(c.shakeParams + own), c.tol, c.posq, c.posqCorrection,
                                                ShakeVelocityShared<Real>{sv, t});
    __syncwarp();
    if constexpr (std::is_same<Real, float>::value) {
        // Single precision has registers to spare: v and the old position stay in
        // registers across the position constraint.
        typename CV<Real>::v4 drift = nvzero4<typename CV<Real>::v4>();
        typename CV<Real>::v4 pos0 = nvzero4<typename CV<Real>::v4>();
        if (atom != -1) {
            v = loadD4(sv[t], tag);
            if (v.w != 0.0f) {
                drift = part2_v(a, atom, v);
                v = round_velocity(v);   // stored by part2, loaded by part3
                sd[t] = storeD4(drift);
            }
            pos0 = loadPos4<Real>(c.posq, c.posqCorrection, atom);
        }
        __syncwarp();
        if (own != -1) shakeClusterPositions<Real>(__ldg(c.shakeAtoms + own), __ldg(c.shakeParams + own), c.tol, c.posq, c.posqCorrection,
                                                   ShakeDeltaShared<Real>{sd, t});
        __syncwarp();
        if (atom != -1 && v.w != 0.0f)
            part3_v(c, atom, v, loadD4(sd[t], tag), drift, pos0);
    } else {
        // part2
        typename CV<Real>::v4 drift = nvzero4<typename CV<Real>::v4>();
        if (atom != -1) {
            v = loadD4(sv[t], tag);
            if (v.w != 0.0f) {
                drift = part2_v(a, atom, v);
                v = round_velocity(v);   // stored by part2, loaded by part3
                sd[t] = storeD4(drift);
            }
            // v and the old position are needed again only in part3: v waits in sv (the
            // velocity stage is done with it; the displacement storage copies exactly) and
            // the position is loaded again then, so neither is held in registers across
            // the position constraint.
            sv[t] = storeD4(v);
        }
        __syncwarp();
        if (own != -1) shakeClusterPositions<Real>(__ldg(c.shakeAtoms + own), __ldg(c.shakeParams + own), c.tol, c.posq, c.posqCorrection,
                                                   ShakeDeltaShared<Real>{sd, t});
        __syncwarp();
        // part3
        if (atom != -1) {
            v = loadD4(sv[t], tag);
            if (v.w != 0.0f)
                part3_v(c, atom, v, loadD4(sd[t], tag), drift, loadPos4<Real>(c.posq, c.posqCorrection, atom));
        }
    }
    // The block may take another unit next (the grid-stride loop in
    // fusedLangevinMiddleStep), which writes sv again: keep that behind every lane's
    // reads of this unit.
    __syncwarp();
}

// The next work unit for this block, past the first wave: the block's threads wait
// until the unit they ran is done, thread 0 takes a unit and the block reads it.
// With heavy units (see heavy_start), nextUnit hands out units from nHeavy on, and
// once those run out the block takes any heavy unit no SM's first block claimed
// (when fewer SMs held blocks than there are heavy units); otherwise from the grid
// size on.
template<class Real, bool HEAVY>
static __device__ __forceinline__ int next_unit(const ChainArgs<Real> &c, int &slot, int units) {
    __syncthreads();                       // the unit is done before the broadcast slot is reused
    if (threadIdx.x == 0) {
        const bool heavy = HEAVY && c.nHeavy > 0;
        int n = (heavy ? c.nHeavy : (int) gridDim.x) + (int) atomicAdd(c.nextUnit, 1u);
        if (heavy && n >= units) {
            const int h = (int) atomicAdd(c.heavyNext, 1u);
            if (h < c.nHeavy) n = h;
        }
        slot = n;
    }
    __syncthreads();
    return slot;
}

// Heavy SHAKE units on SMs of their own. The SHAKE units with the largest clusters
// run longest, and run longer still when SETTLE blocks share their SM (on dhfr and
// tyk2 they set the kernel's end). With c.nHeavy > 0, the first block to start on
// each SM claims one of SHAKE units 0 .. nHeavy-1 (c.heavyNext); if it got one,
// the SM's other blocks exit at once, and the block takes ordinary units after its
// heavy one. Every other unit is handed out by nextUnit from nHeavy on. Which
// block runs which unit changes no value.
//
// Per SM slot (the SM id modulo NV_FUSED_SM_SLOTS), a rank word counts the blocks
// that started on it this launch and a state word says whether its first block
// took a heavy unit; both carry the launch's tag (c.launchTag) in their top 32
// bits, so a word left by an earlier launch reads as empty and nothing needs
// resetting. The tag is a count of the launches, not the step number: the step
// number goes back when a checkpoint is loaded, and a word left by the earlier
// launch of that step would then read as this launch's. A block that is not first
// on its SM waits for the state word; the first block is already running, so the
// wait ends. Returns the block's first unit, or -1 to exit.
#define NV_FUSED_SM_SLOTS 512
template<class Real>
static __device__ __forceinline__ int heavy_start(const ChainArgs<Real> &c, unsigned launchTag, int &slotOut) {
    if (threadIdx.x == 0) {
        unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        unsigned long long *rankWord = c.smWords + (sm % NV_FUSED_SM_SLOTS);
        unsigned long long *stateWord = rankWord + NV_FUSED_SM_SLOTS;
        const unsigned long long tag = (unsigned long long) launchTag << 32;
        unsigned long long old = *(volatile unsigned long long *) rankWord, seen;
        unsigned rank;
        for (;;) {
            rank = (old & ~0xffffffffull) == tag ? (unsigned) old : 0u;
            seen = atomicCAS(rankWord, old, tag | (rank + 1u));
            if (seen == old) break;
            old = seen;
        }
        int first;
        if (rank == 0) {
            const int h = (int) atomicAdd(c.heavyNext, 1u);
            const bool heavy = h < c.nHeavy;
            atomicExch(stateWord, tag | (heavy ? 2ull : 1ull));
            first = heavy ? h : c.nHeavy + (int) atomicAdd(c.nextUnit, 1u);
        } else {
            unsigned long long w;
            do w = *(volatile unsigned long long *) stateWord; while ((w & ~0xffffffffull) != tag);
            first = (w & 3ull) == 2ull ? -1 : c.nHeavy + (int) atomicAdd(c.nextUnit, 1u);
        }
        slotOut = first;
    }
    __syncthreads();
    return slotOut;
}

// The step counter. part1 bumps it and part2 keys its noise on the bumped value;
// in one launch, every block reads the old value and uses it plus one, and the
// last block to finish stores it, so no block can read a value another has already
// bumped. doneBlocks counts finished blocks and is reset by the last one, so it is
// zero again at the next launch, eager or replayed from a graph.
// PACKED: whether the launch has SHAKE (packed) units; when false, c.packedBlocks is 0
// and the SHAKE path is not compiled in.
// HEAVY: whether the heavy SHAKE units (heavy_start) are compiled in: with SHAKE
// units, in mixed precision only (see buildFusedLayout).
template<class Real, bool PACKED> static constexpr bool fusedHasHeavy = PACKED && !std::is_same<Real, float>::value;
template <class Real, bool CONSTS, bool PACKED>
__global__ void __launch_bounds__(LANES, PACKED ? NV_FUSED_MIN_BLOCKS : NV_FUSED_MIN_BLOCKS_SETTLE)
fusedLangevinMiddleStep(const __grid_constant__ ChainArgs<Real> c) {
    const unsigned step = c.stepCount[0] + 1u;
    // ==== BEGIN Langevin part 2 constants ====
    const P2Args<Real> a{Real(c.langevinCoefficients[0]), Real(c.langevinCoefficients[1]), 0.5f * Real(c.dt[0].y), step, c.seed};
    // ==== END Langevin part 2 constants ====
    // The work units are the SHAKE (and packed free-atom) blocks, then the SETTLE
    // chunks. The grid is what the GPU holds resident (buildFusedLayout): block b
    // starts on unit b, and later units go to whichever block finishes first
    // (nextUnit), balancing the slower SHAKE units on any GPU.
    __shared__ int s_unit;
    const int units = c.packedBlocks + c.settleChunks;
    // nextUnit hands out units in increasing order, so a block takes all its SHAKE
    // units before its SETTLE ones. Two loops, one per kind, keep the registers of
    // one kind's chain from being allocated alongside the other's.
    constexpr bool HEAVY = fusedHasHeavy<Real, PACKED>;
    constexpr int EXITS = 0x7fffffff;   // u of a block that exits at once
    int u = blockIdx.x;
    if (HEAVY && c.nHeavy > 0) {
        u = heavy_start(c, threadIdx.x == 0 ? *c.launchTag : 0u, s_unit);
        if (u < 0) u = EXITS;
    }
#if NV_FUSED_TRACE
    // A unit's end is stamped after a barrier, when every thread of the block is done.
    unsigned long long tBlock = 0, tUnit = 0;
    if (c.trace && threadIdx.x == 0) tBlock = fused_trace_clock();
    if (u == EXITS) tBlock = 0;   // a block that exits at once: no record
#define NV_TRACE_BEGIN() do { if (c.trace && threadIdx.x == 0) tUnit = fused_trace_clock(); } while (0)
#define NV_TRACE_END() do { __syncthreads(); if (c.trace && threadIdx.x == 0) c.trace[u] = make_ulonglong2(tUnit, fused_trace_clock()); } while (0)
#else
#define NV_TRACE_BEGIN() do {} while (0)
#define NV_TRACE_END() do {} while (0)
#endif
    // The two loops, one per kind of unit.
#define NV_UNIT_LOOPS() \
    while (PACKED && u < c.packedBlocks) { \
        NV_TRACE_BEGIN(); \
        shake_block(c, a, __ldg(c.laneAtom + u * LANES + threadIdx.x), __ldg(c.laneCluster + u * LANES + threadIdx.x)); \
        NV_TRACE_END(); \
        u = next_unit<Real, HEAVY>(c, s_unit, units); \
    } \
    while ((!HEAVY || u >= c.packedBlocks) && u < units) { \
        NV_TRACE_BEGIN(); \
        settle_block<Real, CONSTS>(c, a, u - c.packedBlocks); \
        NV_TRACE_END(); \
        u = next_unit<Real, HEAVY>(c, s_unit, units); \
    }
    if constexpr (HEAVY) {
        // The outer loop only repeats when next_unit hands a late heavy unit to a
        // block already in the SETTLE loop.
        while (u < units) {
            NV_UNIT_LOOPS();
        }
    } else {
        NV_UNIT_LOOPS();
    }
#undef NV_UNIT_LOOPS
#undef NV_TRACE_BEGIN
#undef NV_TRACE_END
#if NV_FUSED_TRACE
    if (c.trace && threadIdx.x == 0) {
        unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        c.trace[units + blockIdx.x] = make_ulonglong2((tBlock & 0xffffffffffffull) | ((unsigned long long) sm << 48),
                                                      fused_trace_clock());
    }
#endif
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(c.doneBlocks, 1u) == gridDim.x - 1) {
            c.stepCount[0] = step;
            *c.doneBlocks = 0;
            *c.nextUnit = 0;
            if (HEAVY && c.nHeavy > 0) {
                *c.heavyNext = 0;
                // Every block has read the tag by now; no other block writes it.
                const unsigned next = *c.launchTag + 1u;
                *c.launchTag = next != 0u ? next : 1u;
            }
        }
    }
}

}  // namespace NvOpenMM

#endif  // NV_FUSED_CHAINS_CUH_
