// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "CudaLangevinMiddleKernel.hpp"
#include "common/CudaGraphEnv.hpp"
#include "common/ClearMigrationGraph.hpp"
#include "CudaContext.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/common/ComputeVectorTypes.h"
#include "openmm/common/IntegrationUtilities.h"
#include "openmm/common/NonbondedUtilities.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/MonteCarloBarostat.h"
#include "nvopenmm/MonteCarloBarostat.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/CustomExternalForce.h"
#include "nvopenmm/NonbondedForce.h"
#include "nvopenmm/internal/MonteCarloBarostatImpl.h"
#include <typeinfo>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <map>
#include <mutex>
#include <ostream>
#include <random>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// AOT (ahead-of-time, nvcc-compiled) kernels for the Nvidia LangevinMiddle
// integrator. Compiled into libOpenMMNvidiaCUDA.so at build time (CMake globs
// src/*.cu, CUDA_ARCHITECTURES = the build's sm target). No OpenMM runtime
// compiler (NVRTC) is involved for these kernels.
//
// Structure:
//   - Precision-independent helpers (Philox RNG, step counter) at file scope.
//   - The precision-dependent bodies (Part1/2/3, CoM, SETTLE, SHAKE) live in
//     Constraints.cuh as one templated kernel each, instantiated on the compute
//     type: single -> float, mixed -> float-float (df64, two fp32 emulating fp64).
//   - File-local launchers, templated on that same compute type, launch on a
//     caller-supplied stream (so the whole step is capturable into a CUDA graph).
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

#include "common.cuh"

// ---- precision-independent: counter-based Philox RNG + step counter ----------

static __device__ void nvPhilox4x32(unsigned int c0, unsigned int c1, unsigned int c2, unsigned int c3,
                                    unsigned int k0, unsigned int k1, unsigned int out[4]) {
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        unsigned int hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u*c0;
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u*c2;
        unsigned int n0 = hi1 ^ c1 ^ k0, n1 = lo1, n2 = hi0 ^ c3 ^ k1, n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

// Three independent standard-normal deviates for one atom at a given step.
static __device__ void nvGaussian3(unsigned int atom, unsigned int step, unsigned int seed, float g[3]) {
    unsigned int r[4];
    nvPhilox4x32(atom, step, 0u, 0u, seed, 0x1234567u, r);
    const float norm = 2.3283064365386963e-10f; // 1/2^32
    float u1 = ((float) r[0] + 0.5f)*norm;
    float u2 = ((float) r[1] + 0.5f)*norm;
    float u3 = ((float) r[2] + 0.5f)*norm;
    float u4 = ((float) r[3] + 0.5f)*norm;
    float R1 = sqrtf(-2.0f*logf(u1)), t1 = 6.2831853071795862f*u2;
    float R2 = sqrtf(-2.0f*logf(u3)), t2 = 6.2831853071795862f*u4;
    g[0] = R1*cosf(t1);
    g[1] = R1*sinf(t1);
    g[2] = R2*cosf(t2);
}

// All per-step device kernels (BAOAB Part1/2/3, fused CoM, SETTLE, SHAKE) as one
// templated body each -- instantiated on the compute type: single -> float,
// mixed -> float-float (df64). (The precision-independent Philox RNG + step
// counter above stay at file scope.)
#include "Constraints.cuh"
#include "ConstraintClusters.hpp"
#include "FusedChains.cuh"

// ---- launch helpers ----------------------------------------------------------

static const int NV_BLOCK = 64;    // element-wise kernels (Part1/2/3, settle/shake)

static inline int gridFor(int n, int block) {
    int g = (n + block - 1) / block;
    return g < 1 ? 1 : g;
}

// ---- launchers -------------------------------------------------------------
//
// These were extern "C" in a separate translation unit, purely because the host
// side was a .cpp and g++ cannot parse <<<>>>. That exported ten symbols from the
// shared library for what is entirely private, so the host side is a .cu now and
// they are file-local.
//
// Each is TEMPLATED on the compute type -- single -> float, mixed -> matx::fltflt
// -- rather than branching on a runtime flag, so there is one body per launcher
// instead of two. The pointer types the two instantiations disagree on come from
// NvOpenMM::CV<Real>, the same trait the kernels use to declare those very
// parameters; hand-duplicating that mapping was the only reason a launcher needed
// to know about precision at all. The runtime flag is bound to a type in exactly
// two places now: launchStep, and the CoM block in execute().
//
// posqCorrection is a float4* in BOTH instantiations -- single's loadPos4/storePos
// take it and ignore it, and launchStep passes null there -- so it needs no
// special case either.
//
// The void* survive: OpenMM's ComputeArray hands out untyped device addresses, so
// the cast has to happen somewhere, and here is where the type is known.

// How velm/posDelta/plannedDisplacement (s4), dt (s2) and langevinCoefficients (s) are stored in MEMORY for
// a given compute type: float* in single, double* in mixed.
template<class Real> using NvS4 = typename NvOpenMM::CV<Real>::s4;
// posDelta/plannedDisplacement only: for fltflt these hold the hi/lo pair directly rather than a
// double, skipping the fp64 round-trip. Same 32-byte stride, so it reinterprets the
// buffer OpenMM already allocated. See the CVD note in Constraints.cuh for why these
// two buffers are safe to relayout and velm is not.
template<class Real> using NvS4D = typename NvOpenMM::CVD<Real>::s4;
template<class Real> using NvS2 = typename NvOpenMM::CV<Real>::s2;
template<class Real> using NvS  = typename NvOpenMM::CV<Real>::s;

// ==== BEGIN Langevin launchers ====
// The launchers of the three integration stages and of the centre-of-mass removal
// (kernels in Constraints.cuh). Each launches on `stream` and returns at once; the
// pointers are device addresses of the buffers named, typed here from Real.
template<class Real>
static void nvlmi_part1(cudaStream_t stream, int numAtoms, int paddedNumAtoms,
                        void* velm, const void* force, const void* dt, void* stepCount) {
    NvOpenMM::langevinKick<Real><<<gridFor(numAtoms, NV_BLOCK), NV_BLOCK, 0, stream>>>(
        numAtoms, paddedNumAtoms, (NvS4<Real>*) velm, (const long long*) force, (const NvS2<Real>*) dt,
        (unsigned int*) stepCount);
}

// With atomIdx null, stage 2 runs over atoms [0, numAtoms); otherwise over the
// numAtoms atom indices in atomIdx.
template<class Real>
static void nvlmi_part2(cudaStream_t stream, int numAtoms, const void* atomIdx,
                        void* velm, void* posDelta, void* plannedDisplacement, const void* langevinCoefficients, const void* dt,
                        const void* stepCount, unsigned int seed) {
    if (atomIdx != nullptr)
        NvOpenMM::langevinDrift<Real, true><<<gridFor(numAtoms, NV_BLOCK), NV_BLOCK, 0, stream>>>(
            numAtoms, (const int*) atomIdx, (NvS4<Real>*) velm, (NvS4D<Real>*) posDelta, (NvS4D<Real>*) plannedDisplacement,
            (const NvS<Real>*) langevinCoefficients, (const NvS2<Real>*) dt, (const unsigned int*) stepCount, seed);
    else
        NvOpenMM::langevinDrift<Real, false><<<gridFor(numAtoms, NV_BLOCK), NV_BLOCK, 0, stream>>>(
            numAtoms, nullptr, (NvS4<Real>*) velm, (NvS4D<Real>*) posDelta, (NvS4D<Real>*) plannedDisplacement,
            (const NvS<Real>*) langevinCoefficients, (const NvS2<Real>*) dt, (const unsigned int*) stepCount, seed);
}

template<class Real>
static void nvlmi_part3(cudaStream_t stream, int numAtoms,
                        void* posq, void* velm, void* posDelta, void* plannedDisplacement, const void* dt, void* posqCorrection) {
    NvOpenMM::langevinCommit<Real><<<gridFor(numAtoms, NV_BLOCK), NV_BLOCK, 0, stream>>>(
        numAtoms, (float4*) posq, (NvS4<Real>*) velm, (const NvS4D<Real>*) posDelta, (const NvS4D<Real>*) plannedDisplacement,
        (const NvS2<Real>*) dt, (float4*) posqCorrection);
}

// The two centre-of-mass launches must use the same grid: the second sums one
// partial momentum per block of the first. Only the velocity storage type matters.
template<class Real>
static void nvlmi_calc_com(cudaStream_t stream, int numAtoms, const void* velm, void* comMomentumPartials) {
    NvOpenMM::comPartialMomentum<NvS<Real>><<<gridFor(numAtoms, NvOpenMM::COM_BLOCK), NvOpenMM::COM_BLOCK, 0, stream>>>(
        numAtoms, (const NvS4<Real>*) velm, (float4*) comMomentumPartials);
}

template<class Real>
static void nvlmi_remove_com(cudaStream_t stream, int numAtoms, void* velm,
                             const void* comMomentumPartials, float inverseTotalMass) {
    NvOpenMM::comSubtract<NvS<Real>><<<gridFor(numAtoms, NvOpenMM::COM_BLOCK), NvOpenMM::COM_BLOCK, 0, stream>>>(
        numAtoms, (NvS4<Real>*) velm, (const float4*) comMomentumPartials, inverseTotalMass);
}
// ==== END Langevin launchers ====

// No tolerance, posDelta or clusterParams: SETTLE is closed-form and its velocity
// pass works off the current geometry (see the kernel).
template<class Real>
static void nvlmi_settle_velocities(cudaStream_t stream, int numClusters,
                             const void* oldPos, void* velm,
                             const void* clusterAtoms, const void* posqCorrection) {
    NvOpenMM::settleToVelocities<Real><<<gridFor(numClusters, NV_BLOCK), NV_BLOCK, 0, stream>>>(
        numClusters, (const float4*) oldPos, (NvS4<Real>*) velm,
        (const int4*) clusterAtoms, (const float4*) posqCorrection);
}

template<class Real>
static void nvlmi_shake_velocities(cudaStream_t stream, int numClusters, double tol,
                            const void* oldPos, void* velm,
                            const void* clusterAtoms, const void* clusterParams, const void* posqCorrection) {
    // velm is bound to the kernel's posDelta slot (SHAKE-vel operates on velocities).
    NvOpenMM::shakeToVelocities<Real><<<gridFor(numClusters, NV_BLOCK), NV_BLOCK, 0, stream>>>(
        numClusters, Real(tol), (const float4*) oldPos, (NvS4<Real>*) velm,
        (const int4*) clusterAtoms, (const float4*) clusterParams, (const float4*) posqCorrection);
}

// `velm` supplies the inverse masses, not velocities. No tolerance (closed-form).
template<class Real>
static void nvlmi_settle_positions(cudaStream_t stream, int numClusters,
                            const void* oldPos, void* posDelta, const void* velm,
                            const void* clusterAtoms, const void* clusterParams, const void* posqCorrection) {
    NvOpenMM::settleToPositions<Real><<<gridFor(numClusters, NV_BLOCK), NV_BLOCK, 0, stream>>>(
        numClusters, (const float4*) oldPos, (NvS4D<Real>*) posDelta, (const NvS4<Real>*) velm,
        (const int4*) clusterAtoms, (const float2*) clusterParams, (const float4*) posqCorrection);
}

template<class Real>
static void nvlmi_shake_positions(cudaStream_t stream, int numClusters, double tol,
                           const void* oldPos, void* posDelta,
                           const void* clusterAtoms, const void* clusterParams, const void* posqCorrection) {
    NvOpenMM::shakeToPositions<Real><<<gridFor(numClusters, NV_BLOCK), NV_BLOCK, 0, stream>>>(
        numClusters, Real(tol), (const float4*) oldPos, (NvS4D<Real>*) posDelta,
        (const int4*) clusterAtoms, (const float4*) clusterParams, (const float4*) posqCorrection);
}


// Definitions live in the plugin's namespace, so our NonbondedForce et al.
// win unqualified lookup over OpenMM's same-named classes rather than
// being ambiguous with them.
namespace NvOpenMM {
using namespace std;

// Boltzmann constant in OpenMM units (kJ/mol/K); matches OpenMM's BOLTZ = RGAS/KILO.
static const double NV_BOLTZ = 0.00831446261815324;
// Steps to run eagerly before capturing the per-step graph (lets OpenMM settle:
// first-run allocations, an initial atom reorder).
static const int NV_GRAPH_WARMUP = 25;

// Fetch an OpenMM/our ComputeArray's device address as a raw void* for the launchers.
static inline void* dptr(OpenMM::CudaContext& cu, OpenMM::ArrayInterface& a) {
    return (void*) (uintptr_t) cu.unwrap(a).getDevicePointer();
}

CudaIntegrateLangevinMiddleStepKernel::~CudaIntegrateLangevinMiddleStepKernel() {
    if (graphExec != nullptr)
        cudaGraphExecDestroy(graphExec);
    if (graph != nullptr)
        cudaGraphDestroy(graph);
    if (ownStream != nullptr)
        cudaStreamDestroy(ownStream);
    if (forceDoneEvent != nullptr)
        cudaEventDestroy(forceDoneEvent);
    if (graphDoneEvent != nullptr)
        cudaEventDestroy(graphDoneEvent);
    if (shakeStream != nullptr)
        cudaStreamDestroy(shakeStream);
    if (constraintFork != nullptr)
        cudaEventDestroy(constraintFork);
    if (constraintJoin != nullptr)
        cudaEventDestroy(constraintJoin);
}

// Sort the System's constraints into SETTLE and SHAKE clusters
// (ConstraintClusters.hpp) and build our own device arrays from them, so the
// whole step can run on our capturable stream. numLeftoverConstraints counts
// constraints neither solver takes; initialize() rejects those Systems.
void CudaIntegrateLangevinMiddleStepKernel::buildConstraintClusters(const OpenMM::System& system) {
    const ConstraintClusters found = findConstraintClusters(system);
    const vector<OpenMM::mm_int4>& settleA = found.settleAtoms;
    const vector<OpenMM::mm_float2>& settleP = found.settleParams;
    const vector<OpenMM::mm_int4>& shakeA = found.shakeAtoms;
    const vector<OpenMM::mm_float4>& shakeP = found.shakeParams;
    numLeftoverConstraints = found.numLeftoverConstraints;

    numSettle = (int) settleA.size();
    numShake = (int) shakeA.size();
    if (numSettle > 0) {
        settleAtoms.initialize<OpenMM::mm_int4>(cc, numSettle, "nvSettleAtoms");
        settleParams.initialize<OpenMM::mm_float2>(cc, numSettle, "nvSettleParams");
        settleAtoms.upload(settleA);
        settleParams.upload(settleP);
    }
    if (numShake > 0) {
        shakeAtoms.initialize<OpenMM::mm_int4>(cc, numShake, "nvShakeAtoms");
        shakeParams.initialize<OpenMM::mm_float4>(cc, numShake, "nvShakeParams");
        shakeAtoms.upload(shakeA);
        shakeParams.upload(shakeP);
    }

    buildFusedLayout(system, settleA, shakeA, shakeP);

    // Atom partition for the forked part2 (only meaningful when both solvers exist,
    // which is exactly when launchStep forks). part2[i] reads only velm[i], so an
    // atom belongs to whichever velocity solver wrote it; unconstrained atoms need
    // only part1 and ride with the SETTLE side rather than paying for a third launch.
    if (numSettle > 0 && numShake > 0) {
        vector<bool> ownedByShake(cc.getNumAtoms(), false);
        for (int i = 0; i < (int) shakeA.size(); i++) {
            ownedByShake[shakeA[i].x] = true;
            ownedByShake[shakeA[i].y] = true;
            if (shakeA[i].z != -1) ownedByShake[shakeA[i].z] = true;
            if (shakeA[i].w != -1) ownedByShake[shakeA[i].w] = true;
        }
        vector<int> shakeIdx, otherIdx;
        for (int i = 0; i < cc.getNumAtoms(); i++)
            (ownedByShake[i] ? shakeIdx : otherIdx).push_back(i);
        numPart2Shake = (int) shakeIdx.size();
        numPart2Other = (int) otherIdx.size();
        part2ShakeIdx.initialize<int>(cc, numPart2Shake, "nvPart2ShakeIdx");
        part2OtherIdx.initialize<int>(cc, numPart2Other, "nvPart2OtherIdx");
        part2ShakeIdx.upload(shakeIdx);
        part2OtherIdx.upload(otherIdx);
    }
}

// FMA throughput probe for chooseMixedCompute. Each thread runs eight independent
// chains of fused multiply-adds, enough independent work that the time is set by the
// GPU's FMA rate for T and not by the latency of one FMA. The sum is stored only if it
// is -1, which it never is: the store keeps the compiler from removing the work.
template<class T>
__global__ void fmaRateProbe(T* sink, int iters) {
    T a[8];
#pragma unroll
    for (int k = 0; k < 8; k++)
        a[k] = T(threadIdx.x + k) * T(1e-3);
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int k = 0; k < 8; k++)
            a[k] = fma(a[k], T(0.999), T(1e-4));
    }
    T sum = 0;
#pragma unroll
    for (int k = 0; k < 8; k++)
        sum += a[k];
    if (sum == T(-1))
        sink[0] = sum;
}

// Milliseconds one launch of fmaRateProbe<T> takes: the fastest of three launches,
// after one to warm up. Negative if a CUDA call failed.
template<class T>
static float fmaRateProbeMs(cudaStream_t stream, int blocks, void* sink) {
    cudaEvent_t start = nullptr, stop = nullptr;
    float best = -1.0f;
    if (cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess) {
        for (int r = 0; r < 4; r++) {
            cudaEventRecord(start, stream);
            fmaRateProbe<T><<<blocks, 256, 0, stream>>>((T*) sink, 2048);
            cudaEventRecord(stop, stream);
            float ms = 0.0f;
            if (cudaEventSynchronize(stop) != cudaSuccess || cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) {
                best = -1.0f;
                break;
            }
            if (r > 0 && (best < 0.0f || ms < best))
                best = ms;
        }
    }
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    return best;
}

// FP32 time over FP64 time of the same FMA work on the current device, or 0 if the
// probe could not run. About 2 on GPUs with fast FP64, 32 to 64 elsewhere.
static double measuredDoubleRate(int device) {
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaStream_t stream = nullptr;
    void* sink = nullptr;
    double ratio = 0.0;
    if (sms > 0 && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
            cudaMalloc(&sink, sizeof(double)) == cudaSuccess) {
        const float t32 = fmaRateProbeMs<float>(stream, 8 * sms, sink);
        const float t64 = fmaRateProbeMs<double>(stream, 8 * sms, sink);
        if (t32 > 0.0f && t64 > 0.0f)
            ratio = t64 / t32;
    }
    if (sink) cudaFree(sink);
    if (stream) cudaStreamDestroy(stream);
    cudaGetLastError();   // a failed probe leaves no error behind for later calls
    return ratio;
}

// The compute type of mixed precision: float-float (matx::fltflt) or native double.
// Storage is the same for both (positions as float4 posq plus a float4 correction,
// velocities as double4; posDelta and plannedDisplacement are 32 bytes per atom, float-float
// pairs or double4), so only the arithmetic inside the kernels changes.
//
// A float-float add is 20 FP32 instructions and a multiply 7 (fltflt.hpp). That wins
// where FP64 runs at 1/64 of the FP32 rate (sm_86, sm_89, sm_103 B300, sm_120). On GPUs
// with FP32:FP64 = 2:1 (A100 sm_80, H100/H200 sm_90, B200/GB200 sm_100; CUDA
// Programming Guide, "Compute Capabilities", table 30) a double operation costs two
// FP32 issue slots, less than its emulation, so native double is used there.
// Double is also more accurate, so its results are not bit-identical to float-float's.
//
// Double is chosen when two tests agree. First, the driver's FP32:FP64 ratio
// (cudaDevAttrSingleToDoublePrecisionPerfRatio) is at most 2; elsewhere the choice is
// float-float and nothing is measured. Second, a short probe (fmaRateProbe, a few
// milliseconds once per device and process) measures FP64 FMA work at most 3 times
// slower than the same FP32 work. The probe is there because some products of a fast
// FP64 architecture have their FP64 rate cut far below 1/2 (the H20, an sm_90 part, is
// listed at 1 TFLOPS FP64 against 44 TFLOPS FP32), and the documentation does not say
// whether the driver's ratio describes the product or its architecture.
//
// CUPO_PLUGIN_MIXED_COMPUTE=fltflt or =double overrides the choice, for testing; unset
// or "auto" chooses as above.
static bool chooseMixedCompute(int device) {
    if (const char* env = getenv("CUPO_PLUGIN_MIXED_COMPUTE")) {
        const std::string v(env);
        if (v == "fltflt") return false;
        if (v == "double") return true;
        if (v != "auto" && !v.empty())
            throw OpenMM::OpenMMException("CUPO_PLUGIN_MIXED_COMPUTE must be auto, fltflt or double");
    }
    int ratio = 0;
    if (cudaDeviceGetAttribute(&ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device) != cudaSuccess ||
            ratio < 1 || ratio > 2)
        return false;
    // One probe per device and process, so every Context on a device makes the same
    // choice.
    static std::mutex mutex;
    static std::map<int, bool> chosen;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = chosen.find(device);
    if (it == chosen.end()) {
        const double measured = measuredDoubleRate(device);
        it = chosen.emplace(device, measured > 0.0 && measured <= 3.0).first;
    }
    return it->second;
}

// Builds fusedLangevinMiddleStep's lane layout (see FusedChains.cuh) from the
// constraint clusters. The fused step is used in single and mixed precision unless
// a constrained atom is massless: the separate kernels then leave a stale posDelta
// for the constraint solve, which the fused kernel does not reproduce. Massless
// unconstrained atoms (virtual sites) are fine. CUPO_PLUGIN_FUSED_INTEGRATOR=0
// keeps the separate kernels.
void CudaIntegrateLangevinMiddleStepKernel::buildFusedLayout(const OpenMM::System& system,
        const vector<OpenMM::mm_int4>& settleA, const vector<OpenMM::mm_int4>& shakeA,
        const vector<OpenMM::mm_float4>& shakeP) {
    useFused = false;
    const char* env = getenv("CUPO_PLUGIN_FUSED_INTEGRATOR");
    if (env != nullptr && env[0] == '0')
        return;
    const int numAtoms = system.getNumParticles();
    vector<char> inCluster(numAtoms, 0);
    auto take = [&](int a) { if (a >= 0) inCluster[a] = 1; };
    for (const auto& c : settleA) { take(c.x); take(c.y); take(c.z); }
    for (const auto& c : shakeA) { take(c.x); take(c.y); take(c.z); take(c.w); }
    for (int i = 0; i < numAtoms; i++)
        if (inCluster[i] && system.getParticleMass(i) == 0.0)
            return;

    // The SHAKE clusters, largest first; each warp holds clusters of one size, and
    // no cluster straddles two warps. The order changes no value: each cluster is
    // solved on its own, and the noise is keyed on the atom.
    const int L = LANES, W = 32;
    auto size = [](const OpenMM::mm_int4& c) { return c.w != -1 ? 4 : c.z != -1 ? 3 : 2; };
    vector<int> order(shakeA.size());
    for (int i = 0; i < (int) order.size(); i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return size(shakeA[a]) > size(shakeA[b]); });
    vector<OpenMM::mm_int4> sortedA;
    vector<OpenMM::mm_float4> sortedP;
    vector<int> lane, laneCl;   // per lane: its atom, and the cluster it solves (-1: none)
    auto padWarp = [&] { while (lane.size() % W) { lane.push_back(-1); laneCl.push_back(-1); } };
    int prev = 0;
    for (int i : order) {
        const OpenMM::mm_int4 c = shakeA[i];
        const int na = size(c);
        if (na != prev || (int) (lane.size() % W) + na > W) padWarp();
        prev = na;
        const int at[4] = {c.x, c.y, c.z, c.w};
        for (int k = 0; k < na; k++) {
            lane.push_back(at[k]);
            laneCl.push_back(k == 0 ? (int) sortedA.size() : -1);
        }
        sortedA.push_back(c);
        sortedP.push_back(shakeP[i]);
    }
    // The free atoms go on lanes 30 and 31 of the SETTLE warps when they fit there,
    // else into the SHAKE warps' spare lanes and warps of their own.
    vector<int> freeAtoms;
    for (int i = 0; i < numAtoms; i++)
        if (!inCluster[i]) freeAtoms.push_back(i);
    const int nSettle = (int) settleA.size();
    fusedSettleChunks = (nSettle + SETTLE_PER_BLOCK - 1) / SETTLE_PER_BLOCK;
    const bool freeInSettle = nSettle > 0 &&
        (int) freeAtoms.size() <= fusedSettleChunks * SETTLE_FREE_PER_BLOCK;
    fusedNumFree = freeInSettle ? (int) freeAtoms.size() : 0;
    if (!freeInSettle) {
        size_t f = 0;
        for (size_t t = 0; t < lane.size() && f < freeAtoms.size(); t++)
            if (lane[t] == -1) lane[t] = freeAtoms[f++];
        for (; f < freeAtoms.size(); f++) { lane.push_back(freeAtoms[f]); laneCl.push_back(-1); }
    }
    while (lane.size() % L) { lane.push_back(-1); laneCl.push_back(-1); }
    fusedPackedBlocks = (int) (lane.size() / L);
    // The largest cluster each SHAKE unit solves. The clusters are sorted largest
    // first, so the units with large clusters are the first ones.
    vector<int> unitSize(fusedPackedBlocks, 0);
    for (size_t t = 0; t < laneCl.size(); t++)
        if (laneCl[t] != -1) unitSize[t / L] = std::max(unitSize[t / L], size(sortedA[laneCl[t]]));

    // The grid: the kernel's resident capacity (its occupancy times the SM count),
    // capped at the number of work units. Blocks take one unit each, then the rest
    // dynamically (see fusedLangevinMiddleStep), so no per-architecture tuning is
    // needed.
    //
    // For experiments, CUPO_PLUGIN_FUSED_BLOCKS_PER_SM=<x> overrides the occupancy
    // (a finite number in [0, 32]; 0 launches one block per unit), and
    // CUPO_PLUGIN_FUSED_REPORT_CONFIG=1 prints the layout as one JSON record.
    OpenMM::CudaContext& cu = static_cast<OpenMM::CudaContext&>(cc);
    int sms = 0, blocksPerSM = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, cu.getDeviceIndex());
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, cu.getDeviceIndex());
    // The occupancy of the instantiation this precision launches (both CONSTS variants
    // share the launch bounds, so either answers for both). With no packed units the
    // kernel is the one without the SHAKE path, which has a higher occupancy.
    const bool packed = fusedPackedBlocks > 0;
    if (mixedPrec && mixedDouble)
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, packed ? fusedLangevinMiddleStep<double, false, true>
                                                      : fusedLangevinMiddleStep<double, false, false>, LANES, 0);
    else if (mixedPrec)
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, packed ? fusedLangevinMiddleStep<matx::fltflt, false, true>
                                                      : fusedLangevinMiddleStep<matx::fltflt, false, false>, LANES, 0);
    else
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, packed ? fusedLangevinMiddleStep<float, false, true>
                                                      : fusedLangevinMiddleStep<float, false, false>, LANES, 0);
    const int occupancyBlocksPerSM = std::max(blocksPerSM, 1);
    double blocksPerSm = occupancyBlocksPerSM;
    if (const char* value = getenv("CUPO_PLUGIN_FUSED_BLOCKS_PER_SM")) {
        char* end = nullptr;
        errno = 0;
        blocksPerSm = std::strtod(value, &end);
        if (end == value || *end != '\0' || errno == ERANGE || !std::isfinite(blocksPerSm) ||
                blocksPerSm < 0.0 || blocksPerSm > 32.0)
            throw OpenMM::OpenMMException("CUPO_PLUGIN_FUSED_BLOCKS_PER_SM must be a finite number in [0, 32]; 0 launches one block per work unit");
    }
    const int units = fusedPackedBlocks + fusedSettleChunks;
    // Heavy SHAKE units (see heavy_start in FusedChains.cuh): the units that solve a
    // cluster of at least heavySize atoms, when there are at most a quarter as many
    // of them as SMs. On dhfr and tyk2 the units with 3- and 4-atom clusters set
    // the kernel's end, and run 15-30% faster with an SM to themselves. In mixed
    // precision only: in single precision the heavy units measured 0.1-0.6 us per
    // step slower on the RTX 6000 Ada and the RTX 5090, and the float kernel is
    // compiled without them.
    // CUPO_PLUGIN_FUSED_HEAVY_SIZE=<n> sets heavySize, for measurements and tests (0: off).
    int heavySize = 3;
    if (const char* value = getenv("CUPO_PLUGIN_FUSED_HEAVY_SIZE"))
        heavySize = std::atoi(value);
    fusedHeavy = 0;
    if (heavySize > 0)
        while (fusedHeavy < fusedPackedBlocks && unitSize[fusedHeavy] >= heavySize) fusedHeavy++;
    if (fusedHeavy > sms / 4 || fusedSettleChunks == 0 || !mixedPrec) fusedHeavy = 0;
    // With heavy units, every block's first unit is handed out at run time, and the
    // other blocks on the heavy units' SMs exit. The grid is then sized so that the
    // blocks left over, spread evenly over the other SMs, hold every other unit at
    // once: about (units - heavy) * sms / (sms - heavy), plus the heavy units' own.
    // More blocks than that would let fast blocks claim several units on one SM
    // while other SMs idle.
    const int capacity = std::max((int) (sms * blocksPerSm), 1);
    const int heavyGrid = fusedHeavy > 0 ?
        (int) (((long long) (units - fusedHeavy) * sms + (sms - fusedHeavy) - 1) / (sms - fusedHeavy)) + fusedHeavy : 0;
    // CUPO_PLUGIN_FUSED_BLOCKS_PER_SM, when set, fixes the grid as documented above,
    // heavy units or not.
    const bool gridSet = getenv("CUPO_PLUGIN_FUSED_BLOCKS_PER_SM") != nullptr;
    fusedGridBlocks = blocksPerSm == 0.0 ? units :
        fusedHeavy > 0 && !gridSet ? std::min(capacity, heavyGrid) : std::min(units, capacity);
    if (const char* report = getenv("CUPO_PLUGIN_FUSED_REPORT_CONFIG")) {
        if (report[0] == '1') {
            std::printf("FUSED_INTEGRATOR_CONFIG {\"precision\":\"%s\",\"mixed_compute\":\"%s\",\"compute_major\":%d,"
                "\"sms\":%d,\"threads\":%d,\"occupancy_blocks_per_sm\":%d,\"blocks_per_sm\":%.8g,"
                "\"settle_chunks\":%d,\"packed_blocks\":%d,\"heavy_units\":%d,\"units\":%d,\"grid_blocks\":%d}\n",
                mixedPrec ? "mixed" : "single", !mixedPrec ? "none" : mixedDouble ? "double" : "fltflt", major, sms, L, occupancyBlocksPerSM, blocksPerSm,
                fusedSettleChunks, fusedPackedBlocks, fusedHeavy, units, fusedGridBlocks);
            std::fflush(stdout);
        }
    }
    if (fusedGridBlocks == 0)
        return;

    if (!sortedA.empty()) {
        fusedShakeAtoms.initialize<OpenMM::mm_int4>(cc, sortedA.size(), "nvFusedShakeAtoms");
        fusedShakeParams.initialize<OpenMM::mm_float4>(cc, sortedP.size(), "nvFusedShakeParams");
        fusedShakeAtoms.upload(sortedA);
        fusedShakeParams.upload(sortedP);
    }
    if (!lane.empty()) {
        fusedLaneAtom.initialize<int>(cc, lane.size(), "nvFusedLaneAtom");
        fusedLaneCluster.initialize<int>(cc, laneCl.size(), "nvFusedLaneCluster");
        fusedLaneAtom.upload(lane);
        fusedLaneCluster.upload(laneCl);
    }
    if (fusedNumFree > 0) {
        fusedFree.initialize<int>(cc, fusedNumFree, "nvFusedFree");
        fusedFree.upload(freeAtoms);
    }
    // [0]: blocks finished this launch; [1]: work units handed out past the first wave;
    // [2]: heavy units claimed. All reset by the last block to finish, so they are zero
    // again for the next launch. [3]: the launch tag of heavy_start, 1 at first and
    // advanced by the last block of each launch with heavy units.
    fusedDone.initialize<unsigned int>(cc, 4, "nvFusedDone");
    fusedDone.upload(vector<unsigned int>{0u, 0u, 0u, 1u});
    if (fusedHeavy > 0) {
        // Per-SM rank and state words, tagged with the launch tag (heavy_start); zero
        // reads as empty, since the tag is never 0.
        fusedSmWords.initialize<unsigned long long>(cc, 2 * NV_FUSED_SM_SLOTS, "nvFusedSmWords");
        fusedSmWords.upload(vector<unsigned long long>(2 * NV_FUSED_SM_SLOTS, 0ull));
    }
    // The SETTLE constants table (settle_lanes::Consts) is a candidate when every
    // water has the same three masses, in the same order, and the same parameters;
    // the device confirms it against velm at the first launch (launchFused).
    settleConstsState = 0;
    if (nSettle > 0) {
        bool same = true;
        const OpenMM::mm_int4 a0 = settleA[0];
        const double m0[3] = {system.getParticleMass(a0.x), system.getParticleMass(a0.y),
                              system.getParticleMass(a0.z)};
        for (const auto& c : settleA)
            same = same && system.getParticleMass(c.x) == m0[0] && system.getParticleMass(c.y) == m0[1] &&
                   system.getParticleMass(c.z) == m0[2];
        settleConstsState = same ? 1 : -1;
        if (same) {
            const size_t bytes = !mixedPrec ? sizeof(settle_lanes::Consts<float>)
                                 : mixedDouble ? sizeof(settle_lanes::Consts<double>)
                                               : sizeof(settle_lanes::Consts<matx::fltflt>);
            settleConsts.initialize(cc, 1, bytes, "nvSettleConsts");
            settleConstsOk.initialize<int>(cc, 1, "nvSettleConstsOk");
        }
    }
    useFused = true;
}

template<class Real>
void CudaIntegrateLangevinMiddleStepKernel::launchFusedT(cudaStream_t stream, int numAtoms,
        int paddedNumAtoms, const LangevinMiddleIntegrator& integrator) {
    OpenMM::CudaContext& cu = static_cast<OpenMM::CudaContext&>(cc);
    OpenMM::IntegrationUtilities& integration = cc.getIntegrationUtilities();
    ChainArgs<Real> c{};
    c.nSettle = numSettle;
    c.paddedNumAtoms = paddedNumAtoms;
    c.tol = Real(integrator.getConstraintTolerance());
    c.force = (const long long*) dptr(cu, cc.getLongForceBuffer());
    c.posq = (float4*) dptr(cu, cc.getPosq());
    c.velm = (typename CV<Real>::s4*) dptr(cu, cc.getVelm());
    c.settleAtoms = numSettle ? (const int4*) dptr(cu, settleAtoms) : nullptr;
    c.settleParams = numSettle ? (const float2*) dptr(cu, settleParams) : nullptr;
    c.shakeAtoms = numShake ? (const int4*) dptr(cu, fusedShakeAtoms) : nullptr;
    c.shakeParams = numShake ? (const float4*) dptr(cu, fusedShakeParams) : nullptr;
    c.posqCorrection = mixedPrec ? (float4*) dptr(cu, cc.getPosqCorrection()) : nullptr;
    c.langevinCoefficients = (const typename CV<Real>::s*) dptr(cu, langevinCoefficients);
    c.dt = (const typename CV<Real>::s2*) dptr(cu, integration.getStepSize());
    c.stepCount = (unsigned*) dptr(cu, stepCount);
    c.doneBlocks = (unsigned*) dptr(cu, fusedDone);
    c.nextUnit = c.doneBlocks + 1;
    c.heavyNext = c.doneBlocks + 2;
    c.launchTag = c.doneBlocks + 3;
    c.nHeavy = fusedHeavy;
    c.smWords = fusedHeavy > 0 ? (unsigned long long*) dptr(cu, fusedSmWords) : nullptr;
    c.seed = seed;
    c.freeAtoms = fusedNumFree ? (const int*) dptr(cu, fusedFree) : nullptr;
    c.nFree = fusedNumFree;
    c.packedBlocks = fusedPackedBlocks;
    c.laneAtom = fusedPackedBlocks ? (const int*) dptr(cu, fusedLaneAtom) : nullptr;
    c.laneCluster = fusedPackedBlocks ? (const int*) dptr(cu, fusedLaneCluster) : nullptr;
    c.settleChunks = fusedSettleChunks;
    // On the first launch, build the SETTLE constants table and keep it only if every
    // water matches water 0 bitwise. That launch is eager, never graph-captured.
    if (settleConstsState == 1) {
        vector<int> one(1, 1);
        settleConstsOk.upload(one);
        settle_lanes::consts_kernel<Real><<<std::min((numSettle + 255) / 256, 1024), 256, 0, stream>>>(
            (settle_lanes::Consts<Real>*) dptr(cu, settleConsts), (int*) dptr(cu, settleConstsOk),
            c.velm, c.settleAtoms, c.settleParams, numSettle);
        cudaStreamSynchronize(stream);
        vector<int> ok;
        settleConstsOk.download(ok);
        settleConstsState = ok[0] == 1 ? 2 : -1;
    }
#if NV_FUSED_TRACE
    // CUPO_PLUGIN_FUSED_TRACE=<first>[,<count>]: trace fused launches first ..
    // first+count-1 (counted from 1; count defaults to 3), each followed by a
    // synchronisation, and print each unit's and each block's record (see
    // NV_FUSED_TRACE in FusedChains.cuh) as lines
    //   FTRACE launch=<n> kind=unit id=<u> t0=<ns> t1=<ns> shake=0|1
    //   FTRACE launch=<n> kind=block id=<b> sm=<id> t0=<ns> t1=<ns>
    // with times relative to the launch's earliest block start. Only eager launches
    // are counted: run with the CUDA graphs off.
    static long long traceLaunch = 0;
    traceLaunch++;
    long long traceFirst = 0, traceCount = 3;
    if (const char* t = getenv("CUPO_PLUGIN_FUSED_TRACE")) {
        traceFirst = std::atoll(t);
        if (const char* comma = std::strchr(t, ',')) traceCount = std::atoll(comma + 1);
    }
    const bool traced = traceFirst > 0 && traceLaunch >= traceFirst && traceLaunch < traceFirst + traceCount;
    const int traceUnits = fusedPackedBlocks + fusedSettleChunks;
    ulonglong2* traceBuf = nullptr;
    if (traced) {
        cudaStreamSynchronize(stream);
        cudaMalloc(&traceBuf, sizeof(ulonglong2) * (traceUnits + fusedGridBlocks));
        cudaMemset(traceBuf, 0, sizeof(ulonglong2) * (traceUnits + fusedGridBlocks));
    }
    c.trace = traceBuf;
#endif
    if (settleConstsState == 2) {
        c.settleConsts = (const settle_lanes::Consts<Real>*) dptr(cu, settleConsts);
        if (fusedPackedBlocks > 0)
            fusedLangevinMiddleStep<Real, true, true><<<fusedGridBlocks, LANES, 0, stream>>>(c);
        else
            fusedLangevinMiddleStep<Real, true, false><<<fusedGridBlocks, LANES, 0, stream>>>(c);
    }
    else if (fusedPackedBlocks > 0)
        fusedLangevinMiddleStep<Real, false, true><<<fusedGridBlocks, LANES, 0, stream>>>(c);
    else
        fusedLangevinMiddleStep<Real, false, false><<<fusedGridBlocks, LANES, 0, stream>>>(c);
#if NV_FUSED_TRACE
    if (traced) {
        cudaStreamSynchronize(stream);
        vector<ulonglong2> rec(traceUnits + fusedGridBlocks);
        cudaMemcpy(rec.data(), traceBuf, sizeof(ulonglong2) * rec.size(), cudaMemcpyDeviceToHost);
        cudaFree(traceBuf);
        const unsigned long long mask = 0xffffffffffffull;
        unsigned long long t0 = ~0ull;
        for (int b = 0; b < fusedGridBlocks; b++)
            if ((rec[traceUnits + b].x & mask) != 0) t0 = std::min(t0, rec[traceUnits + b].x & mask);
        std::printf("FTRACE launch=%lld units=%d packed=%d grid=%d\n", traceLaunch, traceUnits, fusedPackedBlocks, fusedGridBlocks);
        for (int u = 0; u < traceUnits; u++)
            std::printf("FTRACE launch=%lld kind=unit id=%d t0=%lld t1=%lld shake=%d\n", traceLaunch, u,
                        (long long) ((rec[u].x & mask) - t0), (long long) ((rec[u].y & mask) - t0), u < fusedPackedBlocks ? 1 : 0);
        for (int b = 0; b < fusedGridBlocks; b++) {
            const ulonglong2 r = rec[traceUnits + b];
            if ((r.x & mask) == 0) {   // exited at once (heavy_start)
                std::printf("FTRACE launch=%lld kind=exit id=%d sm=%d\n", traceLaunch, b, (int) (r.x >> 48));
                continue;
            }
            std::printf("FTRACE launch=%lld kind=block id=%d sm=%d t0=%lld t1=%lld\n", traceLaunch, b, (int) (r.x >> 48),
                        (long long) ((r.x & mask) - t0), (long long) ((r.y & mask) - t0));
        }
        std::fflush(stdout);
    }
#endif
}

void CudaIntegrateLangevinMiddleStepKernel::initialize(const OpenMM::System& system, const LangevinMiddleIntegrator& integrator) {
    cc.initializeContexts();
    OpenMM::ContextSelector selector(cc);
    if (cc.getUseDoublePrecision())
        throw OpenMM::OpenMMException("LangevinMiddleIntegrator supports only single or mixed precision, not double.");
    // Counter-based Philox noise -- no OpenMM RNG buffer needed. Resolve the seed
    // (0 => pick a nondeterministic one, matching OpenMM's convention).
    comFrequency = integrator.getComFrequency();
    seed = (unsigned int) integrator.getRandomNumberSeed();
    if (seed == 0) {
        std::random_device rd;
        seed = rd();
        if (seed == 0)
            seed = 1u;
    }
    // The inverse total mass for the centre-of-mass removal, from the masses at this
    // point; 0 when the System has no mass at all.
    double massSum = 0;
    for (int i = 0; i < system.getNumParticles(); i++)
        massSum += system.getParticleMass(i);
    inverseTotalMass = massSum == 0 ? 0.0f : (float) (1.0 / massSum);
    mixedPrec = cc.getUseMixedPrecision() ? 1 : 0;
    mixedDouble = mixedPrec && chooseMixedCompute(static_cast<OpenMM::CudaContext&>(cc).getDeviceIndex());
    // One partial momentum per block of the centre-of-mass launches; there are never
    // more blocks than padded atoms.
    comMomentumPartials.initialize<OpenMM::mm_float4>(cc, std::max(1, cc.getPaddedNumAtoms()), "nvComMomentumPartials");
    stepCount.initialize<unsigned int>(cc, 1, "nvStepCount");
    vector<unsigned int> zero(1, 0u);
    stepCount.upload(zero);
    // The two Langevin coefficients (velocity decay, noise scale) in the velocity's
    // storage precision, and stage 2's uncorrected displacement, 32 bytes an atom in
    // mixed precision. The coefficients are computed by the first execute().
    if (mixedPrec) {
        langevinCoefficients.initialize<double>(cc, 2, "nvLangevinCoefficients");
        plannedDisplacement.initialize<OpenMM::mm_double4>(cc, cc.getPaddedNumAtoms(), "nvPlannedDisplacement");
    }
    else {
        langevinCoefficients.initialize<float>(cc, 2, "nvLangevinCoefficients");
        plannedDisplacement.initialize<OpenMM::mm_float4>(cc, cc.getPaddedNumAtoms(), "nvPlannedDisplacement");
    }
    coefficientsValid = false;
    graph = nullptr;
    graphExec = nullptr;
    graphCaptured = false;
    graphWarmup = 0;

    // Build our OWN SETTLE + SHAKE constraint data (so the whole step can run on a
    // capturable stream). We implement only SETTLE (rigid water) and SHAKE
    // (H-clusters); coupled constraints need CCMA, which we do not have, so reject
    // such systems here rather than integrate them incorrectly.
    buildConstraintClusters(system);
    if (numLeftoverConstraints > 0)
        throw OpenMM::OpenMMException("LangevinMiddleIntegrator does not support " +
                to_string(numLeftoverConstraints) + " constraint(s) that require CCMA "
                "(only SETTLE and SHAKE constraints are implemented).");

    // Virtual sites cost one kernel a step, and only where a system has any. They
    // carry no mass, so Part1/2/3 already skip them on `v.w != 0` and no constraint
    // cluster can contain one; all that is missing is re-deriving their positions
    // from the parents this step moved, which OpenMM's own integrators do at the
    // same point. Their FORCES need nothing from us -- distributeForcesFromVirtualSites
    // runs in OpenMM's CalcForcesAndEnergy kernel, which we do not replace.
    hasVsites = false;
    for (int i = 0; i < system.getNumParticles() && !hasVsites; i++)
        hasVsites = system.isVirtualSite(i);

    // Graph capture is ON by default; set CUPO_PLUGIN_CUDA_GRAPH=0 to force the
    // eager per-step-launch path -- the same switch that disables the nonbonded
    // graph, because wanting one without the other has never come up. We launch the whole step (incr + Part1/2/3 + our
    // own SETTLE/SHAKE) via the AOT launchers on our OWN capturable stream, since
    // OpenMM's getCurrentStream() is the uncapturable stream 0 with no
    // setCurrentStream(). A cross-stream event handshake fences against OpenMM's
    // force/KE/reorder work on stream 0 each step.
    //
    // The fused step skips all of that and launches straight on stream 0. It is one
    // kernel, so a graph of it saves no launches, and the hop to our stream and back
    // left the GPU idle before it on the critical path (~2.9 us a step on dhfr_tip4p,
    // 0.3 us without the hop). The graph stays for the separate kernels -- several
    // launches a step -- which run when CUPO_PLUGIN_FUSED_INTEGRATOR=0 or a
    // constrained atom is massless. CUPO_PLUGIN_INTEGRATOR_GRAPH=1 puts the fused step
    // back in the graph, to compare the two schedules.
    const char* integratorGraph = getenv("CUPO_PLUGIN_INTEGRATOR_GRAPH");
    const bool fusedInGraph = integratorGraph != nullptr && integratorGraph[0] == '1';
    const bool graphsOn = nvCudaGraphEnabled();
    graphEnabled = graphsOn && (!useFused || fusedInGraph);
    const char* wholeEnv = std::getenv("CUPO_PLUGIN_WHOLE_STEP_GRAPH");
    if (wholeEnv && std::string(wholeEnv) != "0" && std::string(wholeEnv) != "1")
        throw OpenMM::OpenMMException("CUPO_PLUGIN_WHOLE_STEP_GRAPH must be 0 or 1");
    // Whole-step capture is independent of the component graph setting.
    wholeEnabled = nvCudaGraphEnabled() && (!wholeEnv || std::string(wholeEnv) == "1") && cc.getNumContexts() == 1;
    // Barostat state updates run outside capture. Other allowed forces have no
    // per-step host updates or stock neighbor lists. Exclude subclasses too.
    wholeBoxDependent = hasVsites;
    int numBarostats = 0;
    for (int i = 0; i < system.getNumForces(); ++i) {
        const auto& f = system.getForce(i);
        const bool barostat = typeid(f) == typeid(NvOpenMM::MonteCarloBarostat) ||
                              typeid(f) == typeid(OpenMM::MonteCarloBarostat);
        if (barostat) ++numBarostats;
        // Custom mode and attached cutoff expressions perform host-side parameter
        // and long-range preparation. Keep their ordinary execution path.
        if (typeid(f) == typeid(NvOpenMM::NonbondedForce)) {
            const auto& nonbonded = static_cast<const NvOpenMM::NonbondedForce&>(f);
            wholeEnabled = wholeEnabled && nonbonded.getPairMode() == NvOpenMM::NonbondedForce::Standard
                && !nonbonded.hasCutoffExpression();
        }
        // cuPo reads its box through a device pointer. Stock periodic bonded
        // kernels and virtual-site kernels can capture the box by value.
        if (typeid(f) != typeid(NvOpenMM::NonbondedForce) && f.usesPeriodicBoundaryConditions())
            wholeBoxDependent = true;
        wholeEnabled = wholeEnabled && (typeid(f) == typeid(NvOpenMM::NonbondedForce) ||
            typeid(f) == typeid(OpenMM::HarmonicBondForce) ||
            typeid(f) == typeid(OpenMM::HarmonicAngleForce) ||
            typeid(f) == typeid(OpenMM::PeriodicTorsionForce) ||
            typeid(f) == typeid(OpenMM::CustomExternalForce) || barostat);
    }
    wholeHasBarostat = numBarostats != 0;
    wholeEnabled = wholeEnabled && numBarostats <= 1;
    if (const char* batch = std::getenv("CUPO_PLUGIN_WHOLE_STEP_BATCH")) {
        char* end = nullptr;
        errno = 0;
        long value = std::strtol(batch, &end, 10);
        if (errno || end == batch || *end || value < 1 || value > 16384)
            throw OpenMM::OpenMMException("CUPO_PLUGIN_WHOLE_STEP_BATCH must be an integer from 1 to 16384");
        wholeBatchSize = (int) value;
    }
    wholeBatchReport = std::getenv("CUPO_PLUGIN_WHOLE_STEP_BATCH_REPORT") &&
        std::string(std::getenv("CUPO_PLUGIN_WHOLE_STEP_BATCH_REPORT")) == "1";
    if (graphEnabled) {
        cudaStreamCreateWithFlags(&ownStream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&forceDoneEvent, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&graphDoneEvent, cudaEventDisableTiming);
    }
    else if (!graphsOn) {
        fprintf(stderr, "cupo warning: LangevinMiddleIntegrator CUDA graph off "
                        "(disabled by CUPO_PLUGIN_CUDA_GRAPH=0); using the eager "
                        "per-step-launch path.\n");
    }

    // SETTLE/SHAKE overlap. Unconditional where it applies: the two solvers own
    // disjoint atoms, so running them concurrently is bitwise identical to running
    // them in sequence -- there is no tradeoff to expose a switch for. The only gate
    // is structural: a fork needs BOTH solvers to have clusters, and a pure-water
    // system is all SETTLE while an unsolvated one is all SHAKE, so for those the
    // two event nodes would buy nothing.
    constraintOverlap = (numSettle > 0 && numShake > 0);
    if (constraintOverlap) {
        cudaStreamCreateWithFlags(&shakeStream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&constraintFork, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&constraintJoin, cudaEventDisableTiming);
    }
}


// Launch the whole step with our OWN SETTLE/SHAKE via the AOT launchers on `stream`.
// Order matches OpenMM's applyConstraints sequence: incr -> Part1 -> {settleVel,
// shakeVel} -> Part2 -> {settlePos, shakePos} -> Part3. The braces are literal: the
// two halves of a constraint block are independent and run CONCURRENTLY, SHAKE on
// its own stream (see `constrain` below). Ordering between blocks is unchanged, so
// the result is identical to the serial order it replaces -- the solvers never see
// each other's atoms. SHAKE-vel operates on velm (bound to its posDelta slot). Runs
// both eagerly (stream 0) and under graph capture (our own stream).
template<class Real>
void CudaIntegrateLangevinMiddleStepKernel::launchStepT(cudaStream_t stream, int numAtoms,
        int paddedNumAtoms, const LangevinMiddleIntegrator& integrator) {
    OpenMM::CudaContext& cu = static_cast<OpenMM::CudaContext&>(cc);
    OpenMM::IntegrationUtilities& integration = cc.getIntegrationUtilities();
    void* velm = dptr(cu, cc.getVelm());
    void* posq = dptr(cu, cc.getPosq());
    void* posDelta = dptr(cu, integration.getPosDelta());
    void* planned = dptr(cu, plannedDisplacement);
    void* force = dptr(cu, cc.getLongForceBuffer());
    void* dt = dptr(cu, integration.getStepSize());
    void* coefficients = dptr(cu, langevinCoefficients);
    void* stepNumber = dptr(cu, stepCount);
    void* posqCorr = mixedPrec ? dptr(cu, cc.getPosqCorrection()) : nullptr;
    void* sAtoms = numSettle ? dptr(cu, settleAtoms) : nullptr;
    void* sParams = numSettle ? dptr(cu, settleParams) : nullptr;
    void* kAtoms = numShake ? dptr(cu, shakeAtoms) : nullptr;
    void* kParams = numShake ? dptr(cu, shakeParams) : nullptr;
    double tol = integrator.getConstraintTolerance();

    // ONE fork spanning both constraint blocks AND the Part2 between them.
    //
    // Safe because the two solvers own DISJOINT atoms -- buildConstraintClusters
    // absorbs water into SETTLE clusters first and gives SHAKE only what is left --
    // and every constraint kernel reads and writes nothing but its own cluster's
    // entries of velm/posDelta. Part2 respects that same partition rather than
    // breaking it: part2[i] reads only velm[i], so running it per branch over that
    // branch's own atoms is the identical arithmetic on identical values, and the
    // Philox noise is keyed on the ATOM index, so the result is bitwise unchanged.
    //
    // Pulling Part2 inside the fork is the point. As one kernel over all atoms it
    // was a barrier -- it had to wait for BOTH velocity solvers and be waited on by
    // BOTH position solvers, which is the gap on either side of it in a timeline.
    // Split, the two chains are independent end to end:
    //     SETTLE: settleVel -> part2(other) -> settlePos
    //     SHAKE:  shakeVel  -> part2(shake) -> shakePos
    // so the step pays max(chain) instead of max(vel) + part2 + max(pos).
    //
    // Unconstrained atoms have no velocity solver to wait on, so they ride with the
    // SETTLE branch; a third launch for them would cost more fixed overhead than the
    // slightly tighter dependency is worth.
    //
    // Worth doing at all because these kernels are too SMALL to fill the GPU rather
    // than long enough to saturate it: at 64 threads/block dhfr runs SHAKE as ~13
    // blocks and SETTLE as ~110 on a 170-SM part, so a second solver runs beside the
    // first for close to free.
    nvlmi_part1<Real>(stream, numAtoms, paddedNumAtoms, velm, force, dt, stepNumber);
    if (constraintOverlap) {
        void* shakeIdx = dptr(cu, part2ShakeIdx);
        void* otherIdx = dptr(cu, part2OtherIdx);
        cudaEventRecord(constraintFork, stream);
        cudaStreamWaitEvent(shakeStream, constraintFork, 0);

        nvlmi_settle_velocities<Real>(stream, numSettle, posq, velm, sAtoms, posqCorr);
        nvlmi_part2<Real>(stream, numPart2Other, otherIdx, velm, posDelta, planned, coefficients, dt, stepNumber, seed);
        nvlmi_settle_positions<Real>(stream, numSettle, posq, posDelta, velm, sAtoms, sParams, posqCorr);

        nvlmi_shake_velocities<Real>(shakeStream, numShake, tol, posq, velm, kAtoms, kParams, posqCorr);
        nvlmi_part2<Real>(shakeStream, numPart2Shake, shakeIdx, velm, posDelta, planned, coefficients, dt, stepNumber, seed);
        nvlmi_shake_positions<Real>(shakeStream, numShake, tol, posq, posDelta, kAtoms, kParams, posqCorr);

        cudaEventRecord(constraintJoin, shakeStream);
        cudaStreamWaitEvent(stream, constraintJoin, 0);
    }
    else {
        if (numSettle)
            nvlmi_settle_velocities<Real>(stream, numSettle, posq, velm, sAtoms, posqCorr);
        if (numShake)
            nvlmi_shake_velocities<Real>(stream, numShake, tol, posq, velm, kAtoms, kParams, posqCorr);
        nvlmi_part2<Real>(stream, numAtoms, nullptr, velm, posDelta, planned, coefficients, dt, stepNumber, seed);
        if (numSettle)
            nvlmi_settle_positions<Real>(stream, numSettle, posq, posDelta, velm, sAtoms, sParams, posqCorr);
        if (numShake)
            nvlmi_shake_positions<Real>(stream, numShake, tol, posq, posDelta, kAtoms, kParams, posqCorr);
    }
    nvlmi_part3<Real>(stream, numAtoms, posq, velm, posDelta, planned, dt, posqCorr);
}

// The ONE place the runtime precision flag becomes a type: single -> float, mixed
// -> matx::fltflt (df64), or double on a GPU with fast FP64 (chooseMixedCompute).
// Everything below it -- launchStepT and every launcher it calls -- is templated, so
// no per-launch branch survives.
void CudaIntegrateLangevinMiddleStepKernel::launchStep(cudaStream_t stream, int numAtoms,
        int paddedNumAtoms, const LangevinMiddleIntegrator& integrator) {
    if (useFused) {
        if (mixedPrec && mixedDouble)
            launchFusedT<double>(stream, numAtoms, paddedNumAtoms, integrator);
        else if (mixedPrec)
            launchFusedT<matx::fltflt>(stream, numAtoms, paddedNumAtoms, integrator);
        else
            launchFusedT<float>(stream, numAtoms, paddedNumAtoms, integrator);
    }
    else if (mixedPrec && mixedDouble)
        launchStepT<double>(stream, numAtoms, paddedNumAtoms, integrator);
    else if (mixedPrec)
        launchStepT<matx::fltflt>(stream, numAtoms, paddedNumAtoms, integrator);
    else
        launchStepT<float>(stream, numAtoms, paddedNumAtoms, integrator);
}

void CudaIntegrateLangevinMiddleStepKernel::execute(OpenMM::ContextImpl& context, const LangevinMiddleIntegrator& integrator) {
    OpenMM::ContextSelector selector(cc);
    OpenMM::IntegrationUtilities& integration = cc.getIntegrationUtilities();
    const int numAtoms = cc.getNumAtoms();
    const int paddedNumAtoms = cc.getPaddedNumAtoms();
    const double stepSize = integrator.getStepSize();
    const double temperature = integrator.getTemperature();
    const double friction = integrator.getFriction();
    // Tolerance is captured by value, so a component graph must be rebuilt.
    if (prevConstraintTolerance != integrator.getConstraintTolerance()) {
        if (graphExec) cudaGraphExecDestroy(graphExec);
        if (graph) cudaGraphDestroy(graph);
        graphExec = nullptr;
        graph = nullptr;
        graphCaptured = false;
        prevConstraintTolerance = integrator.getConstraintTolerance();
    }
    comFrequency = integrator.getComFrequency();
    // The step size goes into the dt buffer the kernels read.
    integration.setNextStepSize(stepSize);
    // The Ornstein-Uhlenbeck coefficients of stage 2, recomputed when the temperature,
    // friction or step size changed: over dt, v <- decay v + noise sqrt(1/m) xi with
    // decay = exp(-gamma dt) and noise = sqrt(kT (1 - decay^2)). The upload is
    // synchronous and in place: graphs and the fused kernel's arguments hold the
    // buffer's address.
    if (!coefficientsValid || temperature != coeffTemperature || friction != coeffFriction ||
        stepSize != coeffStepSize) {
        const double kT = NV_BOLTZ * temperature;
        const double decay = exp((-stepSize) * friction);
        const double decaySq = decay * decay;
        const double noise = sqrt(kT * (1.0 - decaySq));
        if (mixedPrec) {
            vector<double> coeffs = {decay, noise};
            langevinCoefficients.upload(coeffs);
        }
        else {
            vector<float> coeffs = {(float) decay, (float) noise};
            langevinCoefficients.upload(coeffs);
        }
        coeffTemperature = temperature;
        coeffFriction = friction;
        coeffStepSize = stepSize;
        coefficientsValid = true;
    }

    OpenMM::CudaContext& cu = static_cast<OpenMM::CudaContext&>(cc);
    cudaStream_t s0 = (cudaStream_t) cu.getCurrentStream();
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    if (wholeEnabled && cudaStreamIsCapturing(s0, &captureStatus) != cudaSuccess)
        throw OpenMM::OpenMMException("Cannot query integrator stream capture status");
    const bool outerCapture = captureStatus == cudaStreamCaptureStatusActive;


    // Fused, periodic center-of-mass motion removal (replaces an every-step
    // CMMotionRemover). Runs on stream 0; only changes velocities, so it does not
    // affect the forces already computed. The graph path fences it via forceDoneEvent.
    if (comFrequency > 0 && cc.getStepCount() % comFrequency == 0) {
        void* velm = dptr(cu, cc.getVelm());
        void* momentumPartials = dptr(cu, comMomentumPartials);
        if (mixedPrec) {
            nvlmi_calc_com<matx::fltflt>(s0, numAtoms, velm, momentumPartials);
            nvlmi_remove_com<matx::fltflt>(s0, numAtoms, velm, momentumPartials, inverseTotalMass);
        }
        else {
            nvlmi_calc_com<float>(s0, numAtoms, velm, momentumPartials);
            nvlmi_remove_com<float>(s0, numAtoms, velm, momentumPartials, inverseTotalMass);
        }
    }

    // Per-step device work, two paths:
    //  - graph off (the default for the fused step; see initialize): our full step
    //    (incl. our SETTLE/SHAKE) eagerly on stream 0.
    //  - graph on: capture the full step once on our own capturable stream and
    //    replay it, fenced against s0 by events. The Philox noise is stateless
    //    (device step counter), so each replay advances it.
    if (outerCapture || !graphEnabled) {
        launchStep(s0, numAtoms, paddedNumAtoms, integrator);
    }
    else {
        // Our stream waits until this step's forces (+ periodic CoM) finish on s0.
        cudaEventRecord(forceDoneEvent, s0);
        cudaStreamWaitEvent(ownStream, forceDoneEvent, 0);
        if (graphCaptured) {
            cudaGraphLaunch(graphExec, ownStream);
        }
        else if (graphWarmup < NV_GRAPH_WARMUP) {
            graphWarmup++;
            launchStep(ownStream, numAtoms, paddedNumAtoms, integrator);
        }
        else if (cudaStreamBeginCapture(ownStream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            // Should not happen on our own stream; degrade to eager if it ever does.
            cudaGetLastError();
            graphEnabled = false;
            launchStep(ownStream, numAtoms, paddedNumAtoms, integrator);
        }
        else {
            launchStep(ownStream, numAtoms, paddedNumAtoms, integrator);
            if (cudaStreamEndCapture(ownStream, &graph) != cudaSuccess ||
                cudaGraphInstantiate(&graphExec, graph, 0) != cudaSuccess) {
                // Capture failed; the ops were recorded (not executed), so run them.
                // If endCapture produced a graph but instantiate failed, free it.
                cudaGetLastError();
                graphEnabled = false;
                if (graph != nullptr)
                    cudaGraphDestroy(graph);
                graph = nullptr;
                graphExec = nullptr;
                launchStep(ownStream, numAtoms, paddedNumAtoms, integrator);
            }
            else {
                graphCaptured = true;
                cudaGraphLaunch(graphExec, ownStream);
            }
        }
        // s0 (this step's reorder download, KE readback, next step's forces) waits
        // until our constrained posq/velm writes on ownStream are visible.
        cudaEventRecord(graphDoneEvent, ownStream);
        cudaStreamWaitEvent(s0, graphDoneEvent, 0);
    }

    // Positions are final for this step, so re-derive the virtual sites from them.
    // This is OpenMM's own kernel, launched on s0, which the join above has already
    // ordered after our stream -- so it sees the constrained positions.
    //
    // Left OUT of the graph, though setCurrentQueue could put it in: computeVirtualSites
    // passes the box as kernel ARGUMENTS, set host-side on every call, and a captured
    // node freezes whatever they were at capture. That is right under NVT and silently
    // wrong the first time a barostat moves the box. One eager launch a step is a
    // cheaper price than a second thing needing the NPT recapture envelope.
    if (hasVsites)
        integration.computeVirtualSites();

    // Capture records device work; the coordinator commits host state on replay.
    if (outerCapture)
        return;
    cc.setTime(cc.getTime()+stepSize);
    cc.setStepCount(cc.getStepCount()+1);
    cc.reorderAtoms();
    if (wholeWarmup < 55) ++wholeWarmup;
}

// Guarded NVT/NPT coordinator. Graphs live for ONE step(n) call: all
// context/force/integrator changes between calls go through an ordinary first
// step, and the next capture sees their current buffers and scalar arguments.
// OpenMM itself is unchanged. Its force pipeline builds the graph, including
// autoclears, bonded forces, pre/post computations and virtual-site forces.
bool CudaIntegrateLangevinMiddleStepKernel::executeSteps(OpenMM::ContextImpl& context,
        const LangevinMiddleIntegrator& integrator, int steps) {
    if (!wholeEnabled || steps < 2)
        return false;
    OpenMM::ContextSelector selector(cc);
    auto& cu = static_cast<OpenMM::CudaContext&>(cc);
    const int groups = integrator.getIntegrationForceGroups();
    const MonteCarloBarostatImpl* fusingBarostat = nullptr;
    if (wholeHasBarostat)
        for (auto* force : context.getForceImpls())
            if (auto* barostat = dynamic_cast<const MonteCarloBarostatImpl*>(force))
                fusingBarostat = barostat;
    bool refreshAfterMove = false;
    auto ordinaryAfterUpdate = [&]() {
        // Preserve the plugin barostat's force reuse on BOTH accept and reject.
        const bool reuseForces = fusingBarostat && fusingBarostat->leftForcesValid();
        if (!reuseForces)
            context.calcForcesAndEnergy(true, false, groups);
        execute(context, integrator);
        // On rejection cuPo's box/influence table can still describe the trial
        // box. One ordinary force evaluation restores them before graph replay.
        refreshAfterMove = reuseForces;
    };
    int completed = 0;
    // The first step also uploads changed parameters/box and prepares changed
    // bonded data. No such host-side operations may be skipped by replay.
    do {
        context.updateContextState();
        ordinaryAfterUpdate();
        ++completed;
    } while (completed < steps && wholeWarmup < 55);
    if (completed == steps)
        return true;

    struct Graphs {
        cudaGraph_t graph[2] = {};
        cudaGraphExec_t exec[2] = {};
        // key: (length, initial CoM phase); -1 means no CoM removal in the batch.
        std::map<std::pair<int, int>, std::pair<cudaGraph_t, cudaGraphExec_t>> batches;
        void clear() {
            for (auto& entry : batches) {
                if (entry.second.second) cudaGraphExecDestroy(entry.second.second);
                if (entry.second.first) cudaGraphDestroy(entry.second.first);
            }
            batches.clear();
            for (int k = 0; k < 2; ++k) {
                if (exec[k]) cudaGraphExecDestroy(exec[k]);
                if (graph[k]) cudaGraphDestroy(graph[k]);
                exec[k] = nullptr; graph[k] = nullptr;
            }
        }
        ~Graphs() { clear(); }
    } graphs;
    // No host state-update callback may be delayed across a batched replay.
    // Retain the existing per-step path for both barostats, including frequency 0.
    // Also exclude any context in which OpenMM may reorder between device steps.
    const bool batchEligible = !wholeHasBarostat && !cc.getNonbondedUtilities().getUseCutoff();
    long long launches = 0, batchedSteps = 0;
    int batchGraphs = 0, maxBatch = 1;
    int clearMigrationOverlaps = 0, clearGridResizes = 0;
    double batchBuildMs = 0;
    auto check = [](cudaError_t err) {
        if (err != cudaSuccess)
            throw OpenMM::OpenMMException(std::string("Whole-step graph: ") + cudaGetErrorString(err));
    };
    const auto savedQueue = cu.getCurrentQueue();
    const cudaStream_t savedStream = (cudaStream_t) cu.getCurrentStream();
    auto queue = cu.createQueue();
    cu.setCurrentQueue(queue);
    const cudaStream_t stream = (cudaStream_t) cu.getCurrentStream();
    cu.setCurrentQueue(savedQueue);
    // Only CAPTURE uses the explicit queue. Graphs replay on OpenMM's original
    // stream, just like cuPo's component graphs. Its energy downloads (including
    // barostat decisions) must see all preceding work on that normal stream.
    bool capturing = false;
    OpenMM::Vec3 graphBox[3];
    if (wholeHasBarostat && wholeBoxDependent)
        context.getPeriodicBoxVectors(graphBox[0], graphBox[1], graphBox[2]);
    try {
        for (; completed < steps; ++completed) {
            const int forceCount = cc.getComputeForceCount();
            context.updateContextState();
            // Both supported barostats evaluate energies when they attempt a
            // move. Observe the real hook, rather than guessing its private
            // counter from the integrator step (frequency can change at runtime).
            const bool attemptedMove = wholeHasBarostat && cc.getComputeForceCount() != forceCount;
            if (attemptedMove || refreshAfterMove) {
                ordinaryAfterUpdate();
                if (wholeBoxDependent) {
                    OpenMM::Vec3 box[3];
                    context.getPeriodicBoxVectors(box[0], box[1], box[2]);
                    for (int i = 0; i < 3; ++i) {
                        if (box[i] != graphBox[i]) graphs.clear();
                        graphBox[i] = box[i];
                    }
                }
                if (cc.getAtomsWereReordered()) graphs.clear();
                continue;
            }
            const int k = comFrequency > 0 && cc.getStepCount() % comFrequency == 0;
            if (!graphs.exec[k]) {
                cu.setCurrentQueue(queue);
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                capturing = true;
                context.calcForcesAndEnergy(true, false, groups);
                execute(context, integrator);
                const cudaError_t end = cudaStreamEndCapture(stream, &graphs.graph[k]);
                capturing = false;
                cu.setCurrentQueue(savedQueue);
                check(end);
                check(cudaGraphInstantiate(&graphs.exec[k], graphs.graph[k], 0));
                ++wholeCaptures;
            } else {
                // The supported force set has no dynamic neighbor-list retries
                // on these ordinary steps. These are the force-pipeline host
                // effects; device autoclears and all force work are in the graph.
                context.getLastForceGroups() = groups;
                cc.setForcesValid(true);
                cc.setComputeForceCount(cc.getComputeForceCount()+1);
                for (auto& p : context.getParameters())
                    cc.getEnergyParamDerivWorkspace()[p.first] = 0;
            }
            int count = batchEligible ? std::min(wholeBatchSize, steps-completed) : 1;
            const int phase = comFrequency > 0 ? cc.getStepCount() % comFrequency : -1;
            // Build each single-step variant through its normal force pipeline
            // at the actual host step. Until both exist, stop before needing the
            // missing variant; never temporarily falsify the Context step count.
            if (count > 1 && comFrequency > 0) {
                if (!graphs.graph[1] && phase != 0)
                    count = std::min(count, comFrequency-phase);
                if (!graphs.graph[0] && phase == 0 && comFrequency != 1)
                    count = 1;
            }
            cudaGraphExec_t executable = graphs.exec[k];
            if (count > 1) {
                const bool containsCom = comFrequency > 0 && (phase == 0 || count > comFrequency-phase);
                const auto key = std::make_pair(count, containsCom ? phase : -1);
                auto found = graphs.batches.find(key);
                if (found == graphs.batches.end()) {
                    const auto started = std::chrono::steady_clock::now();
                    auto& entry = graphs.batches[key];
                    check(cudaGraphCreate(&entry.first, 0));
                    cudaGraphNode_t previous = nullptr;
                    for (int j = 0; j < count; ++j) {
                        const int variant = comFrequency > 0 && (phase+j) % comFrequency == 0;
                        cudaGraphNode_t node;
                        check(cudaGraphAddChildGraphNode(&node, entry.first,
                            previous ? &previous : nullptr, previous ? 1 : 0, graphs.graph[variant]));
                        previous = node;
                    }
                    check(nvOptimizeBatchedForceClear(entry.first, clearMigrationOverlaps, clearGridResizes));
                    check(cudaGraphInstantiate(&entry.second, entry.first, 0));
                    batchBuildMs += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now()-started).count();
                    ++batchGraphs;
                    found = graphs.batches.find(key);
                }
                executable = found->second.second;
                batchedSteps += count;
                maxBatch = std::max(maxBatch, count);
            }
            check(cudaGraphLaunch(executable, savedStream));
            ++launches;
            // Preserve the diagnostic's original meaning: steps advanced through
            // whole-step replay. The report separately counts actual launches.
            wholeReplays += count;
            for (int j = 0; j < count; ++j) {
                if (j != 0)
                    cc.setComputeForceCount(cc.getComputeForceCount()+1);
                // Keep iterative time addition bit-for-bit equal to single steps.
                cc.setTime(cc.getTime()+integrator.getStepSize());
                cc.setStepCount(cc.getStepCount()+1);
                cc.reorderAtoms();
                if (cc.getAtomsWereReordered()) graphs.clear();
            }
            completed += count-1;
        }
    } catch (...) {
        if (capturing) {
            cudaGraph_t discarded = nullptr;
            cudaStreamEndCapture(stream, &discarded);
            if (discarded) cudaGraphDestroy(discarded);
        }
        cu.setCurrentQueue(savedQueue);
        throw;
    }
    if (wholeBatchReport)
        fprintf(stderr, "WHOLE_STEP_BATCH {\"requested\":%d,\"eligible\":%s,\"steps\":%d,\"launches\":%lld,\"batched_steps\":%lld,\"max_batch\":%d,\"batch_graphs\":%d,\"build_ms\":%.6f,\"clear_migration_overlaps\":%d,\"clear_grid_resizes\":%d}\n",
            wholeBatchSize, batchEligible ? "true" : "false", steps, launches,
            batchedSteps, maxBatch, batchGraphs, batchBuildMs, clearMigrationOverlaps, clearGridResizes);
    return true;
}

double CudaIntegrateLangevinMiddleStepKernel::computeKineticEnergy(OpenMM::ContextImpl& context, const LangevinMiddleIntegrator& integrator) {
    // Middle-scheme velocities are synchronized with positions -> no time shift.
    return cc.getIntegrationUtilities().computeKineticEnergy(0.0);
}

void CudaIntegrateLangevinMiddleStepKernel::createCheckpoint(std::ostream& stream) const {
    OpenMM::ContextSelector selector(cc);
    // Both component and whole-step graphs finish on (or join back to) the
    // current OpenMM stream. A blocking download observes the last complete step.
    vector<unsigned int> counter;
    stepCount.download(counter);
    const uint32_t payload[] = {0x4e564c4dU, 1U, seed, counter[0]}; // NVLM, schema 1
    stream.write(reinterpret_cast<const char*>(payload), sizeof(payload));
    if (!stream)
        throw OpenMM::OpenMMException("NvLangevinMiddleIntegrator: failed to write checkpoint state");
}

void CudaIntegrateLangevinMiddleStepKernel::loadCheckpoint(std::istream& stream) {
    uint32_t payload[4] = {};
    stream.read(reinterpret_cast<char*>(payload), sizeof(payload));
    if (!stream || payload[0] != 0x4e564c4dU || payload[1] != 1U || payload[2] == 0)
        throw OpenMM::OpenMMException("NvLangevinMiddleIntegrator: missing or unsupported checkpoint state");
    OpenMM::ContextSelector selector(cc);
    auto& cu = static_cast<OpenMM::CudaContext&>(cc);
    const auto error = cudaStreamSynchronize((cudaStream_t) cu.getCurrentStream());
    if (error != cudaSuccess)
        throw OpenMM::OpenMMException("NvLangevinMiddleIntegrator: checkpoint synchronization failed: "
                                     + string(cudaGetErrorString(error)));
    // Component graphs capture the seed by value. Loading a different Context's
    // checkpoint must discard that seed even if the requested seed was zero.
    // Whole-step/batch graphs are local to executeSteps(), so none survive here.
    if (graphExec != nullptr)
        cudaGraphExecDestroy(graphExec);
    if (graph != nullptr)
        cudaGraphDestroy(graph);
    graphExec = nullptr;
    graph = nullptr;
    graphCaptured = false;
    graphWarmup = 0;
    wholeWarmup = 0;
    seed = payload[2];
    stepCount.upload(vector<unsigned int>(1, payload[3]));
    prevConstraintTolerance = -1;
    // The integrator's settings may differ from the ones the coefficients were last
    // computed from: recompute them on the next execute().
    coefficientsValid = false;
}

} // namespace NvOpenMM
