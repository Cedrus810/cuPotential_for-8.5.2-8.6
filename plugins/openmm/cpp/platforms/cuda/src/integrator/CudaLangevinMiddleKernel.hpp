// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NV_CUDA_LANGEVIN_MIDDLE_KERNEL_H_
#define NV_CUDA_LANGEVIN_MIDDLE_KERNEL_H_

#include "nvopenmm/Kernels.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeVectorTypes.h"
#include <vector>
#include <cuda_runtime.h>

namespace NvOpenMM {



/**
 * CUDA implementation of the Nvidia LangevinMiddle integrator step.
 *
 * The device kernels (BAOAB Part1/2/3, fused center-of-mass removal, and our own
 * SETTLE/SHAKE constraint solvers) are compiled ahead of time by nvcc, as one
 * template per kernel instantiated on the compute type -- no dependency on
 * OpenMM's runtime kernel compiler. Everything lives in CudaLangevinMiddleKernel.cu
 * (host side included, so it can write <<<>>>), which keeps the launchers file-local.
 * The fused one-kernel step runs eagerly on OpenMM's stream 0; the separate
 * kernels run on our own capturable stream so they can be captured into a CUDA
 * graph.
 *
 * Only SETTLE (rigid water) and SHAKE (H-clusters) constraints are implemented;
 * systems needing coupled/CCMA constraints are rejected at
 * initialize() with an OpenMMException rather than silently mis-integrated.
 */
class CudaIntegrateLangevinMiddleStepKernel : public IntegrateLangevinMiddleStepKernel {
public:
    CudaIntegrateLangevinMiddleStepKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc)
        : IntegrateLangevinMiddleStepKernel(name, platform), cc(cc), comFrequency(0), seed(0),
          mixedPrec(0), inverseTotalMass(0.0f),
          graphEnabled(false), graphCaptured(false), graphWarmup(0), graph(nullptr), graphExec(nullptr),
          numSettle(0), numShake(0), numLeftoverConstraints(0), hasVsites(false),
          ownStream(nullptr), forceDoneEvent(nullptr), graphDoneEvent(nullptr),
          constraintOverlap(false), numPart2Shake(0), numPart2Other(0), shakeStream(nullptr),
          constraintFork(nullptr), constraintJoin(nullptr) {}
    ~CudaIntegrateLangevinMiddleStepKernel();
    void initialize(const OpenMM::System& system, const LangevinMiddleIntegrator& integrator);
    void execute(OpenMM::ContextImpl& context, const LangevinMiddleIntegrator& integrator);
    bool executeSteps(OpenMM::ContextImpl& context, const LangevinMiddleIntegrator& integrator, int steps) override;
    long long getWholeStepGraphCaptureCount() const override { return wholeCaptures; }
    long long getWholeStepGraphReplayCount() const override { return wholeReplays; }
    double computeKineticEnergy(OpenMM::ContextImpl& context, const LangevinMiddleIntegrator& integrator);
    void createCheckpoint(std::ostream& stream) const override;
    void loadCheckpoint(std::istream& stream) override;
private:
    // Launch the full step (incr + Part1 + settle/shake vel + Part2 + settle/shake
    // pos + Part3) with our OWN SETTLE/SHAKE, via the AOT launchers on `stream`.
    // Runs both eagerly (stream 0) and under graph capture (our own stream).
    // launchStep binds `mixedPrec` to a compute type and calls launchStepT with it;
    // that is the only precision branch on the whole path. (Real is float or
    // matx::fltflt, or double for mixed precision on a GPU with fast FP64 -- all
    // defined in the .cu, which is the only TU that instantiates this, so the header
    // need not know them.)
    void launchStep(cudaStream_t stream, int numAtoms, int paddedNumAtoms,
                    const LangevinMiddleIntegrator& integrator);
    template<class Real>
    void launchStepT(cudaStream_t stream, int numAtoms, int paddedNumAtoms,
                     const LangevinMiddleIntegrator& integrator);
    // Reproduce OpenMM's constraint clustering and build our own SETTLE/SHAKE
    // device arrays. Sets numSettle/numShake/numLeftoverConstraints.
    void buildConstraintClusters(const OpenMM::System& system);
    // The fused one-launch step (FusedChains.cuh): its lane layout, built from the
    // clusters, and its launch. useFused says whether this system takes it.
    void buildFusedLayout(const OpenMM::System& system, const std::vector<OpenMM::mm_int4>& settleA,
                          const std::vector<OpenMM::mm_int4>& shakeA,
                          const std::vector<OpenMM::mm_float4>& shakeP);
    template<class Real> void launchFusedT(cudaStream_t stream, int numAtoms, int paddedNumAtoms,
                     const LangevinMiddleIntegrator& integrator);

    OpenMM::ComputeContext& cc;
    int comFrequency;
    unsigned int seed;
    int mixedPrec;          // 1 = mixed precision, 0 = single; launchStep binds it to a compute type
    // Mixed precision only: compute in native double instead of float-float
    // (matx::fltflt). Chosen once per device in initialize (chooseMixedCompute).
    bool mixedDouble = false;
    float inverseTotalMass;     // 1 / (sum of particle masses), for the fused CoM removal
    // The temperature, friction and step size the Langevin coefficients in `langevinCoefficients`
    // were computed from; coefficientsValid false makes the next execute() recompute
    // them whatever the settings are.
    bool coefficientsValid = false;
    double coeffTemperature = 0, coeffFriction = 0, coeffStepSize = 0;
    double prevConstraintTolerance = -1;
    OpenMM::ComputeArray langevinCoefficients, plannedDisplacement, comMomentumPartials, stepCount;
    // CUDA-graph capture of the per-step device work (env CUPO_PLUGIN_CUDA_GRAPH=0 disables).
    // Off for the fused step, which runs eagerly on stream 0 unless
    // CUPO_PLUGIN_INTEGRATOR_GRAPH=1.
    bool graphEnabled, graphCaptured;
    int graphWarmup;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    // Our own SETTLE (rigid water) + SHAKE (H-clusters) constraint data. Built once
    // at initialize() from the System's constraints; reorder-safe (same invariant
    // OpenMM's own fixed index arrays rely on).
    OpenMM::ComputeArray settleAtoms, settleParams;   // OpenMM::mm_int4 / OpenMM::mm_float2, one per SETTLE cluster
    OpenMM::ComputeArray shakeAtoms, shakeParams;     // OpenMM::mm_int4 / OpenMM::mm_float4, one per SHAKE cluster
    int numSettle, numShake, numLeftoverConstraints;
    // Whether the system has any virtual site, so a system without one pays
    // nothing for the per-step kernel that re-derives them.
    bool hasVsites;
    // Our own capturable stream + cross-stream ordering events (OpenMM's compute,
    // KE readback and reorder run on stream 0, which cannot be captured).
    cudaStream_t ownStream;
    cudaEvent_t forceDoneEvent, graphDoneEvent;
    // SETTLE/SHAKE overlap. The two solvers own disjoint atom sets, and so does the
    // Part2 between them, so ONE fork spans the whole velocity->Part2->position
    // sequence and needs a single fork/join event pair. Set only when both solvers
    // actually have clusters.
    bool constraintOverlap;
    // Atom partition for the forked part2: atoms owned by a SHAKE cluster, and
    // everything else (SETTLE atoms plus unconstrained ones, which only need part1
    // and so ride along with the SETTLE branch rather than paying for a third
    // launch). Built once at initialize(); empty unless constraintOverlap.
    OpenMM::ComputeArray part2ShakeIdx, part2OtherIdx;
    int numPart2Shake, numPart2Other;
    cudaStream_t shakeStream;
    cudaEvent_t constraintFork, constraintJoin;
    // The fused step: SHAKE clusters sorted by size, the per-lane atom and cluster
    // tables, the free atoms on the SETTLE warps' spare lanes, and a counter of
    // finished blocks the kernel uses to bump the step counter once.
    bool useFused = false;
    bool wholeEnabled = false;
    bool wholeHasBarostat = false, wholeBoxDependent = false;
    int wholeWarmup = 0;
    int wholeBatchSize = 1;
    bool wholeBatchReport = false;
    long long wholeCaptures = 0, wholeReplays = 0;
    OpenMM::ComputeArray fusedShakeAtoms, fusedShakeParams, fusedLaneAtom, fusedLaneCluster,
                         fusedFree, fusedDone, fusedSmWords;
    int fusedPackedBlocks = 0, fusedSettleChunks = 0, fusedGridBlocks = 0, fusedNumFree = 0;
    int fusedHeavy = 0;    // SHAKE units that get an SM of their own (heavy_start in FusedChains.cuh)
    // The SETTLE constants table: 0 no waters, 1 a candidate not yet checked against
    // velm, 2 in use, -1 not usable (waters differ).
    OpenMM::ComputeArray settleConsts, settleConstsOk;
    int settleConstsState = 0;
};

} // namespace NvOpenMM

#endif // NV_CUDA_LANGEVIN_MIDDLE_KERNEL_H_
