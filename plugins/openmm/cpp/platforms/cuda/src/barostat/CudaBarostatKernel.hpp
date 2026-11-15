// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NV_CUDA_BAROSTAT_KERNEL_H_
#define NV_CUDA_BAROSTAT_KERNEL_H_

#include "nvopenmm/Kernels.h"
#include "openmm/common/ComputeArray.h"
#include "openmm/common/ComputeContext.h"
#include <vector>

namespace NvOpenMM {



/**
 * CUDA implementation of the Nvidia Monte Carlo barostat's device work.
 *
 * The Monte Carlo decision lives on the host in MonteCarloBarostatImpl; this
 * class owns only the device state a trial move touches -- snapshot it, scale
 * the coordinates, put it back on rejection.
 *
 * The scaling kernels are compiled ahead of time by nvcc into this plugin's
 * CUDA library (CudaBarostatKernel.cu) and invoked through an extern-C launcher,
 * so no OpenMM runtime kernel compiler is involved. Molecules are partitioned by
 * size once, at first use, so the solvent and a protein chain each get a launch
 * shape that suits them; see the comment in CudaBarostatKernel.cu for why
 * that partition is the entire reason this kernel exists.
 *
 * Single and mixed precision only. Double-precision Contexts are rejected rather
 * than silently mis-scaled, matching LangevinMiddleIntegrator.
 */
class CudaApplyMonteCarloBarostatKernel : public ApplyMonteCarloBarostatKernel {
public:
    CudaApplyMonteCarloBarostatKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc)
        : ApplyMonteCarloBarostatKernel(name, platform), cc(cc),
          hasInitializedMolecules(false), numMolecules(0), numAtomSlots(0),
          numSmall(0), numLarge(0) {}

    void initialize(const OpenMM::System& system, const MonteCarloBarostat& barostat);
    void saveState(OpenMM::ContextImpl& context);
    void scaleCoordinates(OpenMM::ContextImpl& context, double scale);
    void restoreState(OpenMM::ContextImpl& context);

private:
    // Build the molecule tables and size the snapshot buffers. Deferred to first
    // use because the molecule decomposition comes from the ContextImpl, which
    // initialize() does not receive.
    void ensureInitialized(OpenMM::ContextImpl& context);
    // Positions, velocities and forces into the snapshot (true) or back (false).
    void copySnapshot(bool toSnapshot);

    OpenMM::ComputeContext& cc;
    bool hasInitializedMolecules;

    // Molecule decomposition, flattened. moleculeAtoms holds atom indices grouped
    // by molecule; moleculeStart[m]..moleculeStart[m+1] is molecule m's range;
    // atomMolecule[slot] is the molecule owning that slot, so the displacement
    // pass can run flat over atoms instead of over molecules.
    OpenMM::ComputeArray moleculeAtoms, moleculeStart, atomMolecule;
    // Molecule indices split by size, so each gets a launch shape that fits.
    OpenMM::ComputeArray smallMolecules, largeMolecules;
    // Per-molecule centroid displacement, written by the centroid kernels and
    // consumed by the displacement kernel.
    OpenMM::ComputeArray centroidDelta;
    int numMolecules, numAtomSlots, numSmall, numLarge;

    // Snapshot of the context taken by saveState() just before a trial box
    // change, and written back by restoreState() if the move is rejected.
    //
    // The trial force evaluation can do more than see the scaled positions: it
    // may sort the atoms into a new order (which permutes positions, velocities
    // and forces together) and setting the box may rewrap atoms (which rewrites
    // positions and the per-atom wrap counts). So the snapshot covers every
    // per-atom array those two things can change, not only the positions that
    // the scaling moves.
    //
    // Device copies: positions with charges (always float4), velocities with
    // inverse masses (double4 in mixed precision, float4 in single), and the
    // fixed-point force buffer. The forces are kept because after a rejection
    // the integrator goes on using the forces of the pre-move configuration.
    OpenMM::ComputeArray preMovePosq, preMoveVelm, preMoveForces;
    // Host copies: how many box vectors each stored position has been shifted
    // by, and which original particle sits in each slot.
    std::vector<OpenMM::mm_int4> preMoveCellOffsets;
    std::vector<int> preMoveAtomOrder;
    // Whether the force evaluation just before saveState() sorted the atoms
    // into a new order. The snapshot already holds that new order, so setting
    // it again in restoreState() leaves the order as it is; it is still done,
    // because setting the order also re-runs every force's handler that
    // re-permutes its per-atom parameters. Set in scaleCoordinates() before
    // any scaling, cleared in saveState().
    bool reorderedBeforeMove = false;
};

} // namespace NvOpenMM

#endif // NV_CUDA_BAROSTAT_KERNEL_H_
