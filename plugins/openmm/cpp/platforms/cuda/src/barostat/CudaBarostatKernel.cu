// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "CudaBarostatKernel.hpp"
#include "common/CudaFaultHelp.hpp"
#include "CudaContext.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>


// Definitions live in the plugin's namespace, so our NonbondedForce et al.
// win unqualified lookup over OpenMM's same-named classes rather than
// being ambiguous with them.
namespace NvOpenMM {
using std::vector;

// Molecules with at most this many atoms are handled by one warp each; anything
// bigger gets its own block. 32 is the warp width, so a small molecule's centroid
// is a single shuffle reduction with no loop.
static const int NV_BAROSTAT_WARP_MOLECULE_MAX = 32;

// Device address of a ComputeArray, TYPED. The launches below are ordinary <<<>>>
// calls in the same translation unit, so the compiler checks every pointer against
// the kernel's parameter -- which an extern-C bridge cannot do. This file used to
// reach its kernels through one, and cuda_scale_positions took seven void* in a
// row, four of them mutually transposable int arrays.
template <typename T>
static inline T* dptr(OpenMM::CudaContext& cu, OpenMM::ArrayInterface& a) {
    return reinterpret_cast<T*>(static_cast<uintptr_t>(cu.unwrap(a).getDevicePointer()));
}

// -----------------------------------------------------------------------------
// AOT (nvcc-compiled) kernels for the Nvidia Monte Carlo barostat. Compiled into
// libOpenMMNvidiaCUDA.so at build time (CMake globs src/*.cu). No OpenMM runtime
// compiler is involved.
//
// The job: displace every molecule's centroid by center*(scale-1), leaving
// intramolecular geometry rigid, so the box can be scaled without stretching
// bonds.
//
// Why this is not one kernel with one thread per molecule -- which is what
// OpenMM's stock scalePositions does -- is the whole point. That shape makes a
// molecule's cost proportional to its atom count on a SINGLE thread, twice over
// (once to sum the centroid, once to displace). Solvated biomolecular systems
// are exactly the worst case for it: apoA1 has 21,620 molecules whose MEDIAN
// size is 3 (water), and two of 3,205 (the protein chains). Two threads each
// walk 3,205 atoms twice while 21,618 threads sit idle, and the kernel measured
// 725 us per call on a 5090 -- about 4x the entire direct-space force kernel.
//
// So instead:
//   1. Centroids, split by molecule size so neither regime starves:
//        - small (<= WARP_MOLECULE_MAX atoms): one warp per molecule, shuffle
//          reduction. Covers the solvent, which is nearly all of them.
//        - large: one 256-thread block per molecule, shared-memory tree. The
//          3,205-atom chain becomes 13 cooperative passes instead of 3,205
//          serial ones.
//   2. Displacement: flat over atoms, so it is perfectly balanced regardless of
//      how the molecule sizes are distributed.
//
// Centroids accumulate in FLOAT, and the accuracy comes from the reduction being
// a TREE, not from the precision -- those are separate things, and this comment
// used to conflate them by justifying double against a long serial float sum.
//
// The tree is what makes it both accurate and deterministic run to run (fixed
// order), which matters because the rest of this plugin is deterministic. Double
// on top of that buys nothing measurable: the centroid only ever reaches the
// output as delta = centroid * (scale - 1), where scale - 1 is ~1e-3, and delta
// is stored as float4 anyway. Measured against a float64 reference on apoA1, max
// deviation is 1.4e-07 nm in float against 1.8e-07 nm in double -- both sitting
// on the fp32 storage floor, the difference being which way the final store
// rounds rather than any real accuracy.
//
// It is not free, though. sm_120 runs FP64 at 1/64 rate, a double needs two
// __shfl_down_sync instead of one, and the large-molecule block wants 6 KB of
// shared instead of 3 KB. Measured per move on apoA1: centroids_small
// 38.55 -> 5.86 us, centroids_large 10.25 -> 6.53 us, 50.55 -> 14.14 us
// total (3.57x).
//
// Positions live in posq (float4) in both single and mixed precision; the mixed
// mode's low-order posqCorrection is deliberately left alone, matching stock.
// The correction is ~1e-8 relative, far below the scale of a barostat move, and
// touching it would make the two implementations incomparable for no gain.
// -----------------------------------------------------------------------------

#define NV_BAROSTAT_LARGE_BLOCK 256

// One warp per molecule. Handles molecules up to the warp width in a single
// pass and loops beyond that, but the caller only routes small ones here.
__global__ void cuda_centroids_small(int numSmall, const int* __restrict__ smallMolecules,
                                       const int* __restrict__ moleculeStart,
                                       const int* __restrict__ moleculeAtoms,
                                       const float4* __restrict__ posq,
                                       float scaleMinusOne,
                                       float4* __restrict__ delta)
{
    const int warpsPerBlock = blockDim.x / warpSize;
    const int warpInBlock = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;

    for (int idx = blockIdx.x * warpsPerBlock + warpInBlock; idx < numSmall;
         idx += gridDim.x * warpsPerBlock) {
        const int mol = smallMolecules[idx];
        const int first = moleculeStart[mol];
        const int last = moleculeStart[mol + 1];
        const int numAtoms = last - first;

        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        for (int slot = first + lane; slot < last; slot += warpSize) {
            const float4 p = posq[moleculeAtoms[slot]];
            sx += p.x; sy += p.y; sz += p.z;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sx += __shfl_down_sync(0xffffffffu, sx, off);
            sy += __shfl_down_sync(0xffffffffu, sy, off);
            sz += __shfl_down_sync(0xffffffffu, sz, off);
        }
        if (lane == 0) {
            const float inv = 1.0f / (float)numAtoms;
            delta[mol] = make_float4(sx * inv * scaleMinusOne,
                                     sy * inv * scaleMinusOne,
                                     sz * inv * scaleMinusOne, 0.0f);
        }
    }
}

// One block per molecule, for the few molecules too big for a warp to chew
// through without becoming the critical path on its own.
__global__ void cuda_centroids_large(int numLarge, const int* __restrict__ largeMolecules,
                                       const int* __restrict__ moleculeStart,
                                       const int* __restrict__ moleculeAtoms,
                                       const float4* __restrict__ posq,
                                       float scaleMinusOne,
                                       float4* __restrict__ delta)
{
    __shared__ float3 shared[NV_BAROSTAT_LARGE_BLOCK];   // CUDA built-in; 3 KB

    for (int idx = blockIdx.x; idx < numLarge; idx += gridDim.x) {
        const int mol = largeMolecules[idx];
        const int first = moleculeStart[mol];
        const int last = moleculeStart[mol + 1];
        const int numAtoms = last - first;

        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        for (int slot = first + threadIdx.x; slot < last; slot += blockDim.x) {
            const float4 p = posq[moleculeAtoms[slot]];
            sx += p.x; sy += p.y; sz += p.z;
        }
        shared[threadIdx.x].x = sx;
        shared[threadIdx.x].y = sy;
        shared[threadIdx.x].z = sz;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared[threadIdx.x].x += shared[threadIdx.x + stride].x;
                shared[threadIdx.x].y += shared[threadIdx.x + stride].y;
                shared[threadIdx.x].z += shared[threadIdx.x + stride].z;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float inv = 1.0f / (float)numAtoms;
            delta[mol] = make_float4(shared[0].x * inv * scaleMinusOne,
                                     shared[0].y * inv * scaleMinusOne,
                                     shared[0].z * inv * scaleMinusOne, 0.0f);
        }
        __syncthreads();   // shared is reused by the next molecule this block takes
    }
}

// Flat over atom slots: every thread moves exactly one atom, so this pass does
// not care how the molecule sizes are distributed.
__global__ void cuda_apply_delta(int numAtomSlots,
                                   const int* __restrict__ moleculeAtoms,
                                   const int* __restrict__ atomMolecule,
                                   const float4* __restrict__ delta,
                                   float4* __restrict__ posq)
{
    for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < numAtomSlots;
         slot += gridDim.x * blockDim.x) {
        const int atom = moleculeAtoms[slot];
        const float4 d = delta[atomMolecule[slot]];
        float4 p = posq[atom];
        p.x += d.x;
        p.y += d.y;
        p.z += d.z;
        posq[atom] = p;
    }
}

// The snapshot copies of saveState() and restoreState(): positions, velocities and
// forces copied in one launch instead of three device-to-device copies, which on a
// barostat step saves the host time of two copy calls each way. Each array is moved
// as 16-byte words (every array's size is a multiple of 16 bytes: the padded atom
// count is a multiple of 32), so the bytes arrive unchanged.
__global__ void cuda_copy_snapshot(const uint4* __restrict__ src0, uint4* __restrict__ dst0, int n0,
                                   const uint4* __restrict__ src1, uint4* __restrict__ dst1, int n1,
                                   const uint4* __restrict__ src2, uint4* __restrict__ dst2, int n2) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n0 + n1 + n2; i += stride) {
        if (i < n0) dst0[i] = src0[i];
        else if (i < n0 + n1) dst1[i - n0] = src1[i - n0];
        else dst2[i - n0 - n1] = src2[i - n0 - n1];
    }
}

void CudaApplyMonteCarloBarostatKernel::initialize(const OpenMM::System& system,
                                                     const MonteCarloBarostat& barostat) {
    if (cc.getUseDoublePrecision())
        throw OpenMM::OpenMMException("MonteCarloBarostat: double precision is not supported; "
                              "use single or mixed precision");
    // Everything else needs the molecule decomposition, which only the
    // ContextImpl can supply -- see ensureInitialized().
}

void CudaApplyMonteCarloBarostatKernel::ensureInitialized(OpenMM::ContextImpl& context) {
    if (hasInitializedMolecules)
        return;
    hasInitializedMolecules = true;

    const vector<vector<int> >& molecules = context.getMolecules();
    numMolecules = (int) molecules.size();
    if (numMolecules == 0)
        throw OpenMM::OpenMMException("MonteCarloBarostat: the System has no molecules to scale");

    // Flatten the decomposition and, in the same pass, record which molecule owns
    // each slot and sort the molecules into the two size classes.
    vector<int> atoms, start(numMolecules + 1), owner, small, large;
    atoms.reserve(cc.getNumAtoms());
    owner.reserve(cc.getNumAtoms());
    int index = 0;
    for (int m = 0; m < numMolecules; m++) {
        start[m] = index;
        for (int atom : molecules[m]) {
            atoms.push_back(atom);
            owner.push_back(m);
            index++;
        }
        if ((int) molecules[m].size() <= NV_BAROSTAT_WARP_MOLECULE_MAX)
            small.push_back(m);
        else
            large.push_back(m);
    }
    start[numMolecules] = index;
    numAtomSlots = index;
    numSmall = (int) small.size();
    numLarge = (int) large.size();

    moleculeAtoms.initialize<int>(cc, numAtomSlots, "nvBarostatMoleculeAtoms");
    moleculeStart.initialize<int>(cc, numMolecules + 1, "nvBarostatMoleculeStart");
    atomMolecule.initialize<int>(cc, numAtomSlots, "nvBarostatAtomMolecule");
    centroidDelta.initialize<OpenMM::mm_float4>(cc, numMolecules, "nvBarostatCentroidDelta");
    moleculeAtoms.upload(atoms);
    moleculeStart.upload(start);
    atomMolecule.upload(owner);
    // ComputeArray rejects a zero-length allocation, and a System can legitimately
    // be all-small (pure solvent) or all-large (one molecule), so only allocate a
    // class that is actually populated. The launcher skips an empty class.
    if (numSmall > 0) {
        smallMolecules.initialize<int>(cc, numSmall, "nvBarostatSmallMolecules");
        smallMolecules.upload(small);
    }
    if (numLarge > 0) {
        largeMolecules.initialize<int>(cc, numLarge, "nvBarostatLargeMolecules");
        largeMolecules.upload(large);
    }

    // Buffers for the pre-move snapshot, one element per padded atom slot
    // (three for the forces, which are stored as all x, then all y, then all z).
    // Velocities are double4 in mixed precision and float4 in single; positions
    // are float4 in both.
    const int paddedAtoms = cc.getPaddedNumAtoms();
    preMovePosq.initialize(cc, paddedAtoms, sizeof(OpenMM::mm_float4), "nvBarostatSavedPositions");
    preMoveVelm.initialize(cc, paddedAtoms,
                           cc.getUseMixedPrecision() ? sizeof(OpenMM::mm_double4) : sizeof(OpenMM::mm_float4),
                           "nvBarostatSavedVelocities");
    preMoveForces.initialize<long long>(cc, 3 * (size_t) paddedAtoms, "nvBarostatSavedLongForces");
}

// Copies the three per-atom arrays between the context and the snapshot, in one
// launch on the context's current stream.
void CudaApplyMonteCarloBarostatKernel::copySnapshot(bool toSnapshot) {
    OpenMM::CudaContext& cu = static_cast<OpenMM::CudaContext&>(cc);
    OpenMM::ArrayInterface* live[3] = {&cc.getPosq(), &cc.getVelm(), &cc.getLongForceBuffer()};
    OpenMM::ArrayInterface* kept[3] = {&preMovePosq, &preMoveVelm, &preMoveForces};
    const uint4* src[3];
    uint4* dst[3];
    int words[3];
    for (int j = 0; j < 3; j++) {
        OpenMM::ArrayInterface& from = toSnapshot ? *live[j] : *kept[j];
        OpenMM::ArrayInterface& to = toSnapshot ? *kept[j] : *live[j];
        const size_t bytes = (size_t) from.getSize() * from.getElementSize();
        if (bytes != (size_t) to.getSize() * to.getElementSize() || bytes % sizeof(uint4) != 0)
            throw OpenMM::OpenMMException("MonteCarloBarostat: snapshot buffer does not match the context's array");
        src[j] = dptr<const uint4>(cu, from);
        dst[j] = dptr<uint4>(cu, to);
        words[j] = (int) (bytes / sizeof(uint4));
    }
    const int block = 256;
    const int total = words[0] + words[1] + words[2];
    const int grid = std::max(1, std::min(4096, (total + block - 1) / block));
    cuda_copy_snapshot<<<grid, block, 0, cu.getCurrentStream()>>>(src[0], dst[0], words[0], src[1], dst[1], words[1],
                                                                  src[2], dst[2], words[2]);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw OpenMM::OpenMMException("MonteCarloBarostat: snapshot copy launch failed: " +
                                      std::string(cudaGetErrorString(err)) + nvDeviceFaultHelp(err));
}

void CudaApplyMonteCarloBarostatKernel::saveState(OpenMM::ContextImpl& context) {
    OpenMM::ContextSelector selector(cc);
    ensureInitialized(context);
    // Everything a trial move can change in the per-atom state: the scaling
    // and any rewrap change positions and wrap counts, and an atom sort during
    // the trial evaluation permutes positions, velocities and forces and
    // changes the atom order. The low-order position part of mixed precision
    // is left out on purpose; see restoreState().
    copySnapshot(true);
    preMoveCellOffsets = cc.getPosCellOffsets();
    preMoveAtomOrder = cc.getAtomIndex();
    // A new move starts; whether atoms were sorted is recorded in scaleCoordinates().
    reorderedBeforeMove = false;
}

void CudaApplyMonteCarloBarostatKernel::scaleCoordinates(OpenMM::ContextImpl& context, double scale) {
    OpenMM::ContextSelector selector(cc);
    ensureInitialized(context);
    // The context's flag still describes the force evaluation that came just
    // before saveState(), because nothing has been evaluated since. Record it
    // now, before the trial evaluation overwrites it.
    reorderedBeforeMove = cc.getAtomsWereReordered();

    OpenMM::CudaContext& cu = static_cast<OpenMM::CudaContext&>(cc);
    cudaStream_t stream = cu.getCurrentStream();
    const float scaleMinusOne = (float) scale - 1.0f;

    const int* const molStart   = dptr<const int>(cu, moleculeStart);
    const int* const molAtoms   = dptr<const int>(cu, moleculeAtoms);
    const int* const atomMol    = dptr<const int>(cu, atomMolecule);
    const float4* const posqIn  = dptr<const float4>(cu, cc.getPosq());
    float4* const posqOut       = dptr<float4>(cu, cc.getPosq());
    float4* const delta         = dptr<float4>(cu, centroidDelta);

    // Centroids, split by molecule size so neither regime starves the other.
    if (numSmall > 0) {
        const int block = 128;                       // 4 warps -> 4 molecules per block
        const int warpsPerBlock = block / 32;
        const int grid = std::min(65535, (numSmall + warpsPerBlock - 1) / warpsPerBlock);
        cuda_centroids_small<<<grid, block, 0, stream>>>(
            numSmall, dptr<const int>(cu, smallMolecules), molStart, molAtoms,
            posqIn, scaleMinusOne, delta);
    }
    if (numLarge > 0) {
        const int grid = std::min(65535, numLarge);
        cuda_centroids_large<<<grid, NV_BAROSTAT_LARGE_BLOCK, 0, stream>>>(
            numLarge, dptr<const int>(cu, largeMolecules), molStart, molAtoms,
            posqIn, scaleMinusOne, delta);
    }

    // Displacement: flat over atom slots, so it does not care how the molecule
    // sizes are distributed.
    {
        const int block = 256;
        const int grid = std::min(65535, (numAtomSlots + block - 1) / block);
        cuda_apply_delta<<<grid, block, 0, stream>>>(
            numAtomSlots, molAtoms, atomMol, delta, posqOut);
    }
    // A context an earlier kernel already killed fails here too, so the error
    // carries the advice on finding that kernel when it means a device fault.
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw OpenMM::OpenMMException("MonteCarloBarostat: coordinate scaling launch failed: " +
                                      std::string(cudaGetErrorString(err)) +
                                      nvDeviceFaultHelp(err));
}

void CudaApplyMonteCarloBarostatKernel::restoreState(OpenMM::ContextImpl& context) {
    OpenMM::ContextSelector selector(cc);
    // The caller has already put the original box back. Put back the per-atom
    // state saved before the move. The forces are restored too, so the next
    // integration step uses the forces of the restored configuration without
    // a new evaluation.
    copySnapshot(false);
    cc.setPosCellOffsets(preMoveCellOffsets);
    // If the atoms were sorted during the move, put the old order back. Setting
    // the order also makes every force re-permute its per-atom parameters to
    // match, so later force evaluations agree with the restored positions. The
    // order is set again as well when the sort happened in the evaluation just
    // before saveState(); the order does not change then, but the forces'
    // re-permutation still runs.
    //
    // Not restored: the low-order position part used in mixed precision. When
    // atoms have been wrapped, restoring the box rewrites that part for the
    // trial configuration, and it is left as it is. The context's own "atoms
    // were reordered" flag is also left as the trial evaluation set it.
    if (reorderedBeforeMove || cc.getAtomsWereReordered())
        cc.setAtomIndex(preMoveAtomOrder);
}

} // namespace NvOpenMM
