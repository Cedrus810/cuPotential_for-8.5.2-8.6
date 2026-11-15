// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once
#include "cupo/visibility.h"
#include "cupo/pair.h"
#include "cupo/status.h"
#include "cupo/cell_list.h"
#include "cupo/kspace.h"
#include <driver_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Direct-space kernel settings and current-device properties. */
typedef struct {
    int tuning_threads;             /**< Threads per block. */
    int tuning_blocks_per_sm;       /**< Minimum blocks per SM; zero when unspecified. */
    int sm_count;                   /**< Number of multiprocessors on the current device. */
    int launch_blocks_per_grid;     /**< Launch grid size; zero when workspace-dependent. */
    int shared_mem_bytes;           /**< Kernel shared memory per block, in bytes. */
    int cuda_runtime_version;       /**< CUDA runtime version encoded by cudaRuntimeGetVersion. */
} cupo_dspace_build_info;

/** Read the direct-space kernel settings and current CUDA device properties.
 * @param[out] out Host result; zeroed on failure when valid.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_dspace_get_build_info(cupo_dspace_build_info* out);

/** Direct-space workspace. Calls on the same workspace must not overlap. */
typedef struct cupo_dspace_state* cupo_dspace_handle;
#define CUPO_DSPACE_DESC_VERSION 9

/** Reciprocal exclusion weights for a custom pair evaluator.
 * Zero every field to omit a component. The plan must match the evaluator,
 * reciprocal calculation, and cell-list cutoff. Its constants are copied at
 * creation. Weights are borrowed until the workspace and its graphs are destroyed;
 * update them with stream ordering and remap them when input atoms are reordered.
 * Weight values and input atom ordering must match the reciprocal calculation.
 * Keep weights stable during direct and reciprocal evaluation. They must not
 * overlap workspace scratch or writable outputs.
 */
typedef struct {
    cupo_splitting_plan_handle plan; /**< P1 Coulomb or attractive P6 dispersion plan. */
    const float* d_weights;         /**< FP32-aligned device weights in input atom order: q_i or sqrt(C6_i). */
    uint64_t weight_stride_bytes;  /**< Byte stride between weights; a multiple of sizeof(float), at least sizeof(float). */
    uint64_t weight_count;         /**< Number of weights; must cover every input atom. */
} cupo_dspace_exclusion_weights;

/** Direct-space pair configuration and exclusion topology.
 * The cell list must match the standard plan's cutoff and coupling, or the custom
 * particle size and alignment. The cell list must outlive this workspace and its
 * graphs. Descriptor storage and splitting plans are needed only during creation.
 *
 * Exclusion pairs contain distinct input atom indices with no duplicate pairs.
 * Creation validates the initial topology. Standard pairs copy the topology;
 * input atom permutations that change these indices require a new workspace.
 * Custom pairs borrow the array until the workspace and its graphs are destroyed,
 * and read it on each execute. Remap indices in place with stream ordering when
 * input atoms are reordered, keeping the count and topology valid. Cell-list
 * spatial sorting needs no topology update.
 *
 * Direct contributions of excluded pairs are always cancelled. Reciprocal
 * corrections use periodic minimum-image distances, including beyond cutoff.
 * Standard pairs use live packed particle weights and the pair's P1/P6 plans;
 * both explicit bindings below must be empty. Custom pairs correct only the
 * explicitly bound components, which may use different splitting families.
 * Physical replacement interactions are supplied separately through the tuples API.
 */
typedef struct {
    uint32_t struct_bytes;          /**< Set to sizeof(cupo_dspace_desc). */
    uint32_t abi_version;           /**< Set to CUPO_DSPACE_DESC_VERSION. */
    cupo_pair_desc pair;            /**< Pair evaluator and particle layout. */
    cupo_cell_list_handle cell_list; /**< Borrowed cell list. */
    int n_exclusion_pairs;          /**< Fixed number of excluded pairs, at least zero. */
    const int* d_exclusion_pairs;   /**< Device int[2*n_exclusion_pairs], packed as {i,j}; NULL allowed for no pairs. */
    int reciprocal_exclusions;     /**< 1 also subtracts reciprocal interactions; 0 cancels only direct pairs. */
    cupo_dspace_exclusion_weights coulomb_exclusions; /**< Custom P1 weights; requires reciprocal_exclusions=1. */
    cupo_dspace_exclusion_weights dispersion_exclusions; /**< Custom P6 weights; requires reciprocal_exclusions=1. */
} cupo_dspace_desc;

/** Compute the device arena size without allocating, launching, or linking code.
 * @param[in] desc Host workspace description, also used for creation.
 * @param[out] out_device_bytes Required device bytes; zeroed on failure when valid.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_dspace_workspace_size(
    const cupo_dspace_desc* desc, unsigned long long* out_device_bytes);

/** Create a workspace and link any custom pair code.
 * Call outside CUDA graph capture, with no dependent work in flight. The workspace
 * owns linked code; descriptor and input IR storage may be released after creation.
 * There are no per-step allocations.
 * @param[in] desc Host workspace description.
 * @param[in,out] device_arena Caller-owned device memory, aligned to 256 bytes and kept until destroy.
 * @param[in] bytes Arena capacity, at least the workspace_size result.
 * @param[out] out Host handle output; set to NULL on failure when valid.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_dspace_create(
    const cupo_dspace_desc* desc, void* device_arena, unsigned long long bytes,
    cupo_dspace_handle* out);

/** Destroy a workspace without freeing the caller's device arena.
 * Call outside CUDA graph capture, after dependent work completes and dependent
 * graphs are destroyed.
 * @param[in] state Workspace handle; NULL is accepted.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_dspace_destroy(cupo_dspace_handle state);

/** Optionally pick the direct-space kernel while timing it against reciprocal work.
 * Times the complete force-only direct branch (clear, exclusions, pairs, fixed64
 * finish) concurrently with this reciprocal workspace's spread/convolve/gather,
 * through their join. Both workspaces must borrow the SAME cell list; build it and
 * refresh/tune reciprocal influence and convolution first. Call before the first
 * direct execute and before launching production reciprocal work, outside capture,
 * with both workspaces idle. Uses temporary private force storage and preserves
 * direct accumulators and prepare flags. Reciprocal charge/potential/spectrum
 * scratch is overwritten and rebuilt by the next execute. Never writes caller
 * output buffers, changes the convolution policy, or advances coordinates; no
 * dependency is retained. Selection is fixed for the workspace lifetime, including
 * box/parameter updates. Repeated calls are no-ops after selection, as are
 * forced/population/disabled selectors or CUPO_DSPACE_AUTOTUNE_OBJECTIVE=direct.
 * The default objective is concurrent when this hook is called; standalone execute
 * without the hook retains direct-only tuning. Pending tuning during capture is
 * rejected. Errors are reported rather than silently changing the objective.
 * @param[in] state Direct-space workspace handle.
 * @param[in] reciprocal Reciprocal workspace sharing the same cell list.
 * @param[in] d_box Device FP32 box in cupo/box.h layout.
 * @param[in] padded_natoms Eventual force destination stride.
 * @param[in] stream CUDA stream ordering input readiness (including any concurrent direct prepare).
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_dspace_tune_with_kspace(
    cupo_dspace_handle state, cupo_kspace_handle reciprocal, float* d_box,
    int padded_natoms, cudaStream_t stream);

/** Multi-workspace form of the startup selector above.
 * reciprocals is a host array of count distinct, non-NULL handles (count > 0),
 * all borrowing the same cell list as state. Each trial executes them
 * sequentially on one reciprocal stream, in array order, concurrently with
 * direct space, and joins after ALL reciprocal work. Pass Coulomb then
 * dispersion for OpenMM LJPME. Each workspace must have its influence/backend
 * prepared; their convolution policies are preserved. The array and handles are
 * borrowed only for the duration of this call. The single-workspace API remains
 * equivalent to count == 1.
 * @param[in] state Direct-space workspace handle.
 * @param[in] reciprocals Host array of reciprocal workspace handles.
 * @param[in] count Number of handles in reciprocals.
 * @param[in] d_box Device FP32 box in cupo/box.h layout.
 * @param[in] padded_natoms Eventual force destination stride.
 * @param[in] stream CUDA stream ordering input readiness.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_dspace_tune_with_kspaces(
    cupo_dspace_handle state, const cupo_kspace_handle* reciprocals, int count,
    float* d_box, int padded_natoms, cudaStream_t stream);

/** Clear the requested internal accumulators before a direct-space calculation.
 * Prepare may overlap cell-list construction. Order both before execute, and order
 * execute before finish. Use the same energy/force flags for prepare and execute.
 * Unrequested accumulators retain their previous values, initially zero.
 * @param[in] state Workspace handle.
 * @param[in] compute_energy 1 to clear the energy accumulator; 0 to leave it unchanged.
 * @param[in] compute_forces 1 to clear the force accumulator; 0 to leave it unchanged.
 * @param[in] stream CUDA stream for the queued work.
 * @return CUPO_SUCCESS when work is enqueued, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_dspace_prepare(
    cupo_dspace_handle state, int compute_energy, int compute_forces, cudaStream_t stream);

/** Evaluate cutoff pair interactions and apply exclusion corrections.
 * Custom evaluators must follow the repeatability and excluded-pair requirements
 * in cupo/pair_device.cuh. Setting both output flags to zero performs no work.
 * @param[in] state Workspace prepared with the same output flags.
 * @param[in] d_box Device FP32 box in cupo/box.h layout; NULL allowed when both flags are zero.
 * @param[in] compute_energy 1 to compute energy, 0 otherwise.
 * @param[in] compute_forces 1 to compute forces, 0 otherwise.
 * @param[in] stream CUDA stream ordered after prepare and cell-list construction.
 * @return CUPO_SUCCESS when work is enqueued, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_dspace_execute(
    cupo_dspace_handle state, float* d_box, int compute_energy, int compute_forces,
    cudaStream_t stream);

/** Write the accumulated energy and add forces to the caller's output.
 * Energy and force use physical units. Scaled force contributions and the final
 * destination must fit signed fixed64 with 32 fractional bits (about +/-2^31
 * force units). Reading retained forces requires the same cell-list build used
 * to compute them.
 * @param[in] state Workspace handle.
 * @param[out] d_energy Device FP64 scalar, overwritten; NULL skips energy output.
 * @param[in,out] d_force Device fixed64 array in input atom order, with x[P], y[P], z[P] planes; NULL skips forces.
 * @param[in] padded_natoms P, the force-array stride; at least the atom count when d_force is non-NULL.
 * @param[in] force_scale Finite force multiplier, normally 1; does not scale energy.
 * @param[in] stream CUDA stream ordered after execute.
 * @return CUPO_SUCCESS when work is enqueued, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_dspace_finish(
    cupo_dspace_handle state, double* d_energy, unsigned long long* d_force,
    int padded_natoms, float force_scale, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
