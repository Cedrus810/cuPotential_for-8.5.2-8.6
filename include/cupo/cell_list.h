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
#include "cupo/box.h"
#include <driver_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Cell list shared by direct-space and reciprocal-space workspaces.
 * Keep it alive until those workspaces, their graphs, and queued work are finished.
 */
typedef struct cupo_cell_list_impl* cupo_cell_list_handle;

/** Result of a box update. Rejected updates leave the cell grid unchanged. */
typedef enum {
    CUPO_CELL_LIST_BOX_ABOVE_ENVELOPE = -2, /**< Box exceeds the allocated cell-grid envelope. */
    CUPO_CELL_LIST_BOX_BELOW_FLOOR    = -1, /**< An axis has fewer than 2R+1 cells for stencil radius R. */
    CUPO_CELL_LIST_BOX_IN_BAND        =  0, /**< Cell counts are unchanged; captured graphs remain valid. */
    CUPO_CELL_LIST_BOX_BAND_CROSS     =  1  /**< Cell counts changed; the next execute rebuilds the grid. */
} cupo_cell_list_box_status;

/** Box arrays use the layout in cupo/box.h: [a_x, b_y, c_z, b_x, c_x, c_y].
 * Host arrays always contain six floats. Orthorhombic device arrays need only
 * the first three; triclinic device arrays need all six. Geometry is fixed at
 * creation: zero off-diagonals select orthorhombic geometry, which cannot later
 * accept nonzero off-diagonals. Triclinic boxes require positive diagonals and
 * |b_x| <= a_x/2, |c_x| <= a_x/2, |c_y| <= b_y/2.
 *
 * Standard particle records are FP32 {charge, sigma, epsilon} in physical units.
 * Custom records are copied as opaque bytes. The pair description and standard
 * plans are borrowed only for sizing and creation; their values are retained.
 * Cell-list calls do not read custom IR or globals.
 */

/** Compute workspace sizes without allocating memory or launching device work.
 * Use the same arguments for this query and creation.
 * @param[in] cutoff Pair cutoff in the same length units as box; must match a standard pair's plan.
 * @param[in] box_safety_factor Box scale used to reserve capacity, clamped to at least 1.
 * @param[in] natoms Number of input atoms.
 * @param[in] box Host FP32[CUPO_BOX_FLOATS] initial box.
 * @param[in] pair Host pair description defining the particle layout.
 * @param[out] out_device_bytes Required device arena bytes; NULL skips this output.
 * @param[out] out_host_bytes Required caller-provided host bytes (currently zero); NULL skips this output.
 * @return CUPO_SUCCESS on success, or an error status; valid outputs are zeroed on failure. See cupo_last_error().
 */
CUPO_API cupo_status cupo_cell_list_workspace_size(
    float cutoff, float box_safety_factor, int natoms, const float box[CUPO_BOX_FLOATS],
    const cupo_pair_desc* pair,
    unsigned long long* out_device_bytes, unsigned long long* out_host_bytes);

/** Create a cell list with fixed atom count, particle layout, and geometry.
 * @param[in] cutoff Pair cutoff in box length units; must match a standard pair's plan.
 * @param[in] box_safety_factor Box scale used to reserve capacity, clamped to at least 1.
 * @param[in] natoms Number of input atoms.
 * @param[in] box Host FP32[CUPO_BOX_FLOATS] initial box.
 * @param[in] pair Host pair description defining the particle layout.
 * @param[in,out] device_arena Caller-owned device memory aligned to 256 bytes, kept until destroy; NULL allocates internally.
 * @param[in] bytes Arena capacity, at least out_device_bytes from workspace_size; ignored for an internal allocation.
 * @param[out] out Host handle output; set to NULL on failure when valid.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_cell_list_create(
    float cutoff, float box_safety_factor, int natoms, const float box[CUPO_BOX_FLOATS],
    const cupo_pair_desc* pair,
    void* device_arena, unsigned long long bytes, cupo_cell_list_handle* out);

/** Destroy a cell list and free memory allocated internally by create.
 * Caller-owned arena memory is not freed. Complete queued work and destroy
 * dependent workspaces and graphs before this call.
 * @param[in] handle Cell-list handle; NULL is accepted.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_cell_list_destroy(cupo_cell_list_handle handle);

/** Update the cell-grid dimensions for a new box without reallocating memory.
 * @param[in] handle Cell-list handle.
 * @param[in] box Host FP32[CUPO_BOX_FLOATS] updated box.
 * @param[out] out Host update result, initially set to IN_BAND when valid.
 * @return CUPO_SUCCESS for IN_BAND or BAND_CROSS; CUPO_INVALID_VALUE for a rejected
 * box, leaving the grid unchanged. Floor/envelope failures set the matching out
 * value. Other failures are described by cupo_last_error().
 */
CUPO_API cupo_status cupo_cell_list_update_box(
    cupo_cell_list_handle handle, const float box[CUPO_BOX_FLOATS],
    cupo_cell_list_box_status* out);

/** Rebuild the cell list from positions and particle records on a CUDA stream.
 * Positions are wrapped internally; any periodic image is accepted. Non-finite
 * or unwrappable positions cause a device-side error. Each call handles coordinate
 * changes, input atom reordering, and grid changes. Keep device inputs unchanged
 * until the queued work completes.
 * @param[in] handle Cell-list handle.
 * @param[in] d_box Device FP32 box in the layout above.
 * @param[in] d_xyz Device FP32 positions; atom i starts at i*xyz_per_atom.
 * @param[in] xyz_per_atom Position stride in floats, at least 3 (3 for xyz, 4 for xyzq).
 * @param[in] d_atom_params Device particle records in input atom order, matching the pair size and alignment; NULL allowed for zero-byte records.
 * @param[in] parameter_stride_bytes Byte stride between records, including padding; at least the record size and a multiple of its alignment.
 * @param[in] stream CUDA stream for the queued work.
 * @return CUPO_SUCCESS when work is enqueued, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_cell_list_execute(
    cupo_cell_list_handle handle,
    const float* d_box, const float* d_xyz, int xyz_per_atom,
    const void* d_atom_params, unsigned long long parameter_stride_bytes,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
