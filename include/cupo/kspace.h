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
#include "cupo/status.h"
#include "cupo/cell_list.h"       /* cupo_cell_list_handle */
#include "cupo/splitting_plan.h"  /* cupo_splitting_plan_handle */
#include <driver_types.h>         /* cudaStream_t */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Reciprocal-space workspace. */
typedef struct cupo_kspace_impl* cupo_kspace_handle;

/**
 * @file
 * Reciprocal-space energy and forces from a splitting plan and cell list.
 *
 * Functions return cupo_status; cupo_last_error() describes failures on the
 * calling host thread and is cleared on success. No C++ exception crosses this API.
 *
 * Execution and influence refresh enqueue work on the supplied CUDA stream,
 * without allocation or synchronization, and support graph capture. Success
 * means work was enqueued; check CUDA synchronization for device errors. Serialize
 * uses of each workspace and keep its buffers alive through queued work and replay.
 * Build the cell list before spreading; keep the list, box and weights unchanged
 * through interpolation. Callers provide ordering between streams.
 *
 * Device box inputs contain FP32 values in cupo/box.h order:
 * [a_x,b_y,c_z,b_x,c_x,c_y]. Orthorhombic workspaces read only the first three.
 * Forces and energies include the splitting plan's physical coupling factor.
 */

/** FFT-size rounding policy, independent of backend and retained Fourier modes.
 * cuFFT can execute either policy's dimensions. The compact backend requires
 * compiled dimensions and device support, even when retaining the full spectrum.
 */
typedef enum cupo_grid_policy {
    CUPO_GRID_DENSE = 0,   /* Prolate: measured cuFFT sizes, then 7-smooth. Ewald: 7-smooth. */
    CUPO_GRID_COMPACT = 1  /* Compiled compact sizes, at most 512 per axis. */
} cupo_grid_policy;

/** Choose mesh dimensions for a splitting plan without accessing a CUDA device.
 * Implemented in libcupo; no standalone grid-planning object is required.
 * All pointers must be non-null and naturally aligned.
 * @param[in] plan Splitting plan; compact sizing supports Coulomb Prolate P1 only.
 * @param[in] box_edges Host FP64 array [x, y, z] in nm, positive and finite.
 * Include any desired NPT expansion in these lengths.
 * @param[in] policy Dense or compact FFT-size rounding; device support is separate.
 * @param[out] out_mesh Host integer array [nx, ny, nz], each at least six.
 * Unchanged on failure.
 * @return CUPO_SUCCESS, CUPO_INVALID_VALUE for invalid arguments, or
 * CUPO_INVALID_SIZE if the plan/policy has no mesh covering the box.
 * cupo_last_error() describes failures and is cleared on success.
 */
CUPO_API cupo_status cupo_plan_grid(
    cupo_splitting_plan_handle plan, const double box_edges[3],
    cupo_grid_policy policy, int out_mesh[3]);

#define CUPO_KSPACE_DESC_VERSION 1

/**
 * Workspace configuration. Zero-initialize, then set the size and version fields.
 * Mesh dimensions and all bindings are fixed at creation.
 *
 * Explicit weights follow the input atom order of cupo_cell_list_execute, before
 * spatial sorting. They replace posq.w in both spreading and interpolation:
 * electric charges for Coulomb plans, sqrt(C6) for dispersion plans. A shared
 * cell list containing Coulomb charges needs explicit weights for dispersion.
 *
 * The caller owns explicit weights for the workspace lifetime. Values may change
 * between evaluations, with stream ordering, but must remain unchanged between
 * spreading and interpolation. They must not overlap workspace storage or writable
 * outputs. Direct-space, self, background and exclusion terms must use matching weights.
 */
typedef struct {
    uint32_t struct_bytes; /**< sizeof(cupo_kspace_desc). */
    uint32_t abi_version; /**< CUPO_KSPACE_DESC_VERSION. */
    int grid_nx, grid_ny, grid_nz; /**< Mesh axes, each >= spreading width; product <= INT_MAX. */
    cupo_splitting_plan_handle plan; /**< Required plan; copied at creation. */
    cupo_cell_list_handle cell_list; /**< Borrowed list; must outlive the workspace. NULL allowed for sizing. */
    const float* d_charges; /**< Optional FP32-aligned device weights; NULL uses the cell list's posq.w. */
    uint64_t charge_stride_bytes; /**< Weight stride: multiple of sizeof(float), at least sizeof(float); 0 for NULL weights. */
    uint64_t charge_count; /**< Available weights, at least the cell list's atom count; 0 for NULL weights. */
} cupo_kspace_desc;

/**
 * Query caller-owned workspace storage without allocation or CUDA work.
 * The result excludes memory owned by cuFFT and configured convolution backends.
 *
 * @param[in] desc Host configuration with a valid plan; cell_list may be NULL.
 * @param[out] out_device_bytes Optional host byte count for the device arena; zero on failure.
 * @param[out] out_host_bytes Optional host byte count; always zero.
 * @return CUPO_SUCCESS on success, or an error status for an invalid descriptor or mesh.
 */
CUPO_API cupo_status cupo_kspace_workspace_size(
    const cupo_kspace_desc* desc,
    unsigned long long* out_device_bytes, unsigned long long* out_host_bytes);

/**
 * Create a reciprocal-space workspace and its FFT plans on the current device.
 * The splitting plan is copied; the cell list, arena and explicit weights are borrowed.
 *
 * @param[in] desc Host configuration with a valid plan and cell list.
 * @param[in] device_arena Caller-owned device storage, aligned to 256 bytes; free after destroy.
 * @param[in] bytes Arena size in bytes, at least the workspace_size result.
 * @param[out] out Required host handle destination; NULL on failure.
 * @return CUPO_SUCCESS on success, or an error status.
 */
CUPO_API cupo_status cupo_kspace_create(
    const cupo_kspace_desc* desc,
    void* device_arena, unsigned long long bytes, cupo_kspace_handle* out);

/**
 * Release a workspace and its owned FFT resources, leaving caller-owned storage intact.
 * Complete queued work and destroy graph executions using the workspace first.
 *
 * @param[in] handle Workspace to destroy, or NULL.
 * @return CUPO_SUCCESS on success (including NULL), or an error status.
 */
CUPO_API cupo_status cupo_kspace_destroy(cupo_kspace_handle handle);

/** Execution backend, independent of the spectrum policy. */
typedef enum {
    CUPO_KSPACE_BACKEND_CUFFT = 0, /**< Always use cuFFT. */
    CUPO_KSPACE_BACKEND_COMPACT = 1, /**< Require compact support for the build, device and mesh. */
    CUPO_KSPACE_BACKEND_AUTO = 2 /**< Prepare compact when supported; allow tuning to select either backend. */
} cupo_kspace_convolution_backend;

/** Fourier modes retained by either execution backend. */
typedef enum {
    CUPO_KSPACE_SPECTRUM_FULL = 0, /**< Retain all mesh modes. */
    CUPO_KSPACE_SPECTRUM_TRUNCATED_RADIAL = 1, /**< P1 Prolate only: retain 2*pi*cutoff*|k| <= 10.75. */
    CUPO_KSPACE_SPECTRUM_TRUNCATED_BOX = 2 /**< Retain |m_i| <= support[i] along each mesh axis. */
} cupo_kspace_spectrum_policy;

#define CUPO_KSPACE_CONVOLUTION_DESC_VERSION 1

/**
 * Convolution policy. Zero-initialize, then set the size and version fields.
 * The defaults are CUFFT and FULL.
 *
 * Radial truncation uses the current box's reciprocal lattice, including triclinic
 * terms. Its support covers the cell list's accepted box envelope and stays fixed
 * across accepted box changes. Sharp truncation can cause energy jumps as modes
 * cross the cutoff. Both backends apply the same policy to potential and energy.
 */
typedef struct {
    uint32_t struct_bytes; /**< sizeof(cupo_kspace_convolution_desc). */
    uint32_t abi_version; /**< CUPO_KSPACE_CONVOLUTION_DESC_VERSION. */
    cupo_kspace_convolution_backend backend; /**< Requested execution backend. */
    cupo_kspace_spectrum_policy spectrum; /**< Requested Fourier-mode policy. */
    int support[3]; /**< TRUNCATED_BOX radii in [0,grid_axis/2]; all zeros for other policies. */
} cupo_kspace_convolution_desc;

/**
 * Set the execution backend and spectrum policy, and reset tuning results.
 * Preserves grid addresses and rebuilds existing influence tables if the policy changes.
 * Call on the owning device, outside capture, after queued work completes and graphs
 * using convolution or influence refresh have been destroyed. May allocate and synchronize.
 *
 * @param[in] handle Workspace to configure.
 * @param[in] desc Host convolution policy.
 * @return CUPO_SUCCESS on success; an error preserves the previous configuration.
 *         COMPACT returns CUPO_NOT_SUPPORTED for an unsupported build, device or mesh.
 *         AUTO falls back to cuFFT for those cases; allocation and device failures are errors.
 */
CUPO_API cupo_status cupo_kspace_configure_convolution(
    cupo_kspace_handle handle, const cupo_kspace_convolution_desc* desc);

#define CUPO_KSPACE_CONVOLUTION_INFO_VERSION 1

/**
 * Convolution configuration and tuning results.
 * Initialize the size and version fields before querying; all other fields are outputs.
 * Arrays of backends and times are ordered potential-only, energy-only, then both.
 * Fresh workspaces report CUFFT/FULL, half-mesh support and zero timings.
 */
typedef struct {
    uint32_t struct_bytes; /**< sizeof(cupo_kspace_convolution_info). */
    uint32_t abi_version; /**< CUPO_KSPACE_CONVOLUTION_INFO_VERSION. */
    cupo_kspace_convolution_backend requested_backend; /**< Configured backend policy. */
    cupo_kspace_spectrum_policy spectrum; /**< Configured Fourier-mode policy. */
    int support[3]; /**< Resolved support radii, shared by both backends. */
    int compact_available; /**< Nonzero when a compact candidate is available, even if not selected. */
    int tuned; /**< Nonzero after benchmarking; zero if compact is unavailable. */
    cupo_kspace_convolution_backend backend[3]; /**< Selected CUFFT or COMPACT backend for each output mode. */
    float compact_us[3]; /**< Compact convolution times in microseconds, including energy reduction; zero until measured. */
    float cufft_us[3]; /**< cuFFT convolution times in microseconds, including energy reduction; zero until measured. */
    float setup_ms; /**< Time spent tuning in milliseconds; zero until measured. */
} cupo_kspace_convolution_info;

/**
 * Read the current convolution configuration and tuning results without device work.
 *
 * @param[in] handle Workspace to query.
 * @param[in,out] out Host result with struct_bytes and abi_version initialized; unchanged on error.
 * @return CUPO_SUCCESS on success, or an error status.
 */
CUPO_API cupo_status cupo_kspace_get_convolution_info(
    cupo_kspace_handle handle, cupo_kspace_convolution_info* out);

/**
 * Benchmark AUTO candidates and select a backend for each output mode.
 * Call after refresh_influence, before graph capture, on the owning device.
 * Synchronizes stream and allocates temporary storage; no workspace work or graph
 * execution may run concurrently. Charge, potential and influence grids are preserved;
 * spectrum scratch is unspecified afterward.
 *
 * @param[in] handle Workspace configured with backend=AUTO. Results persist until reconfiguration.
 * @param[in] stream CUDA stream carrying prior workspace work; active capture is rejected.
 * @return CUPO_SUCCESS on success, including already-tuned or unsupported-compact no-ops;
 *         an error status for invalid configuration, allocation or execution failure.
 */
CUPO_API cupo_status cupo_kspace_tune_convolution(
    cupo_kspace_handle handle, cudaStream_t stream);

/**
 * Check compact convolution support on the current CUDA device without creating a plan.
 * Call outside graph capture. Allocates no convolution workspace or FFT tables.
 *
 * @param[in] mesh Host array of three positive mesh dimensions; product must fit INT_MAX.
 * @return CUPO_SUCCESS if supported, CUPO_NOT_SUPPORTED for an unsupported build,
 *         device or FFT size, or another error status for invalid input or CUDA failure.
 */
CUPO_API cupo_status cupo_kspace_check_compact_mesh(const int mesh[3]);

/**
 * Clear and populate the workspace charge grid from the bound cell list and weights.
 * Build the cell list first; refresh influence after box changes.
 *
 * @param[in] handle Workspace to use.
 * @param[in] d_box Device FP32 box in cupo/box.h layout, matching the cell list.
 * @param[in] stream CUDA stream for the operation.
 * @return CUPO_SUCCESS when work is enqueued, or an error status.
 */
CUPO_API cupo_status cupo_kspace_spread(
    cupo_kspace_handle handle, float* d_box, cudaStream_t stream);

/**
 * Convolve the charge grid to compute reciprocal energy and/or the internal potential.
 * Uses the configured backend and spectrum; leaves the charge grid unchanged.
 * Energy or potential output requires refreshed influence tables. With neither output,
 * runs only a full forward transform, regardless of the configured policy.
 *
 * @param[in] handle Workspace containing a populated charge grid.
 * @param[out] d_energy Optional device FP64 scalar, overwritten with physical reciprocal energy.
 * @param[in] compute_potential Nonzero to compute the internal FP32 potential for interpolation.
 * @param[in] stream CUDA stream for the operation.
 * @return CUPO_SUCCESS when work is enqueued, or an error status.
 */
CUPO_API cupo_status cupo_kspace_convolve(
    cupo_kspace_handle handle, double* d_energy, int compute_potential,
    cudaStream_t stream);

/**
 * Accumulate reciprocal forces from the workspace potential and bound cell list.
 * Run convolve with compute_potential enabled first; use the same box, list and weights.
 *
 * @param[in] handle Workspace containing a computed potential.
 * @param[in] d_box Device FP32 box in cupo/box.h layout.
 * @param[in,out] d_force_fixed64 Optional device force accumulator, in the layout described by execute.
 * @param[in] padded_natom Force component stride in elements; at least the cell list's atom count.
 * @param[in] stream CUDA stream for the operation.
 * @return CUPO_SUCCESS when work is enqueued (or the force pointer is NULL), or an error status.
 */
CUPO_API cupo_status cupo_kspace_interpolate(
    cupo_kspace_handle handle, float* d_box,
    unsigned long long* d_force_fixed64, int padded_natom, cudaStream_t stream);

/**
 * Run spread, convolve and optional force interpolation on one stream.
 * Build the cell list and refresh influence before evaluation. A NULL output skips
 * its computation; with both outputs NULL, only spreading and a full forward transform run.
 *
 * Forces accumulate in input atom order as signed fixed-point values scaled by 2^32,
 * stored in unsigned 64-bit elements: x[0..P), y[0..P), z[0..P), P=padded_natom.
 * The force buffer must not overlap workspace storage, the box or explicit weights.
 * All device pointers require natural alignment for their element type.
 *
 * @param[in] handle Workspace with a built cell list and current influence tables.
 * @param[in] d_box Device FP32 box in cupo/box.h layout, matching the cell list.
 * @param[out] d_energy Optional device FP64 scalar, overwritten with physical reciprocal energy.
 * @param[in,out] d_force_fixed64 Optional caller-owned device array of 3*padded_natom force accumulators.
 * @param[in] padded_natom Force component stride; at least the atom count when forces are requested.
 * @param[in] stream CUDA stream for all stages.
 * @return CUPO_SUCCESS when work is enqueued, or an error status.
 */
CUPO_API cupo_status cupo_kspace_execute(
    cupo_kspace_handle handle,
    float* d_box,
    double* d_energy, unsigned long long* d_force_fixed64,
    int padded_natom, cudaStream_t stream);

/**
 * Rebuild the reciprocal influence tables for the current box.
 * Call before the first energy or potential evaluation and after every box update,
 * ordered before subsequent workspace work. convolve cannot detect a stale box.
 *
 * @param[in] handle Workspace whose influence tables will be updated.
 * @param[in] d_box Device FP32 box in cupo/box.h layout.
 * @param[in] stream CUDA stream for the update.
 * @return CUPO_SUCCESS when work is enqueued, or an error status.
 */
CUPO_API cupo_status cupo_kspace_refresh_influence(
    cupo_kspace_handle handle, const float* d_box, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
