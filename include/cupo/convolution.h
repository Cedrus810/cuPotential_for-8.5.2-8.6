// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include "cupo/package_version.h"
#include "cupo/visibility.h"
#include "cupo/status.h"
#include <driver_types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Convolution geometry, grid formats and FFT resources on one CUDA device. */
typedef struct cupo_convolution_plan_impl* cupo_convolution_plan;

/**
 * @file
 * Three-dimensional convolution with configurable Fourier support and grid formats.
 *
 * Functions return cupo_status. Use each plan on its creation device and serialize
 * execution. Create, prepare, rebind and destroy outside graph capture, with no
 * queued work using resources being replaced or freed. Destroy graph executions
 * using a plan or workspace before releasing those resources.
 *
 * FFT arithmetic and intermediate storage use FP32. Real-grid axes support lengths
 * in [2,512] with prime factors only 2,3,5,7,11, subject to compiled device support.
 *
 * Supported GPU build targets are SM75, SM80, SM86, SM89, SM90, SM100, SM103 and
 * SM120; a particular library may include only a subset. The compact_architectures
 * integer array in runtime_build.json (build/runtime_build.json in release
 * tarballs) records that subset; an empty array means the compact backend is
 * disabled. Use cupo_convolution_check_mesh to check support for a mesh on the
 * current CUDA device.
 */

typedef enum {
    CUPO_GRID_FP32 = 0, /**< 32-bit floating point. */
    CUPO_GRID_FP16 = 1, /**< 16-bit floating point. */
    CUPO_GRID_INT16 = 2, /**< Signed 16-bit fixed point. */
    CUPO_GRID_INT32 = 3 /**< Signed 32-bit fixed point. */
} cupo_grid_type;

/**
 * Real-grid element encoding.
 * Integers store round_to_even(value * 2^fractional_bits) and decode as
 * stored * 2^(-fractional_bits). Conversion uses FP32; integer outputs saturate
 * to their type's range and encode NaN as zero.
 */
typedef struct {
    cupo_grid_type type; /**< Element type. */
    uint32_t fractional_bits; /**< Fixed-point fraction bits in [0,126]; must be 0 for FP32/FP16. */
} cupo_grid_format;

/**
 * Check dimension and architecture support on the current CUDA device.
 * Creates no plan or workspace and allocates no FFT tables. CUDA initialization
 * may occur; success does not guarantee a later allocation will succeed.
 *
 * @param[in] nx Real-grid X length.
 * @param[in] ny Real-grid Y length.
 * @param[in] nz Real-grid Z length.
 * @return CUPO_SUCCESS if supported, CUPO_INVALID_SIZE for unsupported dimensions,
 *         CUPO_NOT_SUPPORTED for an unsupported architecture, or a CUDA error status.
 */
CUPO_API cupo_status cupo_convolution_check_mesh(int nx, int ny, int nz);

/**
 * Create a convolution plan and allocate its workspace on the current CUDA device.
 * Retains |kx|<=sx, |ky|<=sy and 0<=kz<=sz; other modes are zero.
 *
 * @param[in] nx Real-grid X length.
 * @param[in] ny Real-grid Y length.
 * @param[in] nz Real-grid Z length.
 * @param[in] sx X support radius in [0,nx/2].
 * @param[in] sy Y support radius in [0,ny/2].
 * @param[in] sz Z support radius in [0,nz/2].
 * @param[in] input_format Input real-grid encoding.
 * @param[in] output_format Output real-grid encoding.
 * @param[out] out Required host plan destination; NULL on failure.
 * @return CUPO_SUCCESS on success, or an error status.
 */
CUPO_API cupo_status cupo_convolution_plan_create(
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    cupo_grid_format input_format,
    cupo_grid_format output_format,
    cupo_convolution_plan* out);

/**
 * Create a plan without allocating mutable workspace; bind storage before execution.
 * May allocate immutable FFT tables. Uses the current CUDA device.
 *
 * @param[in] nx Real-grid X length.
 * @param[in] ny Real-grid Y length.
 * @param[in] nz Real-grid Z length.
 * @param[in] sx X support radius in [0,nx/2].
 * @param[in] sy Y support radius in [0,ny/2].
 * @param[in] sz Z support radius in [0,nz/2].
 * @param[in] input_format Input real-grid encoding.
 * @param[in] output_format Output real-grid encoding.
 * @param[out] workspace_bytes Required host workspace byte count; zero on failure.
 * @param[out] out Required host plan destination; NULL on failure.
 * @return CUPO_SUCCESS on success, or an error status.
 */
CUPO_API cupo_status cupo_convolution_plan_prepare(
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    cupo_grid_format input_format,
    cupo_grid_format output_format,
    uint64_t* workspace_bytes,
    cupo_convolution_plan* out);

/**
 * Bind caller-owned device workspace, replacing and freeing any owned workspace.
 * Wait for queued work and destroy graphs using the old binding before rebinding.
 *
 * @param[in] plan Plan to update.
 * @param[in] workspace Device buffer on the plan's device, aligned to 256 bytes.
 *        Keep it alive through queued work and graph replay; the plan does not free it.
 * @param[in] bytes Buffer size, at least the plan's required workspace byte count.
 * @return CUPO_SUCCESS on success, or an error status; failure preserves the old binding.
 */
CUPO_API cupo_status cupo_convolution_plan_set_workspace(
    cupo_convolution_plan plan, void* workspace, uint64_t bytes);

/**
 * Release a plan and its owned resources, leaving borrowed workspace intact.
 *
 * @param[in] plan Plan to destroy on its CUDA device, or NULL.
 * @return CUPO_SUCCESS on success (including NULL), or an error status.
 */
CUPO_API cupo_status cupo_convolution_plan_destroy(cupo_convolution_plan plan);

/**
 * Query required mutable workspace storage, before or after binding it.
 * Includes alignment padding and excludes immutable FFT tables.
 *
 * @param[in] plan Plan to query.
 * @param[out] bytes Required host byte-count destination; unchanged on error.
 * @return CUPO_SUCCESS on success, or an error status.
 */
CUPO_API cupo_status cupo_convolution_plan_workspace_bytes(cupo_convolution_plan plan,
                                                          uint64_t* bytes);

/**
 * Compute inverse_DFT(H * DFT(input)) and/or export the retained forward spectrum.
 * Both transforms are unnormalized; include normalization and physical factors in H.
 *
 * Real grids are contiguous [nx,ny,nz], Z fastest, in their configured formats.
 * Influence and forward grids are packed [min(nx,2*sx+1),min(ny,2*sy+1),sz+1],
 * Z fastest, without padding. X/Y order is DC, positive, then negative frequencies;
 * full even axes include Nyquist once. Z stores nonnegative frequencies.
 * H must satisfy H(-kx,-ky,kz)=H(kx,ky,kz) at kz=0 and retained Z Nyquist.
 *
 * At least one output is required. Exact input/output aliasing is allowed if the
 * allocation covers both formats; all other overlaps, including workspace overlaps,
 * are rejected. Grids require natural alignment; forward_grid requires 8-byte alignment.
 * FP16 output values must fit the half range; no automatic scaling is applied.
 *
 * Enqueues without allocation or synchronization and supports graph capture.
 * Keep buffers, plan and modules alive through queued work and graph replay.
 *
 * @param[in] plan Plan with allocated or bound workspace.
 * @param[out] output_grid Optional device real-grid result. NULL skips influence multiplication and inverse transforms.
 * @param[out] forward_grid Optional device complex FP32 DFT(input), before multiplication by H; interleaved real/imaginary values.
 * @param[in] input_grid Required device real-grid input in the configured format.
 * @param[in] influence_grid Required device FP32 H in the packed layout; not read when output_grid is NULL.
 * @param[in] stream CUDA stream for all work.
 * @return CUPO_SUCCESS when work is enqueued, or an error status. Check CUDA
 *         synchronization for asynchronous device errors.
 */
CUPO_API cupo_status cupo_convolution_execute(
    cupo_convolution_plan plan,
    void* output_grid,
    void* forward_grid,
    const void* input_grid,
    const float* influence_grid,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
