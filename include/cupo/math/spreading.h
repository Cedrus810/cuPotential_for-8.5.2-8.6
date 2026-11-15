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

/** Scalar functions for the fixed C2 order-3 spreading window.
 * Four weights per axis form a separable 4x4x4 stencil. These host/device
 * functions are supplied as LTO-IR and require a CUDA-aware compiler for the
 * __host__ __device__ qualifiers. Output arrays must be writable from the
 * calling host or device code.
 */

/** Compute the four spreading weights for one grid axis.
 * @param[in] t Fractional grid offset in [0, 1).
 * @param[out] w float[4] receiving weights for grid offsets -1, 0, 1, 2.
 */
__host__ __device__ void cupo_spread_1d_f32(float  t, float  w[4]);

/** Compute the four spreading weights for one grid axis.
 * @param[in] t Fractional grid offset in [0, 1).
 * @param[out] w double[4] receiving weights for grid offsets -1, 0, 1, 2.
 */
__host__ __device__ void cupo_spread_1d_f64(double t, double w[4]);

/** Compute the weight derivatives used for force interpolation along one axis.
 * @param[in] t Fractional grid offset in [0, 1).
 * @param[out] w float[4] receiving dw/dt for grid offsets -1, 0, 1, 2.
 */
__host__ __device__ void cupo_interpolate_1d_f32(float  t, float  w[4]);

/** Compute the weight derivatives used for force interpolation along one axis.
 * @param[in] t Fractional grid offset in [0, 1).
 * @param[out] w double[4] receiving dw/dt for grid offsets -1, 0, 1, 2.
 */
__host__ __device__ void cupo_interpolate_1d_f64(double t, double w[4]);

/** Evaluate the spreading stencil's discrete Fourier amplitude for one axis.
 * @param[in] k Integer DFT bin, periodic modulo G; negative and nonnegative bins are valid.
 * @param[in] G Number of grid points along the axis.
 * @return float sqrt(mean(|DFT|^2)) over 16 fractional grid offsets;
 * the DC amplitude is 1.
 */
__host__ __device__ float  cupo_discrete_fourier_kernel_1d_f32(int k, int G);

/** Evaluate the spreading stencil's discrete Fourier amplitude for one axis.
 * @param[in] k Integer DFT bin, periodic modulo G; negative and nonnegative bins are valid.
 * @param[in] G Number of grid points along the axis.
 * @return double sqrt(mean(|DFT|^2)) over 16 fractional grid offsets;
 * the DC amplitude is 1.
 */
__host__ __device__ double cupo_discrete_fourier_kernel_1d_f64(int k, int G);
