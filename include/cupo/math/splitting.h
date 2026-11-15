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

/** Scalar splitting kernels, available on the host and device through LTO-IR.
 * Requires a CUDA-aware compiler for the __host__ __device__ qualifiers.
 * The fixed C2 Prolate models use c=10.75 for P1 (Coulomb) and
 * c=12.764031957789484 for P6 (dispersion); cutoff is the runtime parameter.
 * Kernel values omit interaction factors: multiply P1 by Coulomb*q_i*q_j and
 * P6 by -C6_ij. Self-correction values already include their correction signs.
 * Real-space gradients are d/d(r^2); for energy A*S(r^2), the force on i is
 * -2*A*grad(r^2)*(r_i-r_j). Fourier gradients are d/d(k^2), with k in cycles/length.
 * Fourier kernels retain their tails; c is not a hard reciprocal cutoff.
 * Short-range values and their first two radial derivatives vanish at cutoff
 * mathematically; floating-point evaluation can leave small residuals just inside it.
 */
#include "cupo/package_version.h"

/** Evaluate the P1 short-range kernel.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return float S(r^2) = 1/r - L(r^2); zero for r2 >= cutoff^2.
 */
__host__ __device__ float  cupo_p1_short_range_value_f32(float  r2, float  cutoff);

/** Evaluate the P1 short-range kernel.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return double S(r^2) = 1/r - L(r^2); zero for r2 >= cutoff^2.
 */
__host__ __device__ double cupo_p1_short_range_value_f64(double r2, double cutoff);

/** Differentiate the P1 short-range kernel with respect to r^2.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return float dS/d(r^2); zero for r2 >= cutoff^2.
 */
__host__ __device__ float  cupo_p1_short_range_grad_f32(float  r2, float  cutoff);

/** Differentiate the P1 short-range kernel with respect to r^2.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return double dS/d(r^2); zero for r2 >= cutoff^2.
 */
__host__ __device__ double cupo_p1_short_range_grad_f64(double r2, double cutoff);

/** Evaluate the P1 long-range kernel.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return float L(r^2), with a 1/r tail beyond cutoff.
 */
__host__ __device__ float  cupo_p1_long_range_value_f32(float  r2, float  cutoff);

/** Evaluate the P1 long-range kernel.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return double L(r^2), with a 1/r tail beyond cutoff.
 */
__host__ __device__ double cupo_p1_long_range_value_f64(double r2, double cutoff);

/** Differentiate the P1 long-range kernel with respect to r^2.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return float dL/d(r^2).
 */
__host__ __device__ float  cupo_p1_long_range_grad_f32(float  r2, float  cutoff);

/** Differentiate the P1 long-range kernel with respect to r^2.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return double dL/d(r^2).
 */
__host__ __device__ double cupo_p1_long_range_grad_f64(double r2, double cutoff);

/** Evaluate the P1 long-range Fourier kernel.
 * @param[in] k2 Squared frequency, with k in cycles per unit length.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return float K(k^2); zero at k2=0 by convention.
 */
__host__ __device__ float  cupo_p1_fourier_kernel_value_f32(float  k2, float  cutoff);

/** Evaluate the P1 long-range Fourier kernel.
 * @param[in] k2 Squared frequency, with k in cycles per unit length.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return double K(k^2); zero at k2=0 by convention.
 */
__host__ __device__ double cupo_p1_fourier_kernel_value_f64(double k2, double cutoff);

/** Differentiate the P1 long-range Fourier kernel with respect to k^2.
 * @param[in] k2 Squared frequency, with k in cycles per unit length.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return float dK/d(k^2); zero at k2=0 by convention.
 */
__host__ __device__ float  cupo_p1_fourier_kernel_grad_f32(float  k2, float  cutoff);

/** Differentiate the P1 long-range Fourier kernel with respect to k^2.
 * @param[in] k2 Squared frequency, with k in cycles per unit length.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return double dK/d(k^2); zero at k2=0 by convention.
 */
__host__ __device__ double cupo_p1_fourier_kernel_grad_f64(double k2, double cutoff);

/** Evaluate the signed P1 self-energy correction factor.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return float -0.5*L(0) (negative).
 */
__host__ __device__ float  cupo_p1_self_value_f32(float  cutoff);

/** Evaluate the signed P1 self-energy correction factor.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return double -0.5*L(0) (negative).
 */
__host__ __device__ double cupo_p1_self_value_f64(double cutoff);

/** Evaluate the finite P1 Fourier part after removing the Coulomb pole at k=0.
 * Used for the net-charge background correction; this is not K(0).
 * @param[in] cutoff Split cutoff distance in length units.
 * @return float finite DC coefficient, without charge, coupling, or volume factors.
 */
__host__ __device__ float  cupo_p1_fourier_kernel_dc_f32(float  cutoff);

/** Evaluate the finite P1 Fourier part after removing the Coulomb pole at k=0.
 * Used for the net-charge background correction; this is not K(0).
 * @param[in] cutoff Split cutoff distance in length units.
 * @return double finite DC coefficient, without charge, coupling, or volume factors.
 */
__host__ __device__ double cupo_p1_fourier_kernel_dc_f64(double cutoff);

/** Evaluate the P6 short-range kernel.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return float S(r^2) = 1/r^6 - L(r^2); zero for r2 >= cutoff^2.
 */
__host__ __device__ float  cupo_p6_short_range_value_f32(float  r2, float  cutoff);

/** Evaluate the P6 short-range kernel.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return double S(r^2) = 1/r^6 - L(r^2); zero for r2 >= cutoff^2.
 */
__host__ __device__ double cupo_p6_short_range_value_f64(double r2, double cutoff);

/** Differentiate the P6 short-range kernel with respect to r^2.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return float dS/d(r^2); zero for r2 >= cutoff^2.
 */
__host__ __device__ float  cupo_p6_short_range_grad_f32(float  r2, float  cutoff);

/** Differentiate the P6 short-range kernel with respect to r^2.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return double dS/d(r^2); zero for r2 >= cutoff^2.
 */
__host__ __device__ double cupo_p6_short_range_grad_f64(double r2, double cutoff);

/** Evaluate the P6 long-range kernel.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return float L(r^2), with a 1/r^6 tail beyond cutoff.
 */
__host__ __device__ float  cupo_p6_long_range_value_f32(float  r2, float  cutoff);

/** Evaluate the P6 long-range kernel.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return double L(r^2), with a 1/r^6 tail beyond cutoff.
 */
__host__ __device__ double cupo_p6_long_range_value_f64(double r2, double cutoff);

/** Differentiate the P6 long-range kernel with respect to r^2.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return float dL/d(r^2).
 */
__host__ __device__ float  cupo_p6_long_range_grad_f32(float  r2, float  cutoff);

/** Differentiate the P6 long-range kernel with respect to r^2.
 * @param[in] r2 Squared distance, in length^2.
 * @param[in] cutoff Split cutoff distance, in the same length units.
 * @return double dL/d(r^2).
 */
__host__ __device__ double cupo_p6_long_range_grad_f64(double r2, double cutoff);

/** Evaluate the P6 long-range Fourier kernel.
 * @param[in] k2 Squared frequency, with k in cycles per unit length.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return float K(k^2); includes the finite, nonzero DC value at k2=0.
 */
__host__ __device__ float  cupo_p6_fourier_kernel_value_f32(float  k2, float  cutoff);

/** Evaluate the P6 long-range Fourier kernel.
 * @param[in] k2 Squared frequency, with k in cycles per unit length.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return double K(k^2); includes the finite, nonzero DC value at k2=0.
 */
__host__ __device__ double cupo_p6_fourier_kernel_value_f64(double k2, double cutoff);

/** Differentiate the P6 long-range Fourier kernel with respect to k^2.
 * @param[in] k2 Squared frequency, with k in cycles per unit length.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return float dK/d(k^2); includes the finite analytic derivative at k2=0.
 */
__host__ __device__ float  cupo_p6_fourier_kernel_grad_f32(float  k2, float  cutoff);

/** Differentiate the P6 long-range Fourier kernel with respect to k^2.
 * @param[in] k2 Squared frequency, with k in cycles per unit length.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return double dK/d(k^2); includes the finite analytic derivative at k2=0.
 */
__host__ __device__ double cupo_p6_fourier_kernel_grad_f64(double k2, double cutoff);

/** Evaluate the signed P6 self-energy correction factor.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return float +0.5*L(0) (positive).
 */
__host__ __device__ float  cupo_p6_self_value_f32(float  cutoff);

/** Evaluate the signed P6 self-energy correction factor.
 * @param[in] cutoff Split cutoff distance in length units.
 * @return double +0.5*L(0) (positive).
 */
__host__ __device__ double cupo_p6_self_value_f64(double cutoff);
