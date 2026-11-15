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

#ifdef __cplusplus
extern "C" {
#endif

/** Immutable Coulomb or attractive-dispersion splitting plan.
 * Direct- and reciprocal-space workspaces copy the plan during creation.
 * The caller may destroy it after creating those workspaces. */
typedef struct cupo_splitting_plan_impl* cupo_splitting_plan_handle;

/* All operations run on the host. Status-returning functions return CUPO_SUCCESS
 * or an error described by cupo_last_error(). Query outputs are unchanged on
 * failure. Output pointers must be non-null and naturally aligned. */

/** Create a Coulomb Ewald plan with a fifth-order B-spline spreading window.
 * Alpha is sqrt(-ln(2 * 5e-4)) / cutoff.
 * @param[in] cutoff Real-space cutoff in nm; positive and finite in FP32.
 * @param[in] coulomb Positive finite Coulomb energy coupling.
 * @param[out] out Host pointer to the new handle; set to NULL on failure.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_splitting_plan_create_ewald(
    double cutoff, double coulomb, cupo_splitting_plan_handle* out);

/** Create a Coulomb Prolate P1 plan with a four-point C2 spreading window.
 * Uses the C2 Prolate model with design bandwidth 10.75, fixed 5e-4 design target,
 * and nonzero Fourier tails.
 * @param[in] cutoff Real-space cutoff in nm; positive and finite in FP32.
 * @param[in] coulomb Positive finite Coulomb energy coupling.
 * @param[out] out Host pointer to the new handle; set to NULL on failure.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_splitting_plan_create_prolate(
    double cutoff, double coulomb, cupo_splitting_plan_handle* out);

/** Create an Ewald plan for attractive dispersion, -b_i*b_j/r^6.
 * Particle weights are b_i = sqrt(C6_i) = 2*sqrt(epsilon_i)*sigma_i^3.
 * Uses unit coupling, the attraction sign, a four-point C2 window, and the
 * Coulomb Ewald plan's alpha and mesh rule. The finite k=0 mode is retained.
 * @param[in] cutoff Real-space cutoff in nm; positive and finite in FP32.
 * @param[out] out Host pointer to the new handle; set to NULL on failure.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_splitting_plan_create_ewald_dispersion(
    double cutoff, cupo_splitting_plan_handle* out);

/** Create a Prolate P6 plan for attractive dispersion, -b_i*b_j/r^6.
 * Particle weights are b_i = sqrt(C6_i). Uses unit coupling, the attraction
 * sign, a four-point C2 window, and the Coulomb Prolate plan's mesh rule.
 * The finite k=0 mode is retained.
 * @param[in] cutoff Real-space cutoff in nm; positive and finite in FP32.
 * @param[out] out Host pointer to the new handle; set to NULL on failure.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_splitting_plan_create_prolate_dispersion(
    double cutoff, cupo_splitting_plan_handle* out);

/** Free a splitting plan.
 * @param[in] plan Handle to destroy; NULL is accepted.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_splitting_plan_destroy(cupo_splitting_plan_handle plan);

/** Compute the self-energy correction, including its sign and coupling.
 * @param[in] plan Splitting plan.
 * @param[in] sum_q2 Finite nonnegative sum of q_i^2 for Coulomb or b_i^2 for dispersion.
 * @param[out] out_energy Host FP64 energy; negative for Coulomb, positive for dispersion.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_splitting_plan_get_self_energy(
    cupo_splitting_plan_handle plan, double sum_q2, double* out_energy);

/** Compute the uniform-background energy correction for a charged cell.
 * Recompute after box-volume changes. Dispersion plans return zero because
 * their k=0 contribution is included in the mesh.
 * @param[in] plan Splitting plan.
 * @param[in] sum_q Finite sum of particle charges; zero gives zero energy.
 * @param[in] volume Positive finite cell volume in nm^3.
 * @param[out] out_energy Host FP64 energy, including the plan's coupling.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_splitting_plan_get_net_charge_energy(
    cupo_splitting_plan_handle plan, double sum_q, double volume, double* out_energy);

/** Query the spreading-window width.
 * @param[in] plan Splitting plan.
 * @param[out] out_order Host integer: number of mesh points per axis in the stencil.
 * @return CUPO_SUCCESS on success, or an error status; see cupo_last_error().
 */
CUPO_API cupo_status cupo_splitting_plan_get_spread_order(
    cupo_splitting_plan_handle plan, int* out_order);

/** Legacy Coulomb Ewald factory; prefer cupo_splitting_plan_create_ewald.
 * @param[in] cutoff Positive finite cutoff in nm, representable in FP32.
 * @param[in] coulomb Positive finite Coulomb coupling.
 * @return Owned plan handle, or NULL with details in cupo_last_error().
 */
CUPO_API cupo_splitting_plan_handle cupo_create_ewald_plan(double cutoff, double coulomb);

/** Legacy Coulomb Prolate factory; prefer cupo_splitting_plan_create_prolate.
 * @param[in] cutoff Positive finite cutoff in nm, representable in FP32.
 * @param[in] coulomb Positive finite Coulomb coupling.
 * @return Owned plan handle, or NULL with details in cupo_last_error().
 */
CUPO_API cupo_splitting_plan_handle cupo_create_prolate_plan(double cutoff, double coulomb);

/** Legacy dispersion Ewald factory; prefer cupo_splitting_plan_create_ewald_dispersion.
 * @param[in] cutoff Positive finite cutoff in nm, representable in FP32.
 * @return Owned plan handle, or NULL with details in cupo_last_error().
 */
CUPO_API cupo_splitting_plan_handle cupo_create_ewald_dispersion_plan(double cutoff);

/** Legacy dispersion Prolate factory; prefer cupo_splitting_plan_create_prolate_dispersion.
 * @param[in] cutoff Positive finite cutoff in nm, representable in FP32.
 * @return Owned plan handle, or NULL with details in cupo_last_error().
 */
CUPO_API cupo_splitting_plan_handle cupo_create_prolate_dispersion_plan(double cutoff);

/** Legacy self-energy query; prefer cupo_splitting_plan_get_self_energy.
 * @param[in] plan Splitting plan.
 * @param[in] sum_q2 Finite nonnegative sum of squared particle weights.
 * @return FP64 energy including sign and coupling, or NaN on error.
 */
CUPO_API double cupo_splitting_plan_self_energy(cupo_splitting_plan_handle plan, double sum_q2);

/** Legacy background-energy query; prefer cupo_splitting_plan_get_net_charge_energy.
 * @param[in] plan Splitting plan.
 * @param[in] sum_q Finite sum of particle charges.
 * @param[in] volume Cell volume in nm^3. Finite nonpositive values return zero.
 * @return FP64 energy including coupling, or NaN on error.
 */
CUPO_API double cupo_splitting_plan_net_charge_energy(
    cupo_splitting_plan_handle plan, double sum_q, double volume);

/** Legacy spreading-width query; prefer cupo_splitting_plan_get_spread_order.
 * @param[in] plan Splitting plan.
 * @return Mesh points per stencil axis, or zero on error.
 */
CUPO_API int cupo_splitting_plan_spread_order(cupo_splitting_plan_handle plan);

#ifdef __cplusplus
}
#endif
