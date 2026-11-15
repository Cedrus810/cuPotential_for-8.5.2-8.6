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

#include "cupo/status.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Experimental interaction-tuple handle for bonds, angles, torsions, and pairs.
 * Units are nm, radians, and kJ/mol.
 * All parameter values must be finite. Failures are described by cupo_last_error().
 */
typedef struct cupo_tuples_impl* cupo_tuples_handle;
typedef enum cupo_tuples_kind_t {
    CUPO_TUPLES_HARMONIC_BOND = 0,
    CUPO_TUPLES_HARMONIC_ANGLE = 1,
    CUPO_TUPLES_PERIODIC_TORSION = 2,
    CUPO_TUPLES_COULOMB_LJ_PAIR = 3
} cupo_tuples_kind;
/** Precision of coordinate inputs and interaction evaluation. */
typedef enum cupo_tuples_precision_t {
    CUPO_TUPLES_FLOAT32 = 0,
    CUPO_TUPLES_FLOAT64 = 1
} cupo_tuples_precision;

/** Harmonic bond energy: k*(r-length)^2/2. */
typedef struct cupo_tuples_bond_parameters_t {
    double length, k; /**< Nonnegative equilibrium length in nm and k in kJ/(mol*nm^2). */
} cupo_tuples_bond_parameters;
/** Harmonic angle energy: k*(theta-angle)^2/2. */
typedef struct cupo_tuples_angle_parameters_t {
    double angle, k; /**< Equilibrium angle in [0, pi] radians; nonnegative k in kJ/(mol*rad^2). */
} cupo_tuples_angle_parameters;
/** Periodic torsion energy: k*(1+cos(periodicity*phi-phase)). */
typedef struct cupo_tuples_torsion_parameters_t {
    double k, phase; /**< Energy coefficient in kJ/mol and phase in radians. */
    int periodicity; /**< Positive integer, independent of use_pbc. */
} cupo_tuples_torsion_parameters;
/** Pair energy: coulomb/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6).
 * No cutoff, switching, or splitting is applied.
 */
typedef struct cupo_tuples_pair_parameters_t {
    /** Coulomb coefficient including the unit factor, in kJ*nm/mol;
     * nonnegative sigma in nm; nonnegative epsilon in kJ/mol.
     */
    double coulomb, sigma, epsilon;
} cupo_tuples_pair_parameters;

/** Bond between two atoms in caller system order. */
typedef struct cupo_tuples_bond_t {
    int atoms[2]; /**< System atom indices, each in [0, natoms). */
    cupo_tuples_bond_parameters parameters;
    int use_pbc, group; /**< use_pbc is 0 or 1 (minimum-image distances); group is in [0, 31]. */
} cupo_tuples_bond;
/** Angle between three atoms in caller system order. */
typedef struct cupo_tuples_angle_t {
    int atoms[3]; /**< System atom indices, with the central atom second. */
    cupo_tuples_angle_parameters parameters;
    int use_pbc, group; /**< use_pbc is 0 or 1 (minimum-image displacements); group is in [0, 31]. */
} cupo_tuples_angle;
/** Torsion through four atoms in caller system order. */
typedef struct cupo_tuples_torsion_t {
    int atoms[4]; /**< System atom indices, with the rotation axis between the middle two. */
    cupo_tuples_torsion_parameters parameters;
    int use_pbc, group; /**< use_pbc is 0 or 1 (minimum-image displacements); group is in [0, 31]. */
} cupo_tuples_torsion;
/** Unsplit Coulomb/Lennard-Jones interaction between two atoms. */
typedef struct cupo_tuples_pair_t {
    int atoms[2]; /**< System atom indices, each in [0, natoms). */
    cupo_tuples_pair_parameters parameters;
    int use_pbc, group; /**< use_pbc is 0 or 1 (minimum-image distances); group is in [0, 31]. */
} cupo_tuples_pair;

/** Host array of interactions sharing one kind. */
typedef struct cupo_tuples_term_batch_t {
    cupo_tuples_kind kind; /**< Selects the record type in terms. */
    int count; /**< Nonnegative record count. */
    /** Contiguous host array of cupo_tuples_bond, cupo_tuples_angle,
     * cupo_tuples_torsion, or cupo_tuples_pair records, as selected by kind.
     */
    const void* terms;
} cupo_tuples_term_batch;
/** Replacement parameters for one existing interaction batch. */
typedef struct cupo_tuples_parameter_batch_t {
    cupo_tuples_kind kind; /**< Must match the original batch kind. */
    int count; /**< Must match the original batch count. */
    const void* parameters; /**< Contiguous host array of the corresponding *_parameters type. */
} cupo_tuples_parameter_batch;

/** Copy interaction tuples into a handle owned by the current CUDA device.
 * Call outside graph capture. Reversing a term's complete atom list produces
 * bitwise-identical energy and forces. All later operations use this device.
 * @param[in] natoms Nonnegative number of atoms in caller system order.
 * @param[in] nbatches Nonnegative number of interaction batches.
 * @param[in] batches Host array [nbatches], copied before return; NULL if empty.
 * Each term must have distinct atom indices in [0, natoms).
 * @param[in] precision Coordinate and evaluation precision: FLOAT32 or FLOAT64.
 * @param[out] result Host pointer receiving the new handle.
 * @return CUPO_SUCCESS, or an error status with a valid result set to NULL.
 */
CUPO_API cupo_status cupo_tuples_create(int natoms, int nbatches,
    const cupo_tuples_term_batch* batches, cupo_tuples_precision precision,
    cupo_tuples_handle* result);

/** Replace parameters while keeping topology, use_pbc, and groups.
 * Call outside capture with no outstanding work using the handle. Device
 * addresses stay stable, so captured executions use the updates on replay.
 * @param[in] handle Handle to update on its creating device.
 * @param[in] nbatches Original number of batches.
 * @param[in] batches Host parameter batches in original order, with matching
 * kinds and counts; copied before return.
 * @return CUPO_SUCCESS, or an error status. Invalid input leaves parameters unchanged.
 */
CUPO_API cupo_status cupo_tuples_update_parameters(cupo_tuples_handle handle,
    int nbatches, const cupo_tuples_parameter_batch* batches);

/** Free a tuples handle on its creating device.
 * Call outside capture with no outstanding work using the handle.
 * @param[in] handle Handle to destroy, or NULL.
 * @return CUPO_SUCCESS, including for NULL, or an error status.
 */
CUPO_API cupo_status cupo_tuples_destroy(cupo_tuples_handle handle);

/** Compatibility alias for cupo_tuples_destroy, with the same requirements.
 * @param[in] handle Handle to destroy, or NULL.
 * @return CUPO_SUCCESS, including for NULL, or an error status.
 */
CUPO_API cupo_status cupo_tuples_release(cupo_tuples_handle handle);

/** Add the selected interactions' forces and energy to caller-owned device buffers.
 * Enqueues work asynchronously and supports CUDA graph capture. Concurrent
 * streams may share a handle when they use distinct output buffers. Buffers must
 * not overlap; coordinates must be finite and forces must fit the fixed-point range.
 * Active bonds at zero length, collinear/coincident angle or torsion bonds, and
 * pairs at r=0 are outside the differentiable domain. Inactive terms contribute
 * zero even there: k=0, or for pairs, coulomb=0 and either sigma=0 or epsilon=0.
 * @param[in] handle Handle on its creating device.
 * @param[in] positions Device array [natoms] of float4 or double4, matching precision;
 * aligned to 16 bytes. The first three components are coordinates in nm; w is ignored.
 * @param[in] box Device double[6] in (ax, by, cz, bx, cx, cy) order, in nm.
 * Must be a reduced, lower-triangular OpenMM box with positive diagonal;
 * required if any term uses PBC, otherwise ignored.
 * @param[in,out] forces Device int64_t[3*stride], with x/y/z planes scaled by 2^32;
 * adds force (-gradient) in kJ/(mol*nm). NULL skips force calculation.
 * @param[in] stride Number of entries per force plane; at least natoms when forces is set.
 * @param[in,out] energy Device double receiving an energy increment in kJ/mol;
 * NULL skips energy calculation. Both outputs NULL is a no-op.
 * @param[in] groups Bit mask selecting force groups 0 through 31.
 * @param[in] stream Borrowed cudaStream_t cast to void*.
 * @return CUPO_SUCCESS, or an error status; outputs are never cleared.
 */
CUPO_API cupo_status cupo_tuples_execute(cupo_tuples_handle handle,
    const void* positions, const double* box, int64_t* forces, int stride,
    double* energy, uint32_t groups, void* stream);

#ifdef __cplusplus
}
#endif
