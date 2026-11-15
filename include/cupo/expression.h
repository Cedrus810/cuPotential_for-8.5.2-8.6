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
#include "cupo/pair.h"
#include "cupo/status.h"
#include "cupo/visibility.h"
#include <driver_types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUPO_EXPRESSION_DESC_VERSION 5
/** Compiled expression. Failures in this API are described by cupo_last_error(). */
typedef struct cupo_expression_program* cupo_expression_program_handle;
/** Device buffers and preparation kernels for a compiled expression. */
typedef struct cupo_expression_instance* cupo_expression_instance_handle;

/** A long-range contribution to subtract from the full pair energy.
 * P1 subtracts coupling*wi*wj*LR1(r). P6 adds wi*wj*(LR6(r)-LR6(cutoff)).
 * For geometric C6, use weights such as 2*sqrt(epsilon)*sigma^3; for shifted
 * LJPME, supply U_LJ(r)-U_LJ(cutoff) in the energy expression.
 * The caller supplies matching reciprocal, self, background, and exception
 * calculations. No mixing rule, energy shift, or tail correction is inferred.
 */
typedef struct {
    /** Ewald or Prolate P1/P6 plan, borrowed during compilation; cutoffs must match. */
    cupo_splitting_plan_handle plan;
    /** Copied scalar expression using unsuffixed particle names and globals.
     * Distance r is forbidden. Inputs must give finite per-particle weights.
     * Weights are refreshed with their inputs, even when energy does not use them;
     * they introduce no energy variables and expose no reciprocal-space buffer.
     */
    const char* weight_expression;
} cupo_expression_lattice_direct_term;

/** A symmetric radial pair-energy expression, evaluated within cutoff.
 * Use full unsplit interactions; lattice_direct_terms supply their subtractions.
 * Particle variables have suffixes 1/2, globals use their declared names, and r
 * is distance. Names cannot be shared between particle parameters and globals.
 * Scalar arithmetic, semicolon definitions, and supported Lepton functions are
 * accepted. Input and prepared values are FP32, using the Custom fast-math policy.
 * Use consistent physical units across expressions, positions, and parameters.
 */
typedef struct {
    uint32_t struct_bytes; /**< Set to sizeof(cupo_expression_desc). */
    uint32_t abi_version; /**< Set to CUPO_EXPRESSION_DESC_VERSION. */
    const char* energy; /**< Host, null-terminated expression; copied by compile. */
    const char* const* particle_names; /**< Host array of input-column names; copied. */
    uint32_t particle_parameter_count; /**< Number of input columns per particle. */
    const char* const* global_names; /**< Host array of global names; copied. */
    uint32_t global_count; /**< Number of global values. */
    int compute_capability; /**< Target CUDA architecture, e.g. 89 or 120. */
    double cutoff; /**< Positive distance; must match the consuming cell list and DSpace. */
    /** 0 disables switching; 1 applies a quintic switch on [switch_distance, cutoff].
     * Without lattice terms, the whole energy is switched. With P1 terms, only
     * the residual after extracting additive coupling*wi*wj/r terms is switched;
     * each term must be recognizable as a product/quotient of weights and r.
     * P6 terms do not support automatic switching. For other decompositions,
     * write the switch in energy. Lattice contributions are never switched.
     */
    int use_switching;
    double switch_distance; /**< Switch start distance; used when use_switching is 1. */
    /** Host array of independent, additive lattice terms; repeated terms count
     * repeatedly. Ignored when the count is zero. All terms share cutoff.
     */
    const cupo_expression_lattice_direct_term* lattice_direct_terms;
    uint32_t lattice_direct_term_count; /**< Number of terms; at most INT_MAX/sizeof(float). */
} cupo_expression_desc;

/** Compile a pair expression without creating a CUDA context or launching GPU work.
 * @param[in] desc Host expression descriptor; source text and names are copied.
 * @param[out] out Host pointer receiving the immutable program handle.
 * @return CUPO_SUCCESS, or an error status with a valid out set to NULL.
 */
CUPO_API cupo_status cupo_expression_compile(
    const cupo_expression_desc* desc, cupo_expression_program_handle* out);

/** Borrow the compiled program's serialized bytes until program destruction.
 * Packages are versioned implementation data, not a portable cross-toolkit format.
 * @param[in] program Compiled or loaded program.
 * @param[out] out_data Host pointer receiving the immutable host byte buffer.
 * @param[out] out_bytes Host uint64_t receiving its size in bytes.
 * @return CUPO_SUCCESS, or an error status. If both output pointers are valid,
 * failure sets out_data to NULL and out_bytes to zero.
 */
CUPO_API cupo_status cupo_expression_get_binary(
    cupo_expression_program_handle program, const void** out_data, uint64_t* out_bytes);

/** Copy and validate a serialized expression program.
 * @param[in] data Host byte buffer returned by cupo_expression_get_binary.
 * @param[in] bytes Buffer size in bytes.
 * @param[out] out Host pointer receiving the loaded program handle.
 * @return CUPO_SUCCESS, or an error status with a valid out set to NULL.
 */
CUPO_API cupo_status cupo_expression_load(
    const void* data, uint64_t bytes, cupo_expression_program_handle* out);

/** Destroy a program handle; existing instances retain their program data.
 * @param[in] program Program to destroy, or NULL.
 * @return CUPO_SUCCESS, including when program is NULL.
 */
CUPO_API cupo_status cupo_expression_program_destroy(cupo_expression_program_handle program);

/** Borrowed inputs for the cell-list and DSpace APIs.
 * Pass prepared records through unchanged. The descriptor's referenced storage,
 * device addresses, and stride stay valid until instance destruction.
 * Complete the first prepare before executing consumers.
 */
typedef struct {
    cupo_pair_desc pair; /**< Custom pair descriptor, including IR and device globals. */
    const void* d_particles; /**< Device array of opaque prepared particle records. */
    uint64_t particle_stride_bytes; /**< Byte distance between prepared particle records. */
} cupo_expression_binding;

/** Allocate expression buffers and preparation kernels in the current CUDA context.
 * The context must match the program's target architecture. Call outside stream
 * capture. The instance retains the program, so its handle may be destroyed.
 * No cell list or DSpace workspace is created.
 * @param[in] program Compiled or loaded program.
 * @param[in] natoms Nonnegative atom count; original and prepared scalar arrays
 * must fit signed 32-bit indexing.
 * @param[out] out Host pointer receiving the instance handle.
 * @param[out] out_binding Host binding receiving stable, initially unprepared buffers.
 * @return CUPO_SUCCESS, or an error status. If both output pointers are valid,
 * failure sets out to NULL and clears out_binding.
 */
CUPO_API cupo_status cupo_expression_instantiate(
    cupo_expression_program_handle program, int natoms,
    cupo_expression_instance_handle* out, cupo_expression_binding* out_binding);

/** Optional parameter updates, in the cell list's input atom order.
 * After changing that order, resupply reordered particles and remap exclusions.
 * The first prepare requires every nonempty host input array. Later NULL arrays
 * preserve current values. Host inputs must be finite and may be reused on return.
 */
typedef struct {
    /** Host FP32 rows [natoms][particle_parameter_count], in declared column order. */
    const float* particles;
    const float* globals; /**< Host FP32 array [global_count], in declared order. */
    /** Optional device FP32 column applied after host updates. Values must be
     * finite; they are not validated by readback. Keep it alive until stream work
     * completes. A later full host update replaces the entire original record.
     */
    const float* d_column;
    /** Byte distance between device column entries; a multiple of sizeof(float)
     * and at least sizeof(float). Must be zero when d_column is NULL.
     */
    uint64_t column_stride_bytes;
    uint32_t column_index; /**< Input column to replace; zero when d_column is NULL. */
} cupo_expression_update;

/** Update expression inputs and refresh dependent prepared values.
 * Call outside capture on the owning context. Serialize calls and order earlier
 * readers before this call and later readers after its stream work completes.
 * Device buffers are not reallocated. After initialization, unchanged host
 * inputs and no device column enqueue no work.
 * @param[in] instance Instance to update.
 * @param[in] update Host update descriptor, or NULL to preserve initialized inputs.
 * @param[in] stream CUDA stream for uploads and preparation kernels.
 * @return CUPO_SUCCESS, or an error status. After a CUDA preparation failure,
 * resupply all original inputs before retrying.
 */
CUPO_API cupo_status cupo_expression_prepare(
    cupo_expression_instance_handle instance, const cupo_expression_update* update,
    cudaStream_t stream);

/** Free an instance's device buffers and kernels on its owning CUDA context.
 * Call outside capture, after destroying dependent graphs and completing
 * dependent cell-list and DSpace work. Borrowed bindings become invalid.
 * @param[in] instance Instance to destroy, or NULL.
 * @return CUPO_SUCCESS, including for NULL, or an error status.
 */
CUPO_API cupo_status cupo_expression_instance_destroy(cupo_expression_instance_handle instance);

#ifdef __cplusplus
}
#endif
