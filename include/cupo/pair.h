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
#include "cupo/splitting_plan.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUPO_PAIR_ABI_VERSION 6

/** Standard LJ with Coulomb and/or dispersion splitting, or a custom cutoff evaluator. */
typedef enum { CUPO_PAIR_STANDARD = 0, CUPO_PAIR_CUSTOM = 1 } cupo_pair_kind;

/** Standard LJ combining rule; custom evaluators implement their own. */
typedef enum {
    CUPO_LJ_LORENTZ_BERTHELOT = 0, /**< Arithmetic sigma, geometric epsilon. */
    CUPO_LJ_GEOMETRIC = 1        /**< Geometric sigma and epsilon. */
} cupo_lj_combining_rule;

/** Raw LTO IR (for example, from nvrtcGetLTOIR) or an nvcc fatbin containing LTO IR. */
typedef enum { CUPO_IR_LTOIR = 0, CUPO_IR_FATBIN = 1 } cupo_ir_format;

/** Input code linked during workspace creation. The host storage may then be freed. */
typedef struct {
    const void* data;       /**< Host module bytes. */
    uint64_t bytes;         /**< Module size in bytes, greater than zero. */
    cupo_ir_format format; /**< Module encoding. */
} cupo_ir_module;

/** Built-in LJ and lattice-summed Coulomb and/or dispersion interactions.
 * Input particles are FP32 {charge, sigma (nm), epsilon (kJ/mol)}. At least one
 * splitting plan is required; KSpace must use the matching plan. Plan values are
 * copied when a cell list or direct-space workspace is created.
 *
 * Without dispersion_plan, LJ is ordinary 12-6, optionally quintic-switched.
 * With dispersion_plan, the attractive 1/r^6 part uses that splitting and direct
 * LJ is shifted to zero at cutoff; switching must be disabled. Reciprocal weights
 * are 2*sqrt(epsilon)*sigma^3 (geometric C6), independent of the direct combining
 * rule. If both plans are present, their cutoff and splitting family must match.
 * If only dispersion_plan is present, particle charges are ignored.
 * Combining and switching apply only to LJ. Reciprocal exclusions use the original
 * particle weights and these plans; physical replacements use the tuples API.
 */
typedef struct {
    cupo_splitting_plan_handle coulomb_plan; /**< Optional P1 plan specifying cutoff, splitting, and coupling. */
    cupo_lj_combining_rule combining_rule;   /**< Direct LJ combining rule. */
    int use_switching;                     /**< 1 enables the LJ switch; 0 disables it. */
    double switch_distance;                /**< Switch start in nm, at least zero and below cutoff when enabled. */
    cupo_splitting_plan_handle dispersion_plan; /**< Optional attractive P6 plan. */
} cupo_standard_pair_desc;

/** Caller-supplied cutoff interaction, with no implicit reciprocal calculation.
 * Modules must define one cupo_custom_pair evaluator following cupo/pair_device.cuh;
 * extra modules may supply helpers. The evaluator must be pure, repeatable, and
 * symmetric under particle exchange, including evaluation of excluded pairs.
 * It returns physical energy and radial derivative with coupling included and
 * implements any cutoff switching or shift itself.
 *
 * Particle records and globals are opaque bytes. Globals are borrowed device
 * memory: keep their address valid for all launches and graphs, and order content
 * updates before execute on the appropriate stream. The evaluator supports the
 * cell list's orthorhombic or reduced triclinic box, within its stencil limits.
 */
typedef struct {
    const cupo_ir_module* modules; /**< Host module array, borrowed during workspace creation. */
    uint32_t module_count;        /**< Number of modules, greater than zero. */
    uint32_t particle_bytes;      /**< Bytes per particle; a multiple of particle_alignment, or zero for no parameters. */
    uint32_t particle_alignment;  /**< Particle alignment in bytes; a power of two from 1 through 256. */
    const void* d_globals;        /**< Borrowed device globals; may be NULL when globals_bytes is zero. */
    uint64_t globals_bytes;       /**< Size of the device globals buffer in bytes. */
} cupo_custom_pair_desc;

/** Pair evaluator and particle layout shared by cell lists and direct space.
 * Zero-initialize, set struct_bytes and abi_version, and fill the selected model.
 * Code or layout changes require a new direct-space workspace. Particle and global
 * values may change in their existing buffers without relinking.
 */
typedef struct cupo_pair_desc {
    uint32_t struct_bytes;    /**< Set to sizeof(cupo_pair_desc). */
    uint32_t abi_version;     /**< Set to CUPO_PAIR_ABI_VERSION. */
    cupo_pair_kind kind;      /**< Selects the active model member. */
    union {
        cupo_standard_pair_desc standard;
        cupo_custom_pair_desc custom;
    } model;
} cupo_pair_desc;

#ifdef __cplusplus
}
#endif
