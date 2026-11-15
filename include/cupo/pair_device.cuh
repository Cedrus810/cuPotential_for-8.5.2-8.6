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
#define CUPO_PAIR_DEVICE_ABI_VERSION 3

#define CUPO_PAIR_ENERGY   1u
#define CUPO_PAIR_GRADIENT 2u

/** FP32 pair distance, squared distance, and their reciprocals. */
struct CupoPairGeometry {
    float r, r2, inv_r, inv_r2;
};
/** Pair energy and radial derivative in the caller's physical units. */
struct CupoPairResult {
    float value; /**< Pair energy U(r). */
    float grad; /**< dU/dr; force on i is -grad * (r_i-r_j)/r. */
};

/** Arguments for a custom pair evaluator. */
#define CUPO_PAIR_ARGUMENTS CupoPairGeometry geometry, const void* particle_i, const void* particle_j, const void* globals, unsigned outputs

/** Evaluate a custom pair energy and radial derivative on the device.
 * Supply this C-linkage function in the custom pair's LTO IR. Include any cutoff
 * switch in the returned energy and derivative; the runtime adds none.
 *
 * Excluded pairs may also be evaluated. The function must have no side effects
 * and must not trap or access invalid memory, including at excluded singularities.
 * Repeatable NaN/Inf results are allowed for those excluded singularities.
 *
 * Identical geometry, record bytes, globals, and output masks must produce
 * bitwise-identical requested results at every inlined call site. Do not depend
 * on thread/block IDs or pointer addresses. Control arithmetic contraction and
 * reassociation where needed, using intrinsics such as __fadd_rn, __fmul_rn,
 * and __fmaf_rn. The expression compiler already does this for generated evaluators.
 * Compile external IR with --use_fast_math to match the runtime's flush-to-zero
 * and approximate division/square-root behavior; that flag alone does not ensure
 * bitwise repeatability. Link options cannot recreate frontend arithmetic choices
 * or guarantee strict math, even for strict-math input IR.
 *
 * @param[in] geometry FP32 distance and reciprocal values shared with exclusion evaluation.
 * @param[in] particle_i Immutable particle record with the declared size and alignment.
 * @param[in] particle_j Immutable particle record with the declared size and alignment.
 * Records may be in global, shared, or local storage. Use generic loads, not
 * global-only intrinsics such as __ldg; do not retain these pointers after return.
 * @param[in] globals Borrowed device globals from cupo_custom_pair_desc; do not modify.
 * @param[in] outputs Bit mask of CUPO_PAIR_ENERGY and CUPO_PAIR_GRADIENT;
 * compute only requested fields.
 * @return CupoPairResult containing FP32 energy and/or dU/dr in physical units.
 */
extern "C" __device__ CupoPairResult cupo_custom_pair(CUPO_PAIR_ARGUMENTS);
