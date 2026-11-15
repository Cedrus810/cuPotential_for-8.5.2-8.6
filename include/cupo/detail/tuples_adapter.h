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
#include "cupo/tuples.h"

/** Used by the OpenMM plugin only; not part of the supported API. */
#ifdef __cplusplus
extern "C" {
#endif
/** Map logical system atom IDs to OpenMM position slots.
 * The initial mapping is identity. Call outside capture with no outstanding work
 * using the handle. Device addresses remain stable for captured executions.
 * @param[in] handle Tuples handle on its creating device.
 * @param[in] natoms Atom count used at creation.
 * @param[in] logical_to_slot Host int[natoms] permutation of [0, natoms), copied
 * before return; logical_to_slot[i] gives the slot for system atom i.
 * @return CUPO_SUCCESS, or an error status. Invalid input leaves the mapping unchanged.
 */
CUPO_API cupo_status cupo_tuples_set_atom_mapping(cupo_tuples_handle handle,
    int natoms, const int* logical_to_slot);
#ifdef __cplusplus
}
#endif
