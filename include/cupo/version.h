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

#ifdef __cplusplus
extern "C" {
#endif

/** ABI version required by these headers. */
#define CUPO_ABI_VERSION 32

/** Query the loaded library's ABI version.
 * @return Integer ABI version; compare with CUPO_ABI_VERSION before using the API.
 */
CUPO_API int cupo_abi_version();
/** Query the loaded library's version string.
 * @return Borrowed host NUL-terminated string, valid while the library is loaded.
 */
CUPO_API const char* cupo_version_string();
/** Query the loaded library's feature flags.
 * @return Bitmask: 0x1 direct space, 0x2 reciprocal space, 0x4 bonded interactions,
 * and 0x8 OpenMM-compatible fixed64 force output.
 */
CUPO_API unsigned long long cupo_runtime_capabilities();

#ifdef __cplusplus
}
#endif
