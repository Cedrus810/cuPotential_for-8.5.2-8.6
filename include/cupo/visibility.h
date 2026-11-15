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

// We build with CXX_VISIBILITY_PRESET hidden; mark public API symbols.
#define CUPO_API __attribute__((visibility("default")))
#define CUPO_LOCAL __attribute__((visibility("hidden")))

#define CUPO_RUNTIME_INTERNAL_API CUPO_API
