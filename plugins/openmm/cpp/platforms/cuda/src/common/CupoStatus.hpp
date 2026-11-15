// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cupo/status.h"
#include "openmm/OpenMMException.h"
#include <cstdio>
#include <string>

namespace NvOpenMM {

inline void cupoCheck(cupo_status status) {
    if (status != CUPO_SUCCESS)
        throw OpenMM::OpenMMException(std::string("cuPo: ") + cupo_last_error());
}

// Destructors must finish releasing their other resources after a cleanup error.
inline void cupoCleanup(cupo_status status) noexcept {
    if (status != CUPO_SUCCESS)
        std::fprintf(stderr, "[nv-plugin cupo] cleanup error: %s\n", cupo_last_error());
}

}  // namespace NvOpenMM
