// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NV_CUDA_GRAPH_ENV_H_
#define NV_CUDA_GRAPH_ENV_H_

#include <cstdlib>

namespace NvOpenMM {



/**
 * Whether per-step work is captured into a CUDA graph. ONE switch for the whole
 * plugin.
 *
 * The nonbonded step and the integrator step are captured independently, by classes
 * that share no state, but nobody wants one captured and the other not: the two
 * knobs only ever moved together, and a single name is what makes
 * CUPO_PLUGIN_CUDA_GRAPH=0 a complete answer to "turn the graphs off" rather than
 * half of one.
 *
 * A function rather than two matching string literals so the name and the parse rule
 * have one definition -- graphs on unless the value begins with '0'.
 */
inline bool nvCudaGraphEnabled() {
    const char* v = std::getenv("CUPO_PLUGIN_CUDA_GRAPH");
    return v == nullptr || v[0] != '0';
}

} // namespace NvOpenMM

#endif // NV_CUDA_GRAPH_ENV_H_
