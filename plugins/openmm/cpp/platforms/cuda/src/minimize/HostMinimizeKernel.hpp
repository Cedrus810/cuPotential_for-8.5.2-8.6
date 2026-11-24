// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openmm/KernelFactory.h"
#include <string>

namespace NvOpenMM {

// OpenMM's "Minimize" kernel for the CUDA platform (OpenMM 8.5 and later; the
// name does not exist before 8.5, and this returns nullptr there).
//
// Why the plugin provides one: from 8.5 OpenMM minimizes on the GPU
// (CommonMinimizeKernel), adding the constraint springs into the fixed-point force
// buffer. When the spring constant has been raised far enough that those forces
// overflow, it falls back to a Context built from the WHOLE System on the CPU or
// Reference platform -- where this plugin's forces have no kernels, so the
// minimization throws "does not support all required kernels". A System with
// none of the plugin's forces gets OpenMM's own kernel unchanged; one with them
// gets HostMinimizeKernel, the 8.4 algorithm (L-BFGS on the host, springs added
// on the host in double), whose every evaluation goes through this Context.
//
// Returns nullptr for any other kernel name.
OpenMM::KernelImpl* createMinimizeKernel(const std::string& name,
    const OpenMM::Platform& platform, OpenMM::ContextImpl& context);

} // namespace NvOpenMM
