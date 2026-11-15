// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openmm/KernelFactory.h"
#include <cupo/tuples.h>
#include <memory>
#include <vector>
namespace OpenMM { class CudaContext; }
namespace NvOpenMM {
// Physical exception pairs share the context's bonded schedule, independently
// of the switch selecting the backend for ordinary OpenMM bonded forces.
class GroupedBondedPairs {
public:
    GroupedBondedPairs(OpenMM::CudaContext& context,
                       std::vector<cupo_tuples_pair> terms);
    ~GroupedBondedPairs();
    void update(const std::vector<cupo_tuples_pair_parameters>& parameters);
    void prepare();
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
// Returns nullptr for names outside this component; standard OpenMM kernels
// are delegated to the installed CUDA factory when this path is ineligible.
OpenMM::KernelImpl* createGroupedBondedKernel(const std::string& name,
    const OpenMM::Platform& platform, OpenMM::ContextImpl& context);
}
