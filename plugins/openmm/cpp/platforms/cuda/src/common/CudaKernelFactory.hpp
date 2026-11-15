// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENMM_NVCUDAKERNELFACTORY_H_
#define OPENMM_NVCUDAKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace NvOpenMM {



/**
 * This KernelFactory creates kernels for the CUDA implementation of NonbondedForce.
 */
class CudaKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform,
                                 OpenMM::ContextImpl& context) const;
};

} // namespace NvOpenMM

#endif // OPENMM_NVCUDAKERNELFACTORY_H_
