// The plugin registration functions and createKernelImpl() follow OpenMM's example
// plugin (github.com/openmm/openmmexampleplugin),
// platforms/cuda/src/CudaExampleKernelFactory.cpp:
// Copyright (c) 2014 Stanford University and the Authors.
// Author: Peter Eastman.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

#include "CudaKernelFactory.hpp"
#include "nonbonded/CudaNonbondedKernels.hpp"
#include "barostat/CudaBarostatKernel.hpp"
#include "integrator/CudaLangevinMiddleKernel.hpp"
#include "bonded/GroupedBonded.hpp"
#include "openmm/kernels.h"
#include "nvopenmm/Kernels.h"
#include "CudaContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <exception>


// Definitions live in the plugin's namespace, so our NonbondedForce et al.
// win unqualified lookup over OpenMM's same-named classes rather than
// being ambiguous with them.
namespace NvOpenMM {

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        OpenMM::Platform& platform = OpenMM::Platform::getPlatformByName("CUDA");
        CudaKernelFactory* factory = new CudaKernelFactory();
        platform.registerKernelFactory(CalcNonbondedForceKernel::Name(), factory);
        platform.registerKernelFactory(IntegrateLangevinMiddleStepKernel::Name(), factory);
        platform.registerKernelFactory(ApplyMonteCarloBarostatKernel::Name(), factory);
        platform.registerKernelFactory(OpenMM::CalcHarmonicBondForceKernel::Name(), factory);
        platform.registerKernelFactory(OpenMM::CalcHarmonicAngleForceKernel::Name(), factory);
        platform.registerKernelFactory(OpenMM::CalcPeriodicTorsionForceKernel::Name(), factory);
    }
    catch (std::exception& ex) {
        // Ignore — CUDA platform may not be available
    }
}

extern "C" OPENMM_EXPORT void registerNvCudaKernelFactories() {
    try {
        OpenMM::Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        OpenMM::Platform::registerPlatform(new OpenMM::CudaPlatform());
    }
    registerKernelFactories();
}

OpenMM::KernelImpl* CudaKernelFactory::createKernelImpl(std::string name,
                                                       const OpenMM::Platform& platform,
                                                       OpenMM::ContextImpl& context) const {
    if (auto* bonded=createGroupedBondedKernel(name,platform,context)) return bonded;
    OpenMM::CudaContext& cu = *static_cast<OpenMM::CudaPlatform::PlatformData*>(
        context.getPlatformData())->contexts[0];
    if (name == CalcNonbondedForceKernel::Name())
        return new CudaCalcNonbondedForce(name, platform, cu);
    if (name == IntegrateLangevinMiddleStepKernel::Name())
        return new CudaIntegrateLangevinMiddleStepKernel(name, platform, cu);
    if (name == ApplyMonteCarloBarostatKernel::Name())
        return new CudaApplyMonteCarloBarostatKernel(name, platform, cu);
    throw OpenMM::OpenMMException(
        (std::string("Tried to create kernel with unknown name '") + name + "'").c_str());
}

} // namespace NvOpenMM
