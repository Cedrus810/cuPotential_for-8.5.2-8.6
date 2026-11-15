// The integrator's constructor, setters, initialize() and step loop follow OpenMM
// openmmapi/src/LangevinMiddleIntegrator.cpp:
// Copyright (c) 2008-2021 Stanford University and the Authors.
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

#include "nvopenmm/LangevinMiddleIntegrator.h"
#include "nvopenmm/Kernels.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "nvopenmm/internal/MonteCarloBarostatImpl.h"
#include <string>


// Definitions live in the plugin's namespace, so our NonbondedForce et al.
// win unqualified lookup over OpenMM's same-named classes rather than
// being ambiguous with them.
namespace NvOpenMM {
using std::string;
using std::vector;

LangevinMiddleIntegrator::LangevinMiddleIntegrator(double temperature, double frictionCoeff,
                                                               double stepSize, int comFrequency) {
    setTemperature(temperature);
    setFriction(frictionCoeff);
    setStepSize(stepSize);
    setConstraintTolerance(1e-5);
    setRandomNumberSeed(0);
    setComFrequency(comFrequency);
}

LangevinMiddleIntegrator* LangevinMiddleIntegrator::cast(OpenMM::Integrator& integrator) {
    return dynamic_cast<LangevinMiddleIntegrator*>(&integrator);
}

void LangevinMiddleIntegrator::setTemperature(double temp) {
    if (temp < 0)
        throw OpenMM::OpenMMException("Temperature cannot be negative");
    temperature = temp;
}

void LangevinMiddleIntegrator::setFriction(double coeff) {
    if (coeff < 0)
        throw OpenMM::OpenMMException("Friction cannot be negative");
    friction = coeff;
}

void LangevinMiddleIntegrator::initialize(OpenMM::ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMM::OpenMMException("This Integrator is already bound to a context");
    fusingBarostat = nullptr;
    barostatResolved = false;
    context = &contextRef;
    owner = &contextRef.getOwner();
    kernel = context->getPlatform().createKernel(IntegrateLangevinMiddleStepKernel::Name(), contextRef);
    kernel.getAs<IntegrateLangevinMiddleStepKernel>().initialize(contextRef.getSystem(), *this);
}

void LangevinMiddleIntegrator::cleanup() {
    kernel = OpenMM::Kernel();
}

vector<string> LangevinMiddleIntegrator::getKernelNames() {
    return {IntegrateLangevinMiddleStepKernel::Name()};
}

double LangevinMiddleIntegrator::computeKineticEnergy() {
    return kernel.getAs<IntegrateLangevinMiddleStepKernel>().computeKineticEnergy(*context, *this);
}

bool LangevinMiddleIntegrator::kineticEnergyRequiresForce() const {
    return false;
}

void LangevinMiddleIntegrator::createCheckpoint(std::ostream& stream) const {
    kernel.getAs<IntegrateLangevinMiddleStepKernel>().createCheckpoint(stream);
}

void LangevinMiddleIntegrator::loadCheckpoint(std::istream& stream) {
    kernel.getAs<IntegrateLangevinMiddleStepKernel>().loadCheckpoint(stream);
}

// Our barostat's Metropolis test evaluates the energy at two configurations, and asks
// for forces alongside (see MonteCarloBarostatImpl::leftForcesValid). One of the two is
// always the configuration the context keeps, so on a move step the force buffer is
// already what this integrator would have recomputed -- a third traversal of the same
// neighbour structure, ~66 us of it shared, for a result already sitting in the buffer.
//
// The pointer is resolved once: getForceImpls() is fixed for a Context's lifetime, and
// a dynamic_cast per step is a host cost on the step's critical path.
const MonteCarloBarostatImpl* LangevinMiddleIntegrator::findFusingBarostat(
        OpenMM::ContextImpl& ctx) {
    if (!barostatResolved) {
        barostatResolved = true;
        for (auto* f : ctx.getForceImpls())
            if (auto* b = dynamic_cast<const MonteCarloBarostatImpl*>(f)) {
                fusingBarostat = b;
                break;
            }
    }
    return fusingBarostat;
}

void LangevinMiddleIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMM::OpenMMException("This Integrator is not bound to a context!");
    if (kernel.getAs<IntegrateLangevinMiddleStepKernel>().executeSteps(*context, *this, steps))
        return;
    for (int i = 0; i < steps; ++i) {
        context->updateContextState();
        const MonteCarloBarostatImpl* baro = findFusingBarostat(*context);
        if (baro == NULL || !baro->leftForcesValid())
            context->calcForcesAndEnergy(true, false, getIntegrationForceGroups());
        kernel.getAs<IntegrateLangevinMiddleStepKernel>().execute(*context, *this);
    }
}

long long LangevinMiddleIntegrator::getWholeStepGraphCaptureCount() const {
    return context == NULL ? 0 : kernel.getAs<IntegrateLangevinMiddleStepKernel>().getWholeStepGraphCaptureCount();
}

long long LangevinMiddleIntegrator::getWholeStepGraphReplayCount() const {
    return context == NULL ? 0 : kernel.getAs<IntegrateLangevinMiddleStepKernel>().getWholeStepGraphReplayCount();
}

} // namespace NvOpenMM
