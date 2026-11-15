// The Monte Carlo volume move, its acceptance test and the move-size adaptation follow
// OpenMM openmmapi/src/MonteCarloBarostatImpl.cpp:
// Copyright (c) 2010-2025 Stanford University and the Authors.
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

#include "nvopenmm/internal/MonteCarloBarostatImpl.h"
#include "nvopenmm/Kernels.h"
#include "openmm/Context.h"
#include "openmm/Integrator.h"
#include "openmm/MonteCarloBarostat.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/SimTKOpenMMRealType.h"   // BOLTZ, AVOGADRO
#include "openmm/reference/SimTKOpenMMUtilities.h"  // the RNG stock barostats draw from
#include <cmath>
#include <cstdio>
#include <sstream>


// Definitions live in the plugin's namespace, so our NonbondedForce et al.
// win unqualified lookup over OpenMM's same-named classes rather than
// being ambiguous with them.
namespace NvOpenMM {
using std::string;
using std::vector;

MonteCarloBarostatImpl::MonteCarloBarostatImpl(const MonteCarloBarostat& owner)
    : owner(owner), step(0), numAttempted(0), numAccepted(0), volumeScale(0.0) {
}

void MonteCarloBarostatImpl::initialize(OpenMM::ContextImpl& context) {
    if (!context.getSystem().usesPeriodicBoundaryConditions())
        throw OpenMM::OpenMMException("MonteCarloBarostat: a barostat cannot be used with a "
                              "non-periodic system");

    // Our Pressure()/Temperature() parameter names are deliberately the same
    // strings OpenMM's barostats use, so a script can swap the class name and
    // keep working. The cost of that choice is that two barostats in one System
    // would silently share -- and fight over -- one pair of parameters, and
    // would each propose their own moves. Catch it here rather than let it
    // produce a quietly wrong ensemble.
    const OpenMM::System& system = context.getSystem();
    int nBarostats = 0;
    for (int i = 0; i < system.getNumForces(); i++) {
        const OpenMM::Force& f = system.getForce(i);
        if (dynamic_cast<const MonteCarloBarostat*>(&f) != NULL ||
            dynamic_cast<const OpenMM::MonteCarloBarostat*>(&f) != NULL)
            nBarostats++;
    }
    if (nBarostats > 1) {
        std::stringstream msg;
        msg << "MonteCarloBarostat: the System contains " << nBarostats
            << " barostats. They share the '" << MonteCarloBarostat::Pressure()
            << "' and '" << MonteCarloBarostat::Temperature()
            << "' parameters and would each propose independent volume moves. "
               "Remove all but one.";
        throw OpenMM::OpenMMException(msg.str());
    }

    kernel = context.getPlatform().createKernel(ApplyMonteCarloBarostatKernel::Name(), context);
    kernel.getAs<ApplyMonteCarloBarostatKernel>().initialize(system, owner);

    // The move size adapts, starting at 1% of the box volume -- same starting
    // point and same adaptation as the stock barostat, so acceptance rates are
    // comparable run for run.
    OpenMM::Vec3 box[3];
    context.getPeriodicBoxVectors(box[0], box[1], box[2]);
    volumeScale = 0.01 * (box[0][0] * box[1][1] * box[2][2]);
    numAttempted = 0;
    numAccepted = 0;
    OpenMM::SimTKOpenMMUtilities::setRandomNumberSeed(owner.getRandomNumberSeed());
}

void MonteCarloBarostatImpl::updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
    forcesValid = false;          // cleared on EVERY step, including the ones that return early
    if (++step < owner.getFrequency() || owner.getFrequency() == 0)
        return;
    step = 0;
    // FUSION. A Metropolis step evaluates the energy at two configurations and the
    // integrator then evaluates forces at whichever one was kept -- three traversals of
    // the same neighbour structure where two suffice. Measured on bace, the direct-space
    // kernel costs 87 us for energy alone, 94 for forces alone and 115 for both, i.e.
    // 66 us of the work is SHARED and today gets paid twice.
    //
    // Asking both energy passes for forces makes the buffer correct for the kept
    // configuration in either branch: on accept the trial pass computed them there, and
    // on reject restoreState puts back the initial pass's (saveState snapshots
    // getLongForceBuffer precisely because a trial evaluation may reorder atoms). The
    // groups line up too -- `groups` below IS the integrator's own force groups.
    const int groups = context.getIntegrator().getIntegrationForceGroups();

    // Energy of the current configuration.
    //
    // This goes straight to ContextImpl rather than out through
    // Context::getState(State::Energy). Both end up calling exactly this
    // method, but getState additionally calls calcKineticEnergy() -- always,
    // whenever Energy is requested -- and a Metropolis test on volume reads
    // only the potential. On apoA1 that discarded kinetic energy is a
    // single-workgroup reduction over all 92,224 atoms and measured 5.6% of
    // total GPU time across a run. ContextImpl::calcForcesAndEnergy also keeps
    // the atom-reorder retry loop, so nothing is lost by calling it directly.
    const double initialEnergy = context.calcForcesAndEnergy(true, true, groups);

    OpenMM::Vec3 box[3];
    context.getPeriodicBoxVectors(box[0], box[1], box[2]);
    const double volume = box[0][0] * box[1][1] * box[2][2];
    const double deltaVolume =
        volumeScale * 2 * (OpenMM::SimTKOpenMMUtilities::getUniformlyDistributedRandomNumber() - 0.5);
    const double newVolume = volume + deltaVolume;
    const double lengthScale = std::pow(newVolume / volume, 1.0 / 3.0);

    // Snapshot before the trial move. This saves more than the scaling itself
    // touches (positions only) because the energy evaluation below may reorder
    // atoms, which permutes velocities and forces too -- so a rejection has to
    // put all of them back, along with the atom order and cell offsets.
    kernel.getAs<ApplyMonteCarloBarostatKernel>().saveState(context);

    context.setPeriodicBoxVectors(box[0] * lengthScale, box[1] * lengthScale, box[2] * lengthScale);
    kernel.getAs<ApplyMonteCarloBarostatKernel>().scaleCoordinates(context, lengthScale);

    // Energy of the trial configuration.
    const double finalEnergy = context.calcForcesAndEnergy(true, true, groups);

    const double pressure = context.getParameter(MonteCarloBarostat::Pressure()) * (AVOGADRO * 1e-25);
    const double kT = BOLTZ * context.getParameter(MonteCarloBarostat::Temperature());
    const double w = finalEnergy - initialEnergy + pressure * deltaVolume
                     - context.getMolecules().size() * kT * std::log(newVolume / volume);

    const bool reject =
        (w > 0 && OpenMM::SimTKOpenMMUtilities::getUniformlyDistributedRandomNumber() > std::exp(-w / kT));
    if (reject) {
        // Reject: put the box and every buffer the trial evaluation may have
        // disturbed back the way they were.
        context.setPeriodicBoxVectors(box[0], box[1], box[2]);
        kernel.getAs<ApplyMonteCarloBarostatKernel>().restoreState(context);
    }
    else {
        // The accepted trial pass computed the forces at the NEW box, so they are
        // already what the integrator wants: `forcesInvalid` stays as the caller set
        // it, and this is the one path that would otherwise have to set it.
        numAccepted++;
    }
    numAttempted++;
    // Either branch now leaves the buffer matching the configuration we kept.
    forcesValid = true;

    // Adapt the move size toward a 25-75% acceptance band, re-evaluated every
    // 10 attempts. Same schedule and same 1.1 factor as the stock barostat.
    if (numAttempted >= 10) {
        if (numAccepted < 0.25 * numAttempted) {
            volumeScale /= 1.1;
            numAttempted = 0;
            numAccepted = 0;
        }
        else if (numAccepted > 0.75 * numAttempted) {
            volumeScale = std::min(volumeScale * 1.1, volume * 0.3);
            numAttempted = 0;
            numAccepted = 0;
        }
    }
}

std::map<string, double> MonteCarloBarostatImpl::getDefaultParameters() {
    return {{MonteCarloBarostat::Pressure(), getOwner().getDefaultPressure()},
            {MonteCarloBarostat::Temperature(), getOwner().getDefaultTemperature()}};
}

vector<string> MonteCarloBarostatImpl::getKernelNames() {
    return {ApplyMonteCarloBarostatKernel::Name()};
}

} // namespace NvOpenMM
