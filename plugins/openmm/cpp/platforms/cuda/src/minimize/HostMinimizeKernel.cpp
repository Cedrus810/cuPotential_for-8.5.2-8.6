// HostMinimizeKernel follows OpenMM 8.4.0's openmmapi/src/LocalEnergyMinimizer.cpp
// (the last release that minimized on the host):
// Portions copyright (c) 2010-2020 Stanford University and the Authors.
// Authors: Peter Eastman.
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
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

#include "HostMinimizeKernel.hpp"

#ifndef NV_OPENMM_GE_85

namespace NvOpenMM {
// OpenMM before 8.5 has no Minimize kernel: LocalEnergyMinimizer runs on the host
// itself, so there is nothing to provide.
OpenMM::KernelImpl* createMinimizeKernel(const std::string&, const OpenMM::Platform&,
                                         OpenMM::ContextImpl&) {
    return nullptr;
}
} // namespace NvOpenMM

#else

#include "lbfgs.h"
#include "nvopenmm/MonteCarloBarostat.h"
#include "nvopenmm/NonbondedForce.h"
#include "openmm/Context.h"
#include "openmm/LocalEnergyMinimizer.h"
#include "openmm/OpenMMException.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/cuda/CudaKernelFactory.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/kernels.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

namespace NvOpenMM {
namespace {

// Same constants as OpenMM's CommonMinimizeKernel (and 8.4's LocalEnergyMinimizer).
constexpr double kMinConstraintTol = 1e-4;
constexpr double kRestraintScale = 100;
constexpr double kRestraintScaleUp = 10;
constexpr double kPrevMaxErrorInit = 1e10;
constexpr double kConstraintTolScale = 100;
// CUDA accumulates forces in 64-bit fixed point with 2^32 per unit, so a force
// component past ~2.1e9 kJ/mol/nm wraps. 8.4 checked against this same 2e9.
constexpr double kLargeForce = 2e9;

struct MinimizerData {
    OpenMM::Context& context;
    const OpenMM::System& system;
    double k;
    double gradTolerance;
    OpenMM::MinimizationReporter* reporter;
};

double computeForcesAndEnergy(MinimizerData& data, const lbfgsfloatval_t* x, lbfgsfloatval_t* g) {
    const int numParticles = data.system.getNumParticles();
    std::vector<OpenMM::Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++)
        positions[i] = OpenMM::Vec3(x[3*i], x[3*i+1], x[3*i+2]);
    data.context.setPositions(positions);
    data.context.computeVirtualSites();
    OpenMM::State state = data.context.getState(OpenMM::State::Forces | OpenMM::State::Energy, false,
                                                data.context.getIntegrator().getIntegrationForceGroups());
    const std::vector<OpenMM::Vec3>& forces = state.getForces();
    for (int i = 0; i < numParticles; i++) {
        for (int d = 0; d < 3; d++) {
            const double f = forces[i][d];
            if (!(std::fabs(f) < kLargeForce)) {
                // OpenMM's own answer to this is a CPU/Reference Context of the same
                // System, which cannot be built for this plugin's forces.
                std::ostringstream msg;
                msg << "NvOpenMM energy minimization: particle " << i << " has a force component of " << f
                    << " kJ/mol/nm, beyond what CUDA's fixed-point force accumulation represents. "
                       "OpenMM would recompute it on the CPU, but this System contains NvOpenMM forces, "
                       "which have no CPU implementation. Start from a less strained structure "
                       "(e.g. remove overlapping atoms).";
                throw OpenMM::OpenMMException(msg.str());
            }
        }
        if (data.system.getParticleMass(i) == 0) {
            g[3*i] = g[3*i+1] = g[3*i+2] = 0.0;
        }
        else {
            g[3*i] = -forces[i][0];
            g[3*i+1] = -forces[i][1];
            g[3*i+2] = -forces[i][2];
        }
    }
    return state.getPotentialEnergy();
}

lbfgsfloatval_t evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int, const lbfgsfloatval_t) {
    MinimizerData& data = *reinterpret_cast<MinimizerData*>(instance);
    const OpenMM::System& system = data.system;
    double energy = computeForcesAndEnergy(data, x, g);

    // Harmonic springs for the constraints, on the host in double: these are what
    // overflow the GPU's fixed-point buffer once k has been raised far enough.
    const double k = data.k;
    for (int i = 0; i < system.getNumConstraints(); i++) {
        int p1, p2;
        double distance;
        system.getConstraintParameters(i, p1, p2, distance);
        OpenMM::Vec3 delta(x[3*p2]-x[3*p1], x[3*p2+1]-x[3*p1+1], x[3*p2+2]-x[3*p1+2]);
        const double r = std::sqrt(delta.dot(delta));
        delta *= 1/r;
        const double dr = r-distance;
        const double kdr = k*dr;
        energy += 0.5*kdr*dr;
        if (system.getParticleMass(p1) != 0) {
            g[3*p1] -= kdr*delta[0];
            g[3*p1+1] -= kdr*delta[1];
            g[3*p1+2] -= kdr*delta[2];
        }
        if (system.getParticleMass(p2) != 0) {
            g[3*p2] += kdr*delta[0];
            g[3*p2+1] += kdr*delta[1];
            g[3*p2+2] += kdr*delta[2];
        }
    }
    return energy;
}

double maxConstraintError(const OpenMM::System& system, const lbfgsfloatval_t* x, double* restraintEnergy, double k) {
    double maxError = 0.0, restraint = 0.0;
    for (int i = 0; i < system.getNumConstraints(); i++) {
        int p1, p2;
        double distance;
        system.getConstraintParameters(i, p1, p2, distance);
        OpenMM::Vec3 delta(x[3*p2]-x[3*p1], x[3*p2+1]-x[3*p1+1], x[3*p2+2]-x[3*p1+2]);
        const double dr = std::sqrt(delta.dot(delta))-distance;
        restraint += 0.5*k*dr*dr;
        maxError = std::max(maxError, std::fabs(dr)/distance);
    }
    if (restraintEnergy != nullptr)
        *restraintEnergy = restraint;
    return maxError;
}

// Called by lbfgs after every iteration. Stops on OpenMM 8.5+'s convergence
// criterion (|grad| <= tolerance*sqrt(numParticles), as CommonMinimizeKernel
// applies it, i.e. the documented RMS force reaching `tolerance`) or when the
// user's reporter asks to.
int progress(void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
             const lbfgsfloatval_t, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t, int n, int iteration, int) {
    MinimizerData& data = *reinterpret_cast<MinimizerData*>(instance);
    if (data.reporter != nullptr) {
        std::vector<double> xout(x, x+n), gradout(g, g+n);
        double restraintEnergy = 0.0;
        const double maxError = maxConstraintError(data.system, x, &restraintEnergy, data.k);
        std::map<std::string, double> args;
        args["restraint energy"] = restraintEnergy;
        args["system energy"] = fx-restraintEnergy;
        args["restraint strength"] = data.k;
        args["max constraint error"] = maxError;
        if (data.reporter->report(iteration-1, xout, gradout, args))
            return 1;
    }
    return gnorm <= data.gradTolerance ? 1 : 0;
}

struct LbfgsArray {
    explicit LbfgsArray(int n) : p(lbfgs_malloc(n)) {
        if (p == nullptr)
            throw OpenMM::OpenMMException("NvOpenMM energy minimization: failed to allocate memory");
    }
    ~LbfgsArray() { lbfgs_free(p); }
    LbfgsArray(const LbfgsArray&) = delete;
    LbfgsArray& operator=(const LbfgsArray&) = delete;
    lbfgsfloatval_t* p;
};

class HostMinimizeKernel : public OpenMM::MinimizeKernel {
public:
    HostMinimizeKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::MinimizeKernel(name, platform) {}

    void initialize(const OpenMM::System&) override {}

    void execute(OpenMM::ContextImpl& impl, double tolerance, int maxIterations,
                 OpenMM::MinimizationReporter* reporter) override {
        OpenMM::Context& context = impl.getOwner();
        const OpenMM::System& system = context.getSystem();
        const int numParticles = system.getNumParticles();
        const int n = 3*numParticles;
        const double constraintTol = context.getIntegrator().getConstraintTolerance();
        const double workingConstraintTol = std::max(kMinConstraintTol, constraintTol);

        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        param.max_iterations = maxIterations;
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
        // Convergence is decided in progress() by OpenMM 8.5+'s criterion, not by
        // lbfgs's own |g|/max(1,|x|) test, which epsilon = 0 switches off.
        param.epsilon = 0;

        // Make sure the initial configuration satisfies all constraints.
        context.applyConstraints(workingConstraintTol);
        const std::vector<OpenMM::Vec3> initialPos = context.getState(OpenMM::State::Positions).getPositions();
        LbfgsArray x(n), g(n);
        auto resetX = [&]() {
            for (int i = 0; i < numParticles; i++)
                for (int d = 0; d < 3; d++)
                    x.p[3*i+d] = initialPos[i][d];
        };
        resetX();

        MinimizerData data{context, system, kRestraintScale/workingConstraintTol,
                           tolerance*std::sqrt((double) numParticles), reporter};
        double prevMaxError1 = kPrevMaxErrorInit, prevMaxError2 = kPrevMaxErrorInit;
        while (true) {
            // Already converged at this spring strength: skip the round, as
            // CommonMinimizeKernel does when the starting gradient is below tolerance.
            evaluate(&data, x.p, g.p, n, 0);
            double gnorm = 0.0;
            for (int i = 0; i < n; i++)
                gnorm += g.p[i]*g.p[i];
            if (std::sqrt(gnorm) > data.gradTolerance) {
                lbfgsfloatval_t fx;
                lbfgs(n, x.p, &fx, evaluate, progress, &data, &param);
            }
            // The Context holds the last point lbfgs evaluated; make it the result.
            std::vector<OpenMM::Vec3> positions(numParticles);
            for (int i = 0; i < numParticles; i++)
                positions[i] = OpenMM::Vec3(x.p[3*i], x.p[3*i+1], x.p[3*i+2]);
            context.setPositions(positions);

            if (system.getNumConstraints() == 0)
                break;
            const double maxError = maxConstraintError(system, x.p, nullptr, data.k);
            if (maxError <= workingConstraintTol)
                break; // All constraints are satisfied.
            context.setPositions(initialPos);
            if (maxError >= prevMaxError2)
                break; // Tightening the springs further is not helping, so give up.
            prevMaxError2 = prevMaxError1;
            prevMaxError1 = maxError;
            data.k *= kRestraintScaleUp;
            if (maxError > kConstraintTolScale*workingConstraintTol) {
                // Far enough from a valid state that getting back may be hard:
                // restart from the original positions.
                resetX();
            }
        }

        // A final projection so the constraints hold to the full requested precision.
        if (constraintTol < workingConstraintTol)
            context.applyConstraints(constraintTol);
    }
};

bool hasNvForce(const OpenMM::System& system) {
    for (int i = 0; i < system.getNumForces(); i++) {
        const OpenMM::Force& force = system.getForce(i);
        if (dynamic_cast<const NvOpenMM::NonbondedForce*>(&force) != nullptr ||
            dynamic_cast<const NvOpenMM::MonteCarloBarostat*>(&force) != nullptr)
            return true;
    }
    return false;
}

} // namespace

OpenMM::KernelImpl* createMinimizeKernel(const std::string& name, const OpenMM::Platform& platform,
                                         OpenMM::ContextImpl& context) {
    if (name != OpenMM::MinimizeKernel::Name())
        return nullptr;
    if (!hasNvForce(context.getSystem())) {
        OpenMM::CudaKernelFactory stock;
        return stock.createKernelImpl(name, platform, context);
    }
    return new HostMinimizeKernel(name, platform);
}

} // namespace NvOpenMM

#endif // NV_OPENMM_GE_85
