// The class's interface and data members follow OpenMM
// openmmapi/include/openmm/internal/MonteCarloBarostatImpl.h:
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

#ifndef OPENMM_NV_MONTE_CARLO_BAROSTAT_IMPL_H_
#define OPENMM_NV_MONTE_CARLO_BAROSTAT_IMPL_H_

#include "nvopenmm/MonteCarloBarostat.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>
#include <vector>

namespace NvOpenMM {



/**
 * Internal implementation of MonteCarloBarostat.
 *
 * All the Monte Carlo bookkeeping (step counter, attempt/accept tallies, the
 * adaptive volume scale) lives here on the host; the kernel owns only the
 * device-side save / scale / restore.
 */
class MonteCarloBarostatImpl : public OpenMM::ForceImpl {
public:
    MonteCarloBarostatImpl(const MonteCarloBarostat& owner);

    void initialize(OpenMM::ContextImpl& context);

    const MonteCarloBarostat& getOwner() const {
        return owner;
    }

    /** Propose, evaluate and accept-or-reject one volume move. */
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid);

    /**
     * True when the move this step just finished left the force buffer holding forces
     * for the configuration the context KEPT -- so the integrator can skip its own
     * force evaluation entirely.
     *
     * Both Metropolis energies are evaluated with includeForces, and one of the two is
     * always at the kept configuration: on accept the trial pass was, and on reject
     * restoreState puts back the initial pass's forces (saveState snapshots
     * getLongForceBuffer for exactly this reason). False on every non-move step.
     */
    bool leftForcesValid() const { return forcesValid; }

    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces,
                               bool includeEnergy, int groups) {
        // A barostat is not a force: it moves the box, it does not push atoms.
        return 0.0;
    }

    std::map<std::string, double> getDefaultParameters();
    std::vector<std::string> getKernelNames();

private:
    const MonteCarloBarostat& owner;
    int step, numAttempted, numAccepted;
    bool forcesValid = false;
    double volumeScale;
    OpenMM::Kernel kernel;
};

} // namespace NvOpenMM

#endif // OPENMM_NV_MONTE_CARLO_BAROSTAT_IMPL_H_
