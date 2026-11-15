// The class's parameter names, accessors and data members follow OpenMM
// openmmapi/include/openmm/MonteCarloBarostat.h:
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

#ifndef OPENMM_NV_MONTE_CARLO_BAROSTAT_H_
#define OPENMM_NV_MONTE_CARLO_BAROSTAT_H_

#include "openmm/Force.h"
#include <string>

namespace NvOpenMM {



/**
 * An isotropic Monte Carlo barostat, GPU-accelerated by the Nvidia/Cupo plugin.
 *
 * It is a drop-in replacement for OpenMM's MonteCarloBarostat: same move, same
 * acceptance criterion, same adaptive volume-scale tuning, and the same two
 * Context parameters, so an existing script only changes the class name. It is
 * independent of the integrator -- pair it with the stock LangevinMiddle
 * integrator or with LangevinMiddleIntegrator, either works.
 *
 * What differs is entirely in how the move is executed:
 *
 *  - The trial energies are taken straight from ContextImpl::calcForcesAndEnergy
 *    instead of through Context::getState(State::Energy). getState computes the
 *    KINETIC energy unconditionally whenever Energy is requested, and a barostat
 *    reads only the potential -- so the stock path pays for a number it throws
 *    away, twice per move. On apoA1 that kernel is a single-workgroup reduction
 *    over every atom and measured 5.6% of all GPU time.
 *
 *  - Coordinate scaling is atom-parallel. The stock kernel runs one thread per
 *    molecule and each thread walks its own molecule twice, so a system with a
 *    few large molecules among many small ones serializes on the large ones.
 *    apoA1 is exactly that shape (21,620 molecules, median 3 atoms, but two of
 *    3,205) and its scaling kernel measured 725 us per call -- roughly 4x the
 *    whole direct-space force kernel. See CudaBarostatKernel.cu.
 *
 * Like every OpenMM barostat this subclasses Force, which it is not: it applies
 * no force and contributes no energy. ForceImpl::updateContextState is simply
 * the only per-step hook OpenMM exposes outside of Integrator, so anything that
 * has to run once per step has to declare itself a Force to get at it.
 *
 * This class assumes the simulation is also being run at constant temperature
 * and needs to be told that temperature, because it sets the acceptance
 * probability. It does not itself regulate temperature -- use a Langevin
 * integrator or a thermostat alongside it.
 */
class MonteCarloBarostat : public OpenMM::Force {
public:
    /**
     * The name of the parameter holding the current pressure (in bar).
     *
     * Deliberately the same string OpenMM's MonteCarloBarostat uses, so
     * context.setParameter(...) calls in existing scripts keep working when the
     * barostat is swapped. Having both barostats in one System is therefore a
     * configuration error, and is rejected at Context creation.
     */
    static const std::string& Pressure() {
        static const std::string key = "MonteCarloPressure";
        return key;
    }
    /**
     * The name of the parameter holding the temperature the system is held at
     * (in Kelvin). Same string as OpenMM's MonteCarloBarostat -- see Pressure().
     */
    static const std::string& Temperature() {
        static const std::string key = "MonteCarloTemperature";
        return key;
    }
    /**
     * Create an MonteCarloBarostat.
     *
     * @param defaultPressure     default pressure acting on the system (bar)
     * @param defaultTemperature  default temperature the system is held at (Kelvin)
     * @param frequency           attempt a volume move every this many steps;
     *                            0 disables the barostat
     */
    MonteCarloBarostat(double defaultPressure, double defaultTemperature, int frequency = 25);

    /** Borrow the plugin type from an OpenMM Force; nullptr for other force types. */
    static MonteCarloBarostat* cast(OpenMM::Force& force);

    /** Default pressure acting on the system (bar). */
    double getDefaultPressure() const {
        return defaultPressure;
    }
    /**
     * Set the default pressure (bar). Affects Contexts created after this call,
     * not existing ones -- use the Pressure() parameter for those.
     */
    void setDefaultPressure(double pressure);

    /** Default temperature the system is held at (Kelvin). */
    double getDefaultTemperature() const {
        return defaultTemperature;
    }
    /**
     * Set the default temperature (Kelvin). Affects Contexts created after this
     * call, not existing ones -- use the Temperature() parameter for those.
     */
    void setDefaultTemperature(double temp);

    /** Steps between attempted volume moves; 0 disables the barostat. */
    int getFrequency() const {
        return frequency;
    }
    void setFrequency(int freq);

    /**
     * Random number seed. A seed of 0 (the default) draws a unique seed when a
     * Context is created, so independent Contexts get independent move
     * sequences. Two runs with the same non-zero seed take the same sequence of
     * proposals, which is what makes a barostat run reproducible.
     */
    int getRandomNumberSeed() const {
        return randomNumberSeed;
    }
    void setRandomNumberSeed(int seed) {
        randomNumberSeed = seed;
    }

    /** This is not a real force and does not use periodic boundary conditions. */
    bool usesPeriodicBoundaryConditions() const {
        return false;
    }

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    double defaultPressure, defaultTemperature;
    int frequency, randomNumberSeed;
};

} // namespace NvOpenMM

#endif // OPENMM_NV_MONTE_CARLO_BAROSTAT_H_
