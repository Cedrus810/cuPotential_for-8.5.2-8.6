// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENMM_NV_LANGEVIN_MIDDLE_INTEGRATOR_H_
#define OPENMM_NV_LANGEVIN_MIDDLE_INTEGRATOR_H_

#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include <iosfwd>
#include <string>
#include <vector>

namespace NvOpenMM {

class MonteCarloBarostatImpl;   // for the barostat force-fusion hook in step()



/**
 * A LangevinMiddle (BAOAB) integrator, GPU-accelerated by the Nvidia/Cupo plugin.
 *
 * It reproduces OpenMM's stock LangevinMiddleIntegrator math and precision
 * (single or mixed; double is not supported), with its own SETTLE/SHAKE
 * constraint kernels and counter-based Philox noise. Checkpoints preserve the
 * resolved noise seed and private step counter. Another intentional difference is
 * center-of-mass motion removal: instead of a separate CMMotionRemover force
 * running a two-pass reduction every step, this integrator removes the COM
 * velocity itself with a single-pass reduction, every comFrequency steps.  When
 * you use this integrator, remove any CMMotionRemover from the System before
 * creating a Context, or both will run.
 */
class LangevinMiddleIntegrator : public OpenMM::Integrator {
public:
    /**
     * @param temperature    heat-bath temperature (Kelvin)
     * @param frictionCoeff  friction coefficient coupling the system to the bath (1/ps)
     * @param stepSize       integration step size (ps)
     * @param comFrequency   remove center-of-mass motion every this many steps
     *                       (<=0 disables COM removal). Default 100.
     */
    LangevinMiddleIntegrator(double temperature, double frictionCoeff, double stepSize, int comFrequency = 100);

    /** Borrow the plugin type from an OpenMM Integrator; nullptr for other types. */
    static LangevinMiddleIntegrator* cast(OpenMM::Integrator& integrator);

    double getTemperature() const {
        return temperature;
    }
    void setTemperature(double temp);
    double getFriction() const {
        return friction;
    }
    void setFriction(double coeff);
    int getRandomNumberSeed() const {
        return randomNumberSeed;
    }
    void setRandomNumberSeed(int seed) {
        randomNumberSeed = seed;
    }
    /** Number of steps between center-of-mass motion removals (<=0 disables). */
    int getComFrequency() const {
        return comFrequency;
    }
    void setComFrequency(int frequency) {
        comFrequency = frequency;
    }
    /** Advance the simulation by taking a series of time steps. */
    void step(int steps);
    /** Diagnostic counters for the guarded whole-step CUDA graph. */
    long long getWholeStepGraphCaptureCount() const;
    long long getWholeStepGraphReplayCount() const;
protected:
    void initialize(OpenMM::ContextImpl& context);
    void cleanup();
    std::vector<std::string> getKernelNames();
    double computeKineticEnergy();
    bool kineticEnergyRequiresForce() const;
    void createCheckpoint(std::ostream& stream) const override;
    void loadCheckpoint(std::istream& stream) override;
    double getVelocityTimeOffset() const {
        return 0.0;
    }
private:
    // Resolved once from ContextImpl::getForceImpls(); NULL when no NV barostat is
    // attached, which is the ordinary NVT case.
    const MonteCarloBarostatImpl* findFusingBarostat(OpenMM::ContextImpl& ctx);
    const MonteCarloBarostatImpl* fusingBarostat = nullptr;
    bool barostatResolved = false;
    double temperature, friction;
    int randomNumberSeed;
    int comFrequency;
    OpenMM::Kernel kernel;
};

} // namespace NvOpenMM

#endif // OPENMM_NV_LANGEVIN_MIDDLE_INTEGRATOR_H_
