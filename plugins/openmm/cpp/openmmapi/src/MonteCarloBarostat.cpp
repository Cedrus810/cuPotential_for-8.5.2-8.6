// The constructor and setters follow OpenMM openmmapi/src/MonteCarloBarostat.cpp:
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

#include "nvopenmm/MonteCarloBarostat.h"
#include "nvopenmm/internal/MonteCarloBarostatImpl.h"
#include "openmm/OpenMMException.h"


// Definitions live in the plugin's namespace, so our NonbondedForce et al.
// win unqualified lookup over OpenMM's same-named classes rather than
// being ambiguous with them.
namespace NvOpenMM {

MonteCarloBarostat::MonteCarloBarostat(double defaultPressure, double defaultTemperature,
                                           int frequency) {
    setDefaultPressure(defaultPressure);
    setDefaultTemperature(defaultTemperature);
    setFrequency(frequency);
    setRandomNumberSeed(0);
}

MonteCarloBarostat* MonteCarloBarostat::cast(OpenMM::Force& force) {
    return dynamic_cast<MonteCarloBarostat*>(&force);
}

void MonteCarloBarostat::setDefaultPressure(double pressure) {
    if (pressure < 0)
        throw OpenMM::OpenMMException("MonteCarloBarostat: pressure cannot be negative");
    defaultPressure = pressure;
}

void MonteCarloBarostat::setDefaultTemperature(double temp) {
    if (temp < 0)
        throw OpenMM::OpenMMException("MonteCarloBarostat: temperature cannot be negative");
    defaultTemperature = temp;
}

void MonteCarloBarostat::setFrequency(int freq) {
    if (freq < 0)
        throw OpenMM::OpenMMException("MonteCarloBarostat: frequency cannot be negative");
    frequency = freq;
}

OpenMM::ForceImpl* MonteCarloBarostat::createImpl() const {
    return new MonteCarloBarostatImpl(*this);
}

} // namespace NvOpenMM
