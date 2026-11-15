// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <stdexcept>

namespace cupo {

// Radial integrals of the omitted r^-6 and r^-12 terms. For U=C12/r^12-C6/r^6,
// the homogeneous correction is 2*pi*N^2/V * (<C12>*twelfth - <C6>*sixth).
struct LJTailIntegrals {
    double sixth;
    double twelfth;
};

inline LJTailIntegrals lj_tail_integrals(double cutoff)
{
    if (!std::isfinite(cutoff) || cutoff <= 0.0)
        throw std::invalid_argument("LJ tail requires a positive finite cutoff");
    return {1.0 / (3.0 * std::pow(cutoff, 3)),
            1.0 / (9.0 * std::pow(cutoff, 9))};
}

// Includes the entire exterior tail and the energy removed by the quintic switch.
// start==cutoff is the zero-width limit. start==0 has divergent LJ integrals.
inline LJTailIntegrals lj_switched_tail_integrals(double start, double cutoff)
{
    const auto unswitched = lj_tail_integrals(cutoff);
    if (!std::isfinite(start) || !(start > 0.0) || start > cutoff)
        throw std::invalid_argument("switched LJ tail requires 0 < start <= cutoff");
    if (start == cutoff) return unswitched;

    const double t = start / cutoff;
    const double twelfth = (((5.0*t + 9.0)*t + 9.0)*t + 5.0)
                         / (252.0 * std::pow(start, 6) * std::pow(cutoff, 3));
    const double z = (cutoff - start) / (cutoff + start);
    const double z2 = z*z;
    double sixth;
    if (z2 <= 0.61) {
        // Positive series avoids subtracting nearly equal endpoint integrals.
        double term = 1.0/15.0, sum = term;
        for (unsigned n = 0; ; ++n) {
            term *= z2 * (n + 2.0) * (2.0*n + 3.0)
                       / ((n + 1.0) * (2.0*n + 7.0));
            const double updated = sum + term;
            if (updated == sum) break;
            sum = updated;
        }
        sixth = 40.0 * sum / std::pow(cutoff + start, 3);
    } else {
        // Wide intervals have no severe cancellation in the logarithmic form.
        sixth = 10.0 * ((t*t + 4.0*t + 1.0)*std::log(cutoff/start)
                       - 3.0*(1.0 - t*t))
              / (std::pow(cutoff, 3) * std::pow(1.0 - t, 5));
    }
    return {sixth, twelfth};
}

} // namespace cupo
