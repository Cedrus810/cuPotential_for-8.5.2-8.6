// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

namespace NvOpenMM {

/** A pair potential evaluated explicitly within a cutoff, with optional tail treatment.
 *
 * The cutoff applies to explicit pair evaluation. A separate tail correction
 * may estimate the omitted contribution, including energy removed by switching.
 * A Custom expression may also contain bare lattice interactions: separately
 * declared lattice terms supply their direct/reciprocal decomposition.
 *
 * BuiltinLJ supplies the native 12-6 expression, with its own combining rule,
 * optional quintic switch, and optional tail correction. Expression
 * supplies the custom reference energy, including any desired combining or
 * shifting. An optional quintic switch multiplies the complete cutoff expression
 * during direct evaluation. HomogeneousTail estimates the omitted energy by
 * integrating the reference potential beyond the cutoff and the energy removed
 * by that switch, assuming a homogeneous particle distribution. The integration
 * method is internal: analytic for built-in LJ, numerical for expressions. The
 * coefficient is cached between parameter changes and contributes C/volume.
 * Expression tails follow live globals; native LJ tails follow effective
 * parameters by default. A switch written into the expression itself remains
 * literal and does not imply an unswitched reference.
 * These are physical definitions; native versus linked execution is selected
 * when the complete force is lowered to the runtime.
 */
struct CutoffInteraction {
    enum Evaluator { BuiltinLJ = 0, Expression = 1 };
    enum CombiningRule { LorentzBerthelot = 0, Geometric = 1 };
    enum TailPolicy { NoTail = 0, HomogeneousTail = 1 };

    Evaluator evaluator = BuiltinLJ;
    CombiningRule combiningRule = LorentzBerthelot;
    bool useSwitching = false;
    double switchingDistance = -1.0;
    TailPolicy tail = NoTail;
    std::string expression;
};

/** A radial lattice kernel coupling/r^power, independent of particle weights.
 * Currently supports positive P1 coupling and negative P6 coupling.
 * P8 and other powers are reserved for future implementations and rejected.
 * Both fields are explicit: a generic kernel does not silently select Coulomb
 * or its molecular-unit conversion factor. Coupling is signed; absorbing its
 * magnitude into real particle weights cannot represent the attractive sign.
 */
struct RadialKernel {
    RadialKernel(int power, double coupling);
    int power;
    double coupling;
    static RadialKernel inversePower(int power, double coupling);
};

/** Construct P1 electrostatics; the default coupling is in kJ*nm/(mol*e^2). */
RadialKernel CoulombRadialKernel(double coupling = 138.93545764498);

/** Construct attractive P6 dispersion; weights conventionally contain sqrt(C6). */
RadialKernel DispersionRadialKernel(double coupling = -1.0);

/** One independently weighted periodic lattice sum.
 *
 * Ewald and Prolate select computational splittings of the same periodic target.
 * The cutoff controls the direct/reciprocal decomposition; the target includes
 * interactions with periodic images beyond that cutoff.
 * The selected splitting defines both pieces and their matching self, background and
 * exclusion corrections. The target pair potential is
 * kernel.coupling*weight_i*weight_j/r^kernel.power. weightExpression uses
 * unsuffixed particle parameters and globals, with no distance dependence.
 * Standard exposes q, sigma, epsilon; Custom uses its declared parameter schema.
 * Dispersion uses C2-order-3 spreading and the corresponding Coulomb-sized mesh.
 * Lattice terms are never switched. A custom full pair expression must contain
 * the corresponding bare interactions; the runtime subtracts their reciprocal
 * contributions without adding a second bare interaction.
 * P6 retains the LJPME cutoff convention: its direct complement is
 * -kernel.coupling*weight_i*weight_j*(LR6(r)-LR6(cutoff)). Custom source is not
 * shifted implicitly; explicitly shift its non-Coulomb part at the cutoff to
 * reproduce Standard/OpenMM LJPME's direct-energy convention.
 * Automatic switching with P1 requires a recognizable additive bare Coulomb
 * term; otherwise write the intended cutoff switch explicitly in the source.
 * Automatic switching with P6 is unsupported.
 */
struct LatticeSumInteraction {
    enum SplittingMethod { Ewald = 0, Prolate = 1 };
    LatticeSumInteraction(const RadialKernel& kernel, const std::string& weightExpression,
                          SplittingMethod splitting = Prolate)
        : kernel(kernel), splitting(splitting), weightExpression(weightExpression) {}
    RadialKernel kernel;
    SplittingMethod splitting;
    std::string weightExpression;
};

} // namespace NvOpenMM
