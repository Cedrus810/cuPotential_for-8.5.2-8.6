// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#ifdef NVOPENMM_TEST_EXPRESSION_TAIL
#include <cstddef>
#endif

namespace OpenMM {
class Context;
class CustomNonbondedForce;
}

namespace NvOpenMM {

// CPU-only tail correction for a cutoff expression. This owns an independent
// copy of the original definition, not the packed/precomputed DSpace records or
// any injected split Coulomb term. The owner updates both parameter snapshots
// after validating their shared immutable contract. No CUDA work is performed.
//
// The supplied expression is integrated literally. An explicit OpenMM switching
// flag adds the stock switching-deficit correction; a switch written into the
// expression does not imply an unswitched reference model.
class ExpressionTailCorrection {
public:
    explicit ExpressionTailCorrection(const OpenMM::CustomNonbondedForce& force);
    ~ExpressionTailCorrection();
    // Validate all globals, but integrate only when energy is requested. Reuse
    // coefficients for unchanged or previously visited relevant global values.
    void prepareStep(const OpenMM::Context& context, bool includeEnergy=true);
    // Validate an explicit particle update and return its prepared replacement.
    // The original classes and coefficient remain valid if validation or
    // integration fails. The owner commits only after all components succeed.
    std::unique_ptr<ExpressionTailCorrection> prepareUpdate(
        const OpenMM::CustomNonbondedForce& force, const OpenMM::Context& context) const;
    // The caller adds coefficient()/volume to energy using the current box.
    double coefficient() const;

#ifdef NVOPENMM_TEST_EXPRESSION_TAIL
    // Standalone CPU regression instrumentation; absent from the plugin build.
    struct TestStats {
        std::size_t coefficientEvaluations=0, radialIntegrals=0, cacheHits=0, cacheEntries=0;
    };
    TestStats testStats() const;
#endif

private:
    void validateDefinition(const OpenMM::CustomNonbondedForce& force) const;
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace NvOpenMM
