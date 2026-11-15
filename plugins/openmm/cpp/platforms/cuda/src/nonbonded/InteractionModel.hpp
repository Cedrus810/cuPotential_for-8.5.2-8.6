// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "nvopenmm/NonbondedForce.h"
#include "cupo/splitting_plan.h"
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace NvOpenMM {

// kJ*nm/(mol*e^2), shared by plans and the full expression supplied to libcupo.
constexpr double CoulombCoupling = 138.93545764498;

// Execution state for the cutoff term. Its correction is independent of
// the Coulomb splitting. Custom expressions define their own mixing and may use
// the cutoff term's optional switch.
class CutoffEvaluation {
public:
    explicit CutoffEvaluation(const CutoffInteraction& definition)
        : definition_(definition) {}
    const CutoffInteraction& definition() const { return definition_; }
    bool usesExpression() const { return definition_.evaluator == CutoffInteraction::Expression; }
    double tailCoefficientFor(const NonbondedForce& force) const;
    void updateTail(const NonbondedForce& force, double coefficient);
    void updateTail(CutoffInteraction::TailPolicy policy, double coefficient);
    void updateTailCoefficient(double coefficient) { tail_coefficient_ = coefficient; }
    double correctionEnergy(double volume) const { return tail_coefficient_/volume; }
private:
    CutoffInteraction definition_;
    double tail_coefficient_ = 0.0;
};

// Evaluation state for a periodic Coulomb or geometric dispersion target. One
// immutable plan supplies direct/reciprocal pieces and weight-dependent self
// and background corrections (dispersion has no background correction).
class LatticeSumEvaluation {
public:
    struct PreparedWeights {
        std::vector<float> values;
        double sum = 0.0, sumSquares = 0.0;
    };
    LatticeSumEvaluation(const LatticeSumInteraction& definition, double cutoff);
    ~LatticeSumEvaluation();
    LatticeSumEvaluation(LatticeSumEvaluation&&) noexcept;
    LatticeSumEvaluation& operator=(LatticeSumEvaluation&&) noexcept;
    const LatticeSumInteraction& definition() const { return definition_; }
    cupo_splitting_plan_handle plan() const { return plan_.get(); }
    void initializeWeightSchema(const std::vector<std::string>& particleNames,
                                const std::vector<std::string>& globalNames);
    PreparedWeights prepareWeights(const std::vector<float>& raw, int stride,
                                   const std::map<std::string, double>& globals) const;
    void commitWeights(PreparedWeights&& prepared);
    const std::vector<float>& weights() const { return weights_; }
    uint64_t weightsVersion() const { return weights_version_; }
    const std::set<std::string>& globalDependencies() const;
    void updateChargeMoments(double sumQ, double sumQ2);
    double correctionEnergy(double volume) const;
private:
    struct PlanDeleter {
        void operator()(cupo_splitting_plan_handle value) const { cupo_splitting_plan_destroy(value); }
    };
    LatticeSumInteraction definition_;
    std::unique_ptr<cupo_splitting_plan_impl, PlanDeleter> plan_;
    struct WeightState;
    std::unique_ptr<WeightState> weight_state_;
    std::vector<float> weights_;
    uint64_t weights_version_ = 0;
    double self_energy_ = 0.0;
    double sum_q_ = 0.0;
};

// Normalize native LJ or a cutoff expression, optionally accompanied by Coulomb
// and dispersion lattice sums. Every composition uses one cell list/traversal.
class InteractionModel {
public:
    static InteractionModel fromForce(const NonbondedForce& force);
    void validateUpdate(const NonbondedForce& force) const;
    bool usesExpression() const { return expression_; }
    bool hasBuiltinParameters() const { return builtin_parameters_; }
    bool hasLatticeSum() const { return latticeSum.has_value() || dispersion.has_value(); }

    double cutoff;
    CutoffEvaluation cutoffInteraction;
    std::optional<LatticeSumEvaluation> latticeSum;
    std::optional<LatticeSumEvaluation> dispersion;
private:
    InteractionModel(double cutoff, const CutoffInteraction& interaction)
        : cutoff(cutoff), cutoffInteraction(interaction) {}
    bool builtin_parameters_ = true, expression_ = false;
    std::vector<LatticeSumInteraction> lattice_definitions_;
};

} // namespace NvOpenMM
