// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "InteractionModel.hpp"
#include "../common/CupoStatus.hpp"
#include "nvopenmm/internal/NonbondedForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/CustomNonbondedForce.h"
#include "lepton/ExpressionProgram.h"
#include "lepton/Operation.h"
#include "lepton/ParsedExpression.h"
#include "lepton/Parser.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace NvOpenMM {

LatticeSumEvaluation::LatticeSumEvaluation(const LatticeSumInteraction& definition, double cutoff)
    : definition_(definition) {
    const float cutoff32 = static_cast<float>(cutoff);
    if (!(cutoff32 > 0) || !std::isfinite(cutoff32))
        throw OpenMM::OpenMMException("LatticeSumInteraction: cutoff must be positive and finite in FP32");
    RadialKernel::inversePower(definition.kernel.power, definition.kernel.coupling);
    const bool dispersion = definition.kernel.power == 6;
    cupo_splitting_plan_handle handle = nullptr;
    switch (definition.splitting) {
        case LatticeSumInteraction::Ewald:
            cupoCheck(dispersion ? cupo_splitting_plan_create_ewald_dispersion(cutoff, &handle) :
                      cupo_splitting_plan_create_ewald(cutoff, definition.kernel.coupling, &handle));
            break;
        case LatticeSumInteraction::Prolate:
            cupoCheck(dispersion ? cupo_splitting_plan_create_prolate_dispersion(cutoff, &handle) :
                      cupo_splitting_plan_create_prolate(cutoff, definition.kernel.coupling, &handle));
            break;
        default:
            throw OpenMM::OpenMMException("LatticeSumInteraction: unsupported splitting");
    }
    plan_.reset(handle);
}

struct LatticeSumEvaluation::WeightState {
    explicit WeightState(const Lepton::ParsedExpression& expression)
        // Evaluate the original tree. Lepton's constant-folding fixed point can
        // fail to terminate when an invalid domain produces NaN (sqrt(-1), for
        // example); preparation instead rejects its nonfinite result normally.
        : program(expression.createProgram()) {}
    Lepton::ExpressionProgram program;
    std::vector<std::pair<std::string, size_t>> particleFields;
    std::set<std::string> globalNames;
    size_t particleCount = 0;
    enum Evaluation { General, Charge, Dispersion } evaluation = General;
    size_t chargeIndex = 0, sigmaIndex = 0, epsilonIndex = 0;
};

LatticeSumEvaluation::~LatticeSumEvaluation() = default;
LatticeSumEvaluation::LatticeSumEvaluation(LatticeSumEvaluation&&) noexcept = default;
LatticeSumEvaluation& LatticeSumEvaluation::operator=(LatticeSumEvaluation&&) noexcept = default;

void LatticeSumEvaluation::initializeWeightSchema(const std::vector<std::string>& particleNames,
        const std::vector<std::string>& globalNames) {
    const auto expression = Lepton::Parser::parse(definition_.weightExpression);
    std::map<std::string, size_t> fields;
    std::set<std::string> globals;
    for (size_t i = 0; i < particleNames.size(); ++i)
        if (particleNames[i].empty() || particleNames[i] == "r" || !fields.emplace(particleNames[i], i).second)
            throw OpenMM::OpenMMException("LatticeSumInteraction: invalid particle weight schema");
    for (const auto& name : globalNames)
        if (name.empty() || !globals.insert(name).second)
            throw OpenMM::OpenMMException("LatticeSumInteraction: invalid global weight schema");
    // These built-in weight spellings always refer to the native particle
    // fields. Standard offset globals are allowed to have those same names;
    // unrelated declarations must not change the meaning of q or native b6.
    const bool particleWeight = (definition_.weightExpression == "q" && fields.count("q")) ||
        (definition_.weightExpression == "2*sqrt(epsilon)*sigma^3" &&
         fields.count("sigma") && fields.count("epsilon"));
    // Validate the original tree: 0*r must not conceal distance dependence.
    const auto validate = [&](const auto& self, const Lepton::ExpressionTreeNode& node) -> void {
        if (node.getOperation().getId() == Lepton::Operation::VARIABLE) {
            const auto& name = node.getOperation().getName();
            if (name == "r" || (!fields.count(name) && !globals.count(name)))
                throw OpenMM::OpenMMException("LatticeSumInteraction: unknown or forbidden weight variable " + name);
            if (!particleWeight && fields.count(name) && globals.count(name))
                throw OpenMM::OpenMMException("LatticeSumInteraction: ambiguous particle/global weight variable " + name);
        }
        for (const auto& child : node.getChildren()) self(self, child);
    };
    validate(validate, expression.getRootNode());
    auto state = std::make_unique<WeightState>(expression);
    state->particleCount = particleNames.size();
    std::set<std::string> used;
    for (int i = 0; i < state->program.getNumOperations(); ++i)
        if (state->program.getOperation(i).getId() == Lepton::Operation::VARIABLE)
            used.insert(state->program.getOperation(i).getName());
    for (const auto& name : used) {
        const auto field = fields.find(name);
        if (field != fields.end()) state->particleFields.push_back(*field);
        else state->globalNames.insert(name);
    }
    if (definition_.weightExpression == "q" && fields.count("q")) {
        state->evaluation = WeightState::Charge;
        state->chargeIndex = fields.at("q");
    } else if (definition_.weightExpression == "2*sqrt(epsilon)*sigma^3" &&
               fields.count("sigma") && fields.count("epsilon")) {
        state->evaluation = WeightState::Dispersion;
        state->sigmaIndex = fields.at("sigma");
        state->epsilonIndex = fields.at("epsilon");
    }
    weight_state_ = std::move(state);
}

LatticeSumEvaluation::PreparedWeights LatticeSumEvaluation::prepareWeights(
        const std::vector<float>& raw, int stride, const std::map<std::string, double>& globals) const {
    if (!weight_state_)
        throw OpenMM::OpenMMException("LatticeSumInteraction: weight schema was not initialized");
    const auto& state = *weight_state_;
    if (stride <= 0 || size_t(stride) < state.particleCount || raw.size()%size_t(stride))
        throw OpenMM::OpenMMException("LatticeSumInteraction: invalid weight particle record stride");
    std::map<std::string, double> variables;
    for (const auto& name : state.globalNames) {
        const auto found = globals.find(name);
        if (found == globals.end() || !std::isfinite(static_cast<float>(found->second)))
            throw OpenMM::OpenMMException("LatticeSumInteraction: missing or nonfinite weight global " + name);
        variables[name] = static_cast<float>(found->second);
    }
    PreparedWeights prepared;
    prepared.values.resize(raw.size()/size_t(stride));
    // P6 plans carry unit attraction; absorb the physical coupling into both
    // scalar weights so spread, gather, exclusions and self terms agree.
    const double scale = definition_.kernel.power == 6 ? std::sqrt(-definition_.kernel.coupling) : 1.0;
    for (size_t atom = 0; atom < prepared.values.size(); ++atom) {
        const float* row = raw.data()+atom*size_t(stride);
        double evaluated;
        if (state.evaluation == WeightState::Charge) {
            evaluated = row[state.chargeIndex];
        } else if (state.evaluation == WeightState::Dispersion) {
            // Preserve native preprocessing arithmetic and avoid per-atom map
            // updates or a CPU interpreter for the common built-in weights.
            const double sigma = row[state.sigmaIndex], epsilon = row[state.epsilonIndex];
            evaluated = 2*std::sqrt(epsilon)*sigma*sigma*sigma;
        } else {
            for (const auto& field : state.particleFields) {
                const float value = row[field.second];
                if (!std::isfinite(value))
                    throw OpenMM::OpenMMException("LatticeSumInteraction: nonfinite weight particle parameter " + field.first);
                variables[field.first] = value;
            }
            evaluated = state.program.evaluate(variables);
        }
        const float value = static_cast<float>(scale*evaluated);
        if (!std::isfinite(value))
            throw OpenMM::OpenMMException("LatticeSumInteraction: weight expression produced a nonfinite FP32 value");
        prepared.values[atom] = value;
        prepared.sum += value;
        prepared.sumSquares += double(value)*value;
    }
    if (!std::isfinite(prepared.sum) || !std::isfinite(prepared.sumSquares))
        throw OpenMM::OpenMMException("LatticeSumInteraction: weight moments are not finite");
    return prepared;
}

void LatticeSumEvaluation::commitWeights(PreparedWeights&& prepared) {
    const bool changed = weights_version_ == 0 || weights_.size() != prepared.values.size() ||
        (!weights_.empty() && std::memcmp(weights_.data(), prepared.values.data(), weights_.size()*sizeof(float)) != 0);
    updateChargeMoments(prepared.sum, prepared.sumSquares);
    weights_ = std::move(prepared.values);
    if (changed && ++weights_version_ == 0) ++weights_version_;
}

const std::set<std::string>& LatticeSumEvaluation::globalDependencies() const {
    static const std::set<std::string> empty;
    return weight_state_ ? weight_state_->globalNames : empty;
}

void LatticeSumEvaluation::updateChargeMoments(double sumQ, double sumQ2) {
    cupoCheck(cupo_splitting_plan_get_self_energy(plan(), sumQ2, &self_energy_));
    sum_q_ = sumQ;
}

double LatticeSumEvaluation::correctionEnergy(double volume) const {
    double background = 0.0;
    if (volume > 0.0)
        cupoCheck(cupo_splitting_plan_get_net_charge_energy(plan(), sum_q_, volume, &background));
    return self_energy_ + background;
}

double CutoffEvaluation::tailCoefficientFor(const NonbondedForce& force) const {
    return force.getUseDispersionCorrection()
        ? NonbondedForceImpl::calcDispersionCorrection(force) : 0.0;
}

void CutoffEvaluation::updateTail(const NonbondedForce& force, double coefficient) {
    updateTail(force.getCutoffInteraction().tail, coefficient);
}

void CutoffEvaluation::updateTail(CutoffInteraction::TailPolicy policy, double coefficient) {
    definition_.tail = policy;
    tail_coefficient_ = coefficient;
}

InteractionModel InteractionModel::fromForce(const NonbondedForce& force) {
    InteractionModel model(force.getCutoffDistance(), force.getCutoffInteraction());
    model.builtin_parameters_ = force.getPairMode() == NonbondedForce::Standard;
    model.expression_ = !model.builtin_parameters_ || model.cutoffInteraction.usesExpression();
    for (int i = 0; i < force.getNumLatticeSumInteractions(); ++i) {
        const auto definition = force.getLatticeSumInteraction(i);
        model.lattice_definitions_.push_back(definition);
        auto& slot = definition.kernel.power == 1 ? model.latticeSum : model.dispersion;
        if (slot)
            throw OpenMM::OpenMMException("NonbondedForce: at most one lattice component per inverse power is currently supported");
        slot.emplace(definition, model.cutoff);
        const bool canonical = definition.kernel.power == 1 ? definition.weightExpression == "q" :
            definition.weightExpression == "2*sqrt(epsilon)*sigma^3" && definition.kernel.coupling == -1.0;
        if (!canonical) model.expression_ = true;
    }
    if ((!model.latticeSum && !model.dispersion) ||
            (model.latticeSum && model.dispersion &&
             model.latticeSum->definition().splitting != model.dispersion->definition().splitting))
        model.expression_ = true;
    if (model.dispersion && (force.getUseSwitchingFunction() ||
            model.cutoffInteraction.definition().tail != CutoffInteraction::NoTail ||
            (model.builtin_parameters_ && force.hasCutoffExpression() &&
             force.getCutoffExpressionDefinition().getUseLongRangeCorrection())))
        throw OpenMM::OpenMMException("LJPME: switching and homogeneous dispersion tails are incompatible with the lattice sum");
    if (!model.builtin_parameters_) {
        if (!model.cutoffInteraction.usesExpression())
            throw OpenMM::OpenMMException("NonbondedForce: Custom mode requires a cutoff expression");
        NonbondedForceImpl::validateCustomDefinition(force.getCustomDefinition());
    } else {
        if (model.cutoffInteraction.usesExpression()) {
            NonbondedForceImpl::validateCutoffReplacement(force, force.getCutoffExpressionDefinition());
        } else {
            model.cutoffInteraction.updateTail(force, model.cutoffInteraction.tailCoefficientFor(force));
        }
    }
    return model;
}

void InteractionModel::validateUpdate(const NonbondedForce& force) const {
    const auto interaction = force.getCutoffInteraction();
    if (force.getCutoffDistance() != cutoff ||
            (force.getPairMode() == NonbondedForce::Standard) != builtin_parameters_ ||
            size_t(force.getNumLatticeSumInteractions()) != lattice_definitions_.size() ||
            interaction.evaluator != cutoffInteraction.definition().evaluator)
        throw OpenMM::OpenMMException("updateParametersInContext: interaction policy changed; reinitialize Context");
    for (size_t i = 0; i < lattice_definitions_.size(); ++i) {
        const auto next = force.getLatticeSumInteraction(int(i));
        const auto& previous = lattice_definitions_[i];
        if (next.kernel.power != previous.kernel.power || next.kernel.coupling != previous.kernel.coupling ||
                next.splitting != previous.splitting || next.weightExpression != previous.weightExpression)
            throw OpenMM::OpenMMException("updateParametersInContext: lattice interaction changed; reinitialize Context");
    }
    if (dispersion && (force.getUseSwitchingFunction() || interaction.tail != CutoffInteraction::NoTail ||
            (builtin_parameters_ && force.hasCutoffExpression() &&
             force.getCutoffExpressionDefinition().getUseLongRangeCorrection())))
        throw OpenMM::OpenMMException("LJPME: switching and homogeneous dispersion tails are incompatible with the lattice sum");
    if (!cutoffInteraction.usesExpression()) {
        const auto& previous = cutoffInteraction.definition();
        if (interaction.combiningRule != previous.combiningRule)
            throw OpenMM::OpenMMException("updateParametersInContext: LJ combining rule has changed; reinitialize Context");
        if (interaction.useSwitching != previous.useSwitching ||
                (interaction.useSwitching && interaction.switchingDistance != previous.switchingDistance))
            throw OpenMM::OpenMMException("updateParametersInContext: LJ switching settings have changed; reinitialize Context");
    }
    // CustomPairParameters validates expression code, records, names and exclusions.
    // The native cutoff tail flag remains an allowed explicit parameter update.
}

} // namespace NvOpenMM
