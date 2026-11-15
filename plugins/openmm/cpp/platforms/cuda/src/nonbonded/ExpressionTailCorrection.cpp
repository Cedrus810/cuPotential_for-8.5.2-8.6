// Class averaging and midpoint quadrature follow OpenMM's CustomNonbondedForce
// correction. Use the scalar ExpressionProgram interface: the compiled-vector
// class changes layout with OpenMM's private LEPTON_USE_JIT build option and
// cannot safely be owned or destroyed by a separately built plugin.
//
// Adapted from OpenMM CustomNonbondedForceImpl.cpp:
// Copyright (c) 2008-2024 Stanford University and the Authors.
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
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT

#include "ExpressionTailCorrection.hpp"
#include "openmm/Context.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "lepton/ExpressionProgram.h"
#include "lepton/Operation.h"
#include "lepton/ParsedExpression.h"
#include "lepton/Parser.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace NvOpenMM {

namespace {
void requireDefinition(bool condition) {
    if (!condition)
        throw OpenMM::OpenMMException("CustomNonbondedForce: long range correction definition "
                                      "changed; reinitialize the Context");
}

void validateValues(const OpenMM::CustomNonbondedForce& force) {
    if (force.getNumTabulatedFunctions() || force.getNumComputedValues() ||
            force.getNumEnergyParameterDerivatives())
        throw OpenMM::OpenMMException("CustomNonbondedForce: expression tail correction does not "
                                      "support tabulated functions, computed values, or energy derivatives");
    if (!std::isfinite(force.getCutoffDistance()) || force.getCutoffDistance() <= 0)
        throw OpenMM::OpenMMException("CustomNonbondedForce: tail cutoff must be positive and finite");
    if (force.getUseSwitchingFunction() && (!std::isfinite(force.getSwitchingDistance()) ||
            force.getSwitchingDistance() < 0 || force.getSwitchingDistance() >= force.getCutoffDistance()))
        throw OpenMM::OpenMMException("CustomNonbondedForce: tail switching distance must be in [0, cutoff)");
    for (int i=0; i<force.getNumGlobalParameters(); ++i)
        if (!std::isfinite(static_cast<float>(force.getGlobalParameterDefaultValue(i))))
            throw OpenMM::OpenMMException("CustomNonbondedForce globals must be finite FP32 values");
    for (int i=0; i<force.getNumParticles(); ++i) {
        std::vector<double> parameters;
        force.getParticleParameters(i, parameters);
        if (parameters.size() != static_cast<size_t>(force.getNumPerParticleParameters()))
            throw OpenMM::OpenMMException("CustomNonbondedForce: wrong number of particle parameters");
        for (double value : parameters)
            if (!std::isfinite(static_cast<float>(value)))
                throw OpenMM::OpenMMException("CustomNonbondedForce parameters must be finite FP32 values");
    }
    for (int i=0; i<force.getNumInteractionGroups(); ++i) {
        std::set<int> first, second;
        force.getInteractionGroupParameters(i, first, second);
        for (const auto* group : {&first, &second})
            for (int atom : *group)
                if (atom < 0 || atom >= force.getNumParticles())
                    throw OpenMM::OpenMMException("CustomNonbondedForce: invalid tail interaction-group atom");
    }
}
} // namespace

struct ExpressionTailCorrection::State {
    explicit State(const OpenMM::CustomNonbondedForce& force)
        : definition(force), program(Lepton::Parser::parse(force.getEnergyFunction()).optimize().createProgram()) {
        validateValues(force);
        std::set<std::string> referenced;
        for (int i=0; i<program.getNumOperations(); ++i)
            if (program.getOperation(i).getId()==Lepton::Operation::VARIABLE)
                referenced.insert(program.getOperation(i).getName());
        for (int i=0; i<force.getNumGlobalParameters(); ++i)
            if (referenced.count(force.getGlobalParameterName(i))) usedGlobals.push_back(i);
        for (int i=0; i<force.getNumPerParticleParameters(); ++i) {
            const auto& name=force.getPerParticleParameterName(i);
            if (referenced.count(name+"1")) usedFirstFields.push_back(i);
            if (referenced.count(name+"2")) usedSecondFields.push_back(i);
        }
        rebuildClasses(force);
        coefficients.reserve(coefficientCacheCapacity);
        if (active() && !interactions.empty() && !referenced.count("r"))
            throw OpenMM::OpenMMException("CustomNonbondedForce: Cannot use long range correction "
                                          "with a force that does not depend on r.");
    }

    static uint64_t bits(double value) {
        uint64_t result;
        static_assert(sizeof(result)==sizeof(value), "Tail cache requires 64-bit doubles");
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }

    bool active() const {
        return definition.getUseLongRangeCorrection() &&
               definition.getNonbondedMethod()==OpenMM::CustomNonbondedForce::CutoffPeriodic;
    }

    void rebuildClasses(const OpenMM::CustomNonbondedForce& force) {
        std::map<std::vector<double>,int> indices;
        std::vector<std::vector<double>> nextClasses;
        std::vector<int> atomClass(force.getNumParticles());
        std::vector<long long> counts;
        for (int atom=0; atom<force.getNumParticles(); ++atom) {
            std::vector<double> parameters;
            force.getParticleParameters(atom,parameters);
            const auto added=indices.emplace(parameters,static_cast<int>(nextClasses.size()));
            if (added.second) {
                nextClasses.push_back(parameters);
                counts.push_back(0);
            }
            atomClass[atom]=added.first->second;
            ++counts[atomClass[atom]];
        }
        std::map<std::pair<int,int>,long long> nextInteractions;
        if (force.getNumInteractionGroups()==0) {
            // Stock's ungrouped mean-field model includes the diagonal class
            // self count. Exclusions are not subtracted from either model.
            for (int i=0; i<static_cast<int>(nextClasses.size()); ++i)
                for (int j=i; j<static_cast<int>(nextClasses.size()); ++j)
                    nextInteractions[{i,j}]=i==j ? counts[i]*(counts[i]+1)/2 : counts[i]*counts[j];
        } else {
            // Count classes in each set and their intersection, not atom pairs.
            // Each group has its own multiplicity, including repeated groups.
            const int nclasses=static_cast<int>(nextClasses.size());
            for (int group=0; group<force.getNumInteractionGroups(); ++group) {
                std::set<int> first, second;
                force.getInteractionGroupParameters(group, first, second);
                std::vector<long long> a(nclasses), b(nclasses), common(nclasses);
                for (int atom : first) {
                    ++a[atomClass[atom]];
                    if (second.count(atom)) ++common[atomClass[atom]];
                }
                for (int atom : second) ++b[atomClass[atom]];
                for (int i=0; i<nclasses; ++i) {
                    if (a[i]==0 && b[i]==0) continue;
                    for (int j=i; j<nclasses; ++j) {
                        const long long count=i==j ? a[i]*b[i]-common[i]*(common[i]+1)/2 :
                            a[i]*b[j]+a[j]*b[i]-common[i]*common[j];
                        if (count==0) continue;
                        auto& total=nextInteractions[{i,j}];
                        if (count > std::numeric_limits<long long>::max()-total)
                            throw OpenMM::OpenMMException("CustomNonbondedForce: tail interaction count overflow");
                        total+=count;
                    }
                }
            }
        }
        // Different complete records can have identical radial integrands. Keep
        // the original classes, counts and summation order, and only share the
        // integral. First/second roles remain distinct for arbitrary expressions.
        std::map<std::vector<uint64_t>,size_t> integralIds;
        std::vector<size_t> nextIntegralIndices;
        nextIntegralIndices.reserve(nextInteractions.size());
        for (const auto& entry : nextInteractions) {
            std::vector<uint64_t> key;
            key.reserve(usedFirstFields.size()+usedSecondFields.size());
            for (int field : usedFirstFields)
                key.push_back(bits(nextClasses[entry.first.first][field]));
            for (int field : usedSecondFields)
                key.push_back(bits(nextClasses[entry.first.second][field]));
            const auto added=integralIds.emplace(std::move(key),integralIds.size());
            nextIntegralIndices.push_back(added.first->second);
        }
        integralCount=integralIds.size();
        integralIndices.swap(nextIntegralIndices);
        classes.swap(nextClasses);
        interactions.swap(nextInteractions);
        dirty=true;
    }

    template<class Function>
    static double integrate(Function function,const char* failure) {
        double sum=0;
        int points=1;
        for (int iteration=0;;++iteration,points*=3) {
            double added=0;
            for (int point=0;point<points;++point)
                if (point%3!=1) added+=function((point+.5)/points);
            const double previous=sum;
            sum=added/points+previous/3;
            const double relative=std::fabs((sum-previous)/sum);
            if (std::isfinite(sum) && iteration>2 && (relative<1e-5 || sum==0)) return sum;
            if (!std::isfinite(sum) || iteration==10 || (iteration>7 && relative>1e-3))
                throw OpenMM::OpenMMException(std::string("CustomNonbondedForce: Long range correction "
                                                          "did not converge. ")+failure);
        }
    }

    double evaluateCoefficient(const std::vector<double>& currentGlobals) const {
#ifdef NVOPENMM_TEST_EXPRESSION_TAIL
        ++stats.coefficientEvaluations;
#endif
        const auto& force=definition;
        if (!active() || force.getNumParticles()==0 || interactions.empty()) return 0;
        std::map<std::string,double> variables;
        // Preserve stock's FP32 inputs and evaluated values, while using the
        // scalar ABI to avoid OpenMM's private compiled-vector object layout.
        for (int i=0;i<force.getNumGlobalParameters();++i)
            variables[force.getGlobalParameterName(i)]=static_cast<float>(currentGlobals[i]);
        const double cutoff=force.getCutoffDistance();
        double sum=0;
        std::vector<double> integrals(integralCount);
        std::vector<bool> ready(integralCount,false);
        size_t interaction=0;
        for (const auto& entry:interactions) {
            const size_t integral=integralIndices[interaction++];
            if (ready[integral]) {
                sum+=entry.second*integrals[integral];
                continue;
            }
            const auto& first=classes[entry.first.first];
            const auto& second=classes[entry.first.second];
            for (int i=0;i<force.getNumPerParticleParameters();++i) {
                const auto& name=force.getPerParticleParameterName(i);
                variables[name+"1"]=static_cast<float>(first[i]);
                variables[name+"2"]=static_cast<float>(second[i]);
            }
#ifdef NVOPENMM_TEST_EXPRESSION_TAIL
            ++stats.radialIntegrals;
#endif
            const double far=integrate([&](double x) {
                const float r=static_cast<float>(cutoff/x);
                variables["r"]=r;
                const float value=static_cast<float>(program.evaluate(variables));
                const float r2=r*r;
                return static_cast<double>(value*r2*r2);
            },"Does the energy decay faster than 1/r^3?")/cutoff;
            double deficit=0;
            if (force.getUseSwitchingFunction()) {
                const double start=force.getSwitchingDistance(), width=cutoff-start;
                deficit=width*integrate([&](double x) {
                    const float r=static_cast<float>(start+x*width);
                    variables["r"]=r;
                    const float value=static_cast<float>(program.evaluate(variables));
                    const double removed=x*x*x*(10+x*(-15+x*6));
                    return removed*value*r*r;
                },"Is the energy finite everywhere in the switching interval?");
            }
            // A manually written switch is just part of U(r); only the explicit
            // OpenMM switching flag supplies an additional unswitched deficit.
            integrals[integral]=far+deficit;
            ready[integral]=true;
            sum+=entry.second*integrals[integral];
        }
        const double particles=force.getNumParticles();
        const double normalization=particles*(particles+1)/2;
        const double result=2*std::acos(-1.0)*particles*particles*(sum/normalization);
        if (!std::isfinite(result))
            throw OpenMM::OpenMMException("CustomNonbondedForce: tail coefficient is not finite");
        return result;
    }

    OpenMM::CustomNonbondedForce definition;
    Lepton::ExpressionProgram program;
    std::vector<std::vector<double>> classes;
    std::map<std::pair<int,int>,long long> interactions;
    std::vector<int> usedGlobals, usedFirstFields, usedSecondFields;
    std::vector<size_t> integralIndices;
    size_t integralCount=0;
    struct CachedCoefficient {
        std::vector<uint64_t> key;
        double value;
    };
    static constexpr size_t coefficientCacheCapacity=128;
    // Least recently used first. Owned by this immutable definition/particle
    // snapshot; prepareUpdate constructs a new State with an empty cache.
    std::vector<CachedCoefficient> coefficients;
    std::vector<double> globals;
    bool dirty=true;
    double coefficient=0;
#ifdef NVOPENMM_TEST_EXPRESSION_TAIL
    mutable TestStats stats;
#endif
};

ExpressionTailCorrection::ExpressionTailCorrection(const OpenMM::CustomNonbondedForce& force)
    : state_(std::make_unique<State>(force)) {}

ExpressionTailCorrection::~ExpressionTailCorrection() = default;

void ExpressionTailCorrection::prepareStep(const OpenMM::Context& context, bool includeEnergy) {
    const auto& force=state_->definition;
    std::vector<double> next_globals(includeEnergy ? force.getNumGlobalParameters() : 0);
    for (int i=0;i<force.getNumGlobalParameters();++i) {
        const double value=context.getParameter(force.getGlobalParameterName(i));
        if (!std::isfinite(static_cast<float>(value)))
            throw OpenMM::OpenMMException("CustomNonbondedForce globals must be finite FP32 values");
        if (includeEnergy) next_globals[i]=value;
    }
    if (!includeEnergy) return;
    // Track original Context doubles even if direct preprocessing rounded them
    // to unchanged FP32 values. Unused globals are validated but cannot alter C.
    const bool changed=state_->dirty || state_->globals.size()!=next_globals.size() ||
        std::any_of(state_->usedGlobals.begin(),state_->usedGlobals.end(),[&](int i) {
            return State::bits(state_->globals[i])!=State::bits(next_globals[i]);
        });
    if (!changed) return;
    std::vector<uint64_t> key;
    key.reserve(state_->usedGlobals.size());
    for (int i : state_->usedGlobals) key.push_back(State::bits(next_globals[i]));
    auto& cache=state_->coefficients;
    const auto found=std::find_if(cache.begin(),cache.end(),[&](const State::CachedCoefficient& entry) {
        return entry.key==key;
    });
    double coefficient;
    if (found!=cache.end()) {
        coefficient=found->value;
        std::rotate(found,found+1,cache.end());
#ifdef NVOPENMM_TEST_EXPRESSION_TAIL
        ++state_->stats.cacheHits;
#endif
    } else {
        // Integration can fail for a live lambda. Keep the previous coefficient,
        // globals and cache intact until the new result has been validated.
        coefficient=state_->evaluateCoefficient(next_globals);
        if (cache.size()==State::coefficientCacheCapacity) cache.erase(cache.begin());
        cache.push_back({std::move(key),coefficient});
    }
    state_->coefficient=coefficient;
    state_->globals.swap(next_globals);
    state_->dirty=false;
}

#ifdef NVOPENMM_TEST_EXPRESSION_TAIL
ExpressionTailCorrection::TestStats ExpressionTailCorrection::testStats() const {
    auto result=state_->stats;
    result.cacheEntries=state_->coefficients.size();
    return result;
}
#endif

void ExpressionTailCorrection::validateDefinition(const OpenMM::CustomNonbondedForce& force) const {
    const auto& previous=state_->definition;
    requireDefinition(force.getEnergyFunction()==previous.getEnergyFunction() &&
        force.getNumParticles()==previous.getNumParticles() &&
        force.getNumPerParticleParameters()==previous.getNumPerParticleParameters() &&
        force.getNumGlobalParameters()==previous.getNumGlobalParameters() &&
        force.getNumExclusions()==previous.getNumExclusions() &&
        force.getNumInteractionGroups()==previous.getNumInteractionGroups() &&
        force.getNonbondedMethod()==previous.getNonbondedMethod() &&
        force.getCutoffDistance()==previous.getCutoffDistance() &&
        force.getUseSwitchingFunction()==previous.getUseSwitchingFunction() &&
        (!force.getUseSwitchingFunction() ||
         force.getSwitchingDistance()==previous.getSwitchingDistance()) &&
        force.getUseLongRangeCorrection()==previous.getUseLongRangeCorrection());
    for (int i=0;i<force.getNumPerParticleParameters();++i)
        requireDefinition(force.getPerParticleParameterName(i)==previous.getPerParticleParameterName(i));
    for (int i=0;i<force.getNumGlobalParameters();++i)
        requireDefinition(force.getGlobalParameterName(i)==previous.getGlobalParameterName(i));
    for (int i=0;i<force.getNumExclusions();++i) {
        int first, second, oldFirst, oldSecond;
        force.getExclusionParticles(i, first, second);
        previous.getExclusionParticles(i, oldFirst, oldSecond);
        requireDefinition(first==oldFirst && second==oldSecond);
    }
    for (int i=0;i<force.getNumInteractionGroups();++i) {
        std::set<int> first, second, oldFirst, oldSecond;
        force.getInteractionGroupParameters(i, first, second);
        previous.getInteractionGroupParameters(i, oldFirst, oldSecond);
        requireDefinition(first==oldFirst && second==oldSecond);
    }
    validateValues(force);
}

std::unique_ptr<ExpressionTailCorrection> ExpressionTailCorrection::prepareUpdate(
        const OpenMM::CustomNonbondedForce& force, const OpenMM::Context& context) const {
    validateDefinition(force);
    auto next=std::make_unique<ExpressionTailCorrection>(force);
    next->prepareStep(context);
    return next;
}

double ExpressionTailCorrection::coefficient() const {
    return state_->coefficient;
}

}  // namespace NvOpenMM
