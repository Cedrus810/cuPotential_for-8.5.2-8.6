// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ExpressionTailCorrection.hpp"
#include <openmm/Context.h>
#include <openmm/CustomExternalForce.h>
#include <openmm/CustomNonbondedForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
using namespace OpenMM;
using NvOpenMM::ExpressionTailCorrection;
void close(double a,double b) {
    if (std::abs(a-b)>3e-6*(1+std::abs(b)))
        throw std::runtime_error("coefficient mismatch: "+std::to_string(a)+" vs "+std::to_string(b));
}
template<class F> void rejects(F f) {
    bool failed=false; try { f(); } catch (const std::exception&) {failed=true;}
    if (!failed) throw std::runtime_error("expected rejection");
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
uint64_t bits(double value) {
    uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
void same(double actual, double expected) {
    require(bits(actual)==bits(expected), "coefficient is not bitwise identical");
}

// A tail needs only the Context's globals. Keep the Context independent of the
// test interaction, so these tests never ask stock OpenMM to integrate its tail.
struct GlobalContext {
    System system;
    VerletIntegrator integrator{.001};
    std::unique_ptr<Context> context;
    explicit GlobalContext(const CustomNonbondedForce& force) {
        system.addParticle(1);
        auto* globals=new CustomExternalForce("0");
        for (int i=0;i<force.getNumGlobalParameters();++i)
            globals->addGlobalParameter(force.getGlobalParameterName(i),
                                        force.getGlobalParameterDefaultValue(i));
        globals->addParticle(0,{});
        system.addForce(globals);
        context=std::make_unique<Context>(system,integrator,Platform::getPlatformByName("Reference"));
    }
};

CustomNonbondedForce cacheForce() {
    CustomNonbondedForce force("-scale*gain*p1*p2/r^6");
    force.addPerParticleParameter("p");
    force.addPerParticleParameter("tag");
    force.addGlobalParameter("scale",.5);
    force.addGlobalParameter("gain",1.);
    force.addGlobalParameter("unused",1.);
    force.addParticle({1.,1.});force.addParticle({2.,2.});force.addParticle({1.,3.});
    force.setNonbondedMethod(CustomNonbondedForce::CutoffPeriodic);
    force.setCutoffDistance(1.);
    force.setUseLongRangeCorrection(true);
    force.setUseSwitchingFunction(true);
    force.setSwitchingDistance(.8);
    return force;
}

void checkCache() {
    auto force=cacheForce();
    GlobalContext globals(force);
    auto& context=*globals.context;
    auto tail=std::make_unique<ExpressionTailCorrection>(force);
    tail->prepareStep(context);
    const double original=tail->coefficient();
    require(tail->testStats().coefficientEvaluations==1, "initial tail was not integrated once");
    context.setParameter("scale",.7);tail->prepareStep(context);
    const double other=tail->coefficient();
    context.setParameter("scale",.5);tail->prepareStep(context);
    same(tail->coefficient(),original);
    require(tail->testStats().coefficientEvaluations==2 && tail->testStats().cacheHits==1,
            "revisiting a window recomputed its coefficient");
    context.setParameter("unused",2.);tail->prepareStep(context);
    require(tail->testStats().coefficientEvaluations==2, "unused global invalidated the coefficient");
    same(tail->coefficient(),original);
    // Distinct Context doubles are distinct keys even when FP32 evaluation
    // rounds them to the same value. This preserves the original update contract.
    context.setParameter("scale",std::nextafter(.5,1.));tail->prepareStep(context);
    require(tail->testStats().coefficientEvaluations==3, "cache rounded its global key to FP32");
    same(tail->coefficient(),original);
    context.setParameter("scale",.5);context.setParameter("gain",2.);tail->prepareStep(context);
    require(tail->testStats().coefficientEvaluations==4, "cache ignored a second referenced global");
    context.setParameter("gain",1.);tail->prepareStep(context);same(tail->coefficient(),original);
    // A cached C remains valid when volume changes; the caller applies C/V.
    context.setPeriodicBoxVectors(Vec3(5,0,0),Vec3(0,5,0),Vec3(0,0,5));
    tail->prepareStep(context);same(tail->coefficient(),original);
    require(tail->testStats().coefficientEvaluations==4, "box update reintegrated C");
    context.setParameter("unused",std::numeric_limits<double>::infinity());
    rejects([&]{tail->prepareStep(context);});same(tail->coefficient(),original);
    context.setParameter("unused",2.);tail->prepareStep(context);same(tail->coefficient(),original);
    require(tail->testStats().coefficientEvaluations==4, "invalid unused global damaged the cache");

    force.setParticleParameters(0,{3.,1.});
    auto updated=tail->prepareUpdate(force,context);
    require(bits(updated->coefficient())!=bits(original), "particle update reused a stale coefficient");
    require(updated->testStats().coefficientEvaluations==1 && updated->testStats().cacheEntries==1,
            "particle update retained the previous window cache");
    context.setParameter("scale",.7);updated->prepareStep(context);
    require(bits(updated->coefficient())!=bits(other), "particle update retained another stale window");
    tail->prepareStep(context);same(tail->coefficient(),other);
    require(tail->testStats().coefficientEvaluations==4, "preparing an update mutated the old cache");

    // A complete 128-window sweep must remain cached on subsequent sweeps.
    // Then touch its oldest entry and exceed capacity: the touched window must
    // survive while the next oldest is evicted.
    ExpressionTailCorrection bounded(force);
    constexpr int windows=128;
    for (int i=0;i<windows;++i) {
        context.setParameter("scale",double(i)/windows);bounded.prepareStep(context);
    }
    require(bounded.testStats().coefficientEvaluations==windows && bounded.testStats().cacheEntries==windows,
            "cache did not retain 128 distinct windows");
    for (int i=0;i<windows;++i) {
        context.setParameter("scale",double(i)/windows);bounded.prepareStep(context);
    }
    require(bounded.testStats().coefficientEvaluations==windows && bounded.testStats().cacheHits==windows,
            "repeated 128-window sweep recomputed coefficients");
    context.setParameter("scale",0.);bounded.prepareStep(context);
    context.setParameter("scale",1.);bounded.prepareStep(context);
    require(bounded.testStats().coefficientEvaluations==windows+1 && bounded.testStats().cacheEntries==windows,
            "cache capacity is not bounded");
    context.setParameter("scale",0.);bounded.prepareStep(context);
    context.setParameter("scale",2./windows);bounded.prepareStep(context);
    require(bounded.testStats().coefficientEvaluations==windows+1, "cache evicted a recently used window");
    context.setParameter("scale",1./windows);bounded.prepareStep(context);
    require(bounded.testStats().coefficientEvaluations==windows+2 && bounded.testStats().cacheEntries==windows,
            "least recently used window was not evicted");
    ExpressionTailCorrection signedZero(force);
    context.setParameter("scale",+0.);signedZero.prepareStep(context);
    context.setParameter("scale",-0.);signedZero.prepareStep(context);
    require(signedZero.testStats().coefficientEvaluations==2 && signedZero.testStats().cacheEntries==2,
            "cache merged positive and negative zero globals");
    context.setParameter("scale",+0.);signedZero.prepareStep(context);
    require(signedZero.testStats().coefficientEvaluations==2 && signedZero.testStats().cacheHits==1,
            "cache did not retain the original positive-zero key");
    std::cout<<"passed cache\n";
}

void checkIntegralReuse() {
    for (bool switching : {false,true}) {
        auto force=cacheForce();
        // Deliberately asymmetric to make accidentally swapping particle roles
        // observable in this private helper's literal-expression contract.
        force.setEnergyFunction("-scale*(p1+3*p2)/r^6");
        force.setParticleParameters(0,{1.,1.});
        force.setParticleParameters(1,{2.,2.});
        force.setParticleParameters(2,{1.,3.});
        force.addParticle({2.,4.});
        force.setUseSwitchingFunction(switching);
        // Distinct complete records retain their original class order/counts.
        // Keeping tag in an always-true predicate disables integral reuse in the
        // reference without changing any evaluated pair value or quadrature.
        CustomNonbondedForce reference(force);
        reference.setEnergyFunction("select(tag1*tag2,-scale*(p1+3*p2)/r^6,0)");
        GlobalContext globals(force);
        ExpressionTailCorrection reused(force),unshared(reference);
        reused.prepareStep(*globals.context);unshared.prepareStep(*globals.context);
        same(reused.coefficient(),unshared.coefficient());
        require(reused.testStats().radialIntegrals==4,
                "integrals were not reused by referenced fields with particle roles preserved");
        require(unshared.testStats().radialIntegrals==10,
                "reference expression did not retain every complete-record pair");
        globals.context->setParameter("scale",.37);
        reused.prepareStep(*globals.context);unshared.prepareStep(*globals.context);
        same(reused.coefficient(),unshared.coefficient());
    }
    std::cout<<"passed integrals\n";
}

void checkForceOnly() {
    auto force=cacheForce();
    force.setEnergyFunction("scale/r^2-0.03*p1*p2/r^6");
    force.setGlobalParameterDefaultValue(0,0.);
    GlobalContext globals(force);
    auto& context=*globals.context;
    ExpressionTailCorrection tail(force);
    tail.prepareStep(context,false);
    require(tail.testStats().coefficientEvaluations==0 && tail.testStats().cacheEntries==0,
            "initial force-only request integrated a tail");
    tail.prepareStep(context);
    const double original=tail.coefficient();
    context.setParameter("scale",1.);tail.prepareStep(context,false);
    same(tail.coefficient(),original);
    require(tail.testStats().coefficientEvaluations==1 && tail.testStats().cacheEntries==1,
            "changed force-only lambda integrated or published a tail");
    context.setParameter("unused",std::numeric_limits<double>::infinity());
    rejects([&]{tail.prepareStep(context,false);});same(tail.coefficient(),original);
    context.setParameter("unused",1.);
    // A divergent new lambda is diagnosed on the first energy request, without
    // discarding a valid coefficient or caching the failed integral.
    rejects([&]{tail.prepareStep(context);});same(tail.coefficient(),original);
    const auto afterFailure=tail.testStats();
    require(afterFailure.cacheEntries==1, "failed integration was cached");
    context.setParameter("scale",0.);tail.prepareStep(context);
    same(tail.coefficient(),original);
    require(tail.testStats().coefficientEvaluations==afterFailure.coefficientEvaluations,
            "failed energy update discarded the valid window");
    std::cout<<"passed force-only\n";
}

void checkStock() {
    auto& platform=Platform::getPlatformByName("Reference");
    for (int groups=0;groups<3;++groups) for (bool switching : {false,true}) {
        CustomNonbondedForce force("-0.003*lambda*p1*p2/r^6");
        force.addPerParticleParameter("p");force.addGlobalParameter("lambda",.5);
        for (double p : {1.,1.,2.,3.,3.,9.}) force.addParticle({p});
        force.setNonbondedMethod(CustomNonbondedForce::CutoffPeriodic);
        force.setCutoffDistance(1.);force.setUseLongRangeCorrection(true);
        force.setUseSwitchingFunction(switching);force.setSwitchingDistance(.8);
        force.addExclusion(0,1);
        if (groups) force.addInteractionGroup({0,1,2},{1,2,3,4});
        if (groups==2) {
            force.addInteractionGroup({0,1,2},{1,2,3,4});
            force.addInteractionGroup({2,4},{0,3});
        }
        System withTail,withoutTail;
        for(int i=0;i<6;++i) {withTail.addParticle(12);withoutTail.addParticle(12);}
        withTail.setDefaultPeriodicBoxVectors(Vec3(4,0,0),Vec3(0,4,0),Vec3(0,0,4));
        withoutTail.setDefaultPeriodicBoxVectors(Vec3(4,0,0),Vec3(0,4,0),Vec3(0,0,4));
        auto* full=new CustomNonbondedForce(force);
        auto* truncated=new CustomNonbondedForce(force);truncated->setUseLongRangeCorrection(false);
        withTail.addForce(full);withoutTail.addForce(truncated);
        VerletIntegrator i1(.001),i2(.001);
        Context c1(withTail,i1,platform),c2(withoutTail,i2,platform);
        std::vector<Vec3> xyz;
        for(int i=0;i<6;++i) xyz.emplace_back(.2+i*.3,.3+(i%2)*.2,.5+(i%3)*.1);
        c1.setPositions(xyz);c2.setPositions(xyz);
        auto tail=std::make_unique<ExpressionTailCorrection>(force);
        for(double lambda : {.5,.8,0.}) {
            c1.setParameter("lambda",lambda);c2.setParameter("lambda",lambda);tail->prepareStep(c1);
            const double stock=64*(c1.getState(State::Energy).getPotentialEnergy()-c2.getState(State::Energy).getPotentialEnergy());
            close(tail->coefficient(),stock);
        }
        force.setParticleParameters(2,{2.7});full->setParticleParameters(2,{2.7});truncated->setParticleParameters(2,{2.7});
        full->updateParametersInContext(c1);truncated->updateParametersInContext(c2);
        tail=tail->prepareUpdate(force,c1);
        c1.setParameter("lambda",.6);c2.setParameter("lambda",.6);tail->prepareStep(c1);
        close(tail->coefficient(),64*(c1.getState(State::Energy).getPotentialEnergy()-c2.getState(State::Energy).getPotentialEnergy()));
        const double prior=tail->coefficient();
        // Defaults affect new Contexts; a live Context retains its parameter.
        force.setGlobalParameterDefaultValue(0,.9);
        tail=tail->prepareUpdate(force,c1);
        close(tail->coefficient(),prior);
        // An inactive switch distance cannot alter the Hamiltonian or its tail.
        // An active distance remains immutable, including through failed updates.
        force.setSwitchingDistance(.7);
        if (switching) {
            rejects([&]{tail=tail->prepareUpdate(force,c1);});
            close(tail->coefficient(),prior);
            tail->prepareStep(c1);close(tail->coefficient(),prior);
            force.setSwitchingDistance(.8);
        } else {
            tail=tail->prepareUpdate(force,c1);
            close(tail->coefficient(),prior);
        }
        c1.setParameter("lambda",std::numeric_limits<double>::quiet_NaN());
        rejects([&]{tail->prepareStep(c1);});close(tail->coefficient(),prior);
        c1.setParameter("lambda",.6);
        force.setParticleParameters(2,{std::numeric_limits<double>::quiet_NaN()});
        rejects([&]{tail=tail->prepareUpdate(force,c1);});close(tail->coefficient(),prior);
        // Finite inputs may still make quadrature overflow. Candidate preparation
        // must preserve the active coefficient even when integration itself fails.
        force.setParticleParameters(0,{std::numeric_limits<float>::max()});
        force.setParticleParameters(2,{std::numeric_limits<float>::max()});
        rejects([&]{tail=tail->prepareUpdate(force,c1);});close(tail->coefficient(),prior);
        tail->prepareStep(c1);close(tail->coefficient(),prior);
        force.setParticleParameters(0,{1.});
        force.setParticleParameters(2,{2.7});force.setCutoffDistance(1.1);
        rejects([&]{tail->prepareUpdate(force,c1);});
        CustomNonbondedForce empty("1");empty.addGlobalParameter("lambda",.5);
        empty.setNonbondedMethod(CustomNonbondedForce::CutoffPeriodic);empty.setUseLongRangeCorrection(true);
        ExpressionTailCorrection zero(empty);zero.prepareStep(c1);close(zero.coefficient(),0.);
        CustomNonbondedForce noPairs("1");noPairs.addParticle({});noPairs.addInteractionGroup({}, {0});
        noPairs.setNonbondedMethod(CustomNonbondedForce::CutoffPeriodic);noPairs.setUseLongRangeCorrection(true);
        ExpressionTailCorrection nothing(noPairs);nothing.prepareStep(c1);close(nothing.coefficient(),0.);
        std::cout<<"passed groups="<<groups<<" switching="<<switching<<"\n";
    }
}

int main(int argc,char** argv) {
    try {
        const std::string test=argc==2 ? argv[1] : "stock";
        if (test=="stock") checkStock();
        else if (test=="cache") checkCache();
        else if (test=="integrals") checkIntegralReuse();
        else if (test=="force-only") checkForceOnly();
        else throw std::runtime_error("unknown test: "+test);
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<"\n";
        return 1;
    }
}
