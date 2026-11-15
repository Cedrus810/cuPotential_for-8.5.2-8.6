// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nvopenmm/NonbondedForce.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/serialization/SerializationProxy.h"
#include <memory>

using OpenMM::SerializationNode;

namespace NvOpenMM {
namespace {

void serializeInteraction(SerializationNode& node, const CutoffInteraction& interaction) {
    node.setIntProperty("evaluator", interaction.evaluator)
        .setIntProperty("combining", interaction.combiningRule)
        .setIntProperty("switch", interaction.useSwitching)
        .setDoubleProperty("switchDistance", interaction.switchingDistance)
        .setIntProperty("tail", interaction.tail)
        .setStringProperty("expression", interaction.expression);
}

CutoffInteraction deserializeInteraction(const SerializationNode& node) {
    const int evaluator = node.getIntProperty("evaluator");
    const int combining = node.getIntProperty("combining");
    const int tail = node.getIntProperty("tail");
    if (evaluator < 0 || evaluator > 1 || combining < 0 || combining > 1 || tail < 0 || tail > 1)
        throw OpenMM::OpenMMException("Invalid cuPO interaction enum in serialized force");
    CutoffInteraction interaction;
    interaction.evaluator = static_cast<CutoffInteraction::Evaluator>(evaluator);
    interaction.combiningRule = static_cast<CutoffInteraction::CombiningRule>(combining);
    interaction.useSwitching = node.getIntProperty("switch");
    interaction.switchingDistance = node.getDoubleProperty("switchDistance");
    interaction.tail = static_cast<CutoffInteraction::TailPolicy>(tail);
    interaction.expression = node.getStringProperty("expression");
    return interaction;
}

std::unique_ptr<OpenMM::CustomNonbondedForce> deserializeDefinition(const SerializationNode& node) {
    // decodeObject<T>() casts an untyped pointer; check the nested proxy before
    // letting a malformed document select an unrelated object type.
    if (node.getStringProperty("type") != "CustomNonbondedForce")
        throw OpenMM::OpenMMException("Invalid custom definition type in serialized cuPO force");
    return std::unique_ptr<OpenMM::CustomNonbondedForce>(
        node.decodeObject<OpenMM::CustomNonbondedForce>());
}

}  // namespace

class NonbondedForceProxy : public OpenMM::SerializationProxy {
public:
    NonbondedForceProxy() : SerializationProxy("NvNonbondedForce") {}

    void serialize(const void* object, SerializationNode& node) const override {
        const auto& force = *static_cast<const NonbondedForce*>(object);
        node.setIntProperty("version", 2)
            .setIntProperty("mode", force.pairMode)
            .setDoubleProperty("cutoff", force.getCutoffDistance())
            .setDoubleProperty("boxSafety", force.boxSafetyFactor)
            .setStringProperty("name", force.getName())
            .setIntProperty("group", force.getForceGroup())
            .setIntProperty("followParameters", force.dispersionCorrectionFollowsParameters)
            .setIntProperty("exceptionsUsePeriodicBoundaryConditions", force.exceptionsUsePeriodicBoundaryConditions);
        auto& latticeSums = node.createChildNode("LatticeSums");
        for (const auto& interaction : force.latticeSums) {
            latticeSums.createChildNode("LatticeSum")
                .setIntProperty("power", interaction.kernel.power)
                .setDoubleProperty("coupling", interaction.kernel.coupling)
                .setStringProperty("weightExpression", interaction.weightExpression)
                .setIntProperty("splitting", interaction.splitting);
        }
        serializeInteraction(node.createChildNode("Cutoff"), force.cutoffInteraction);
        serializeInteraction(node.createChildNode("ReferenceLJ"), force.referenceLJInteraction);

        auto& particles = node.createChildNode("Particles");
        for (const auto& particle : force.particles) {
            particles.createChildNode("Particle")
                .setDoubleProperty("q", particle.charge)
                .setDoubleProperty("s", particle.sigma)
                .setDoubleProperty("e", particle.epsilon);
        }
        auto& exceptions = node.createChildNode("Exceptions");
        for (const auto& exception : force.exceptions) {
            exceptions.createChildNode("Exception")
                .setIntProperty("a", exception.particle1)
                .setIntProperty("b", exception.particle2)
                .setDoubleProperty("q", exception.chargeProd)
                .setDoubleProperty("s", exception.sigma)
                .setDoubleProperty("e", exception.epsilon);
        }
        auto& globals = node.createChildNode("Globals");
        for (const auto& parameter : force.globalParameters) {
            globals.createChildNode("Global")
                .setStringProperty("name", parameter.name)
                .setDoubleProperty("value", parameter.defaultValue);
        }
        auto& particleOffsets = node.createChildNode("ParticleOffsets");
        for (const auto& offset : force.particleOffsets) {
            particleOffsets.createChildNode("Offset")
                .setIntProperty("g", offset.parameter)
                .setIntProperty("index", offset.particle)
                .setDoubleProperty("q", offset.chargeScale)
                .setDoubleProperty("s", offset.sigmaScale)
                .setDoubleProperty("e", offset.epsilonScale);
        }
        auto& exceptionOffsets = node.createChildNode("ExceptionOffsets");
        for (const auto& offset : force.exceptionOffsets) {
            exceptionOffsets.createChildNode("Offset")
                .setIntProperty("g", offset.parameter)
                .setIntProperty("index", offset.exception)
                .setDoubleProperty("q", offset.chargeProdScale)
                .setDoubleProperty("s", offset.sigmaScale)
                .setDoubleProperty("e", offset.epsilonScale);
        }
        if (force.customDefinition)
            node.createChildNode("CustomDefinition", force.customDefinition.get());
        if (force.cutoffExpressionDefinition)
            node.createChildNode("CutoffExpression", force.cutoffExpressionDefinition.get());
        auto& tails = node.createChildNode("Tails");
        for (const auto& tail : force.tailDefinitions)
            tails.createChildNode("Tail", tail.get());
    }

    void* deserialize(const SerializationNode& node) const override {
        const int version = node.getIntProperty("version");
        if (version != 2)
            throw OpenMM::OpenMMException("Unsupported cuPO force serialization version");
        const int mode = node.getIntProperty("mode");
        if (mode != NonbondedForce::Standard && mode != NonbondedForce::Custom)
            throw OpenMM::OpenMMException("Invalid serialized cuPO force mode");

        // Keep ownership until every nested definition has decoded successfully.
        std::unique_ptr<NonbondedForce> force;
        if (mode == NonbondedForce::Custom) {
            auto definition = deserializeDefinition(node.getChildNode("CustomDefinition"));
            force.reset(new NonbondedForce(*definition, node.getDoubleProperty("boxSafety")));
        } else {
            force.reset(new NonbondedForce(NonbondedForce::PME,
                                          node.getDoubleProperty("cutoff"), node.getDoubleProperty("boxSafety")));
        }
        force->latticeSums.clear();
        const int periodic = node.getIntProperty("exceptionsUsePeriodicBoundaryConditions");
        if (periodic != 0 && periodic != 1)
            throw OpenMM::OpenMMException("Invalid serialized cuPO exception periodicity");
        force->exceptionsUsePeriodicBoundaryConditions = periodic;
        for (const auto& interaction : node.getChildNode("LatticeSums").getChildren()) {
            const int splitting = interaction.getIntProperty("splitting");
            const auto& weight = interaction.getStringProperty("weightExpression");
            if (splitting < 0 || splitting > 1 || weight.find_first_not_of(" \t\r\n") == std::string::npos)
                throw OpenMM::OpenMMException("Invalid serialized cuPO lattice interaction");
            const RadialKernel kernel(interaction.getIntProperty("power"),
                                      interaction.getDoubleProperty("coupling"));
            for (const auto& existing : force->latticeSums)
                if (existing.kernel.power == kernel.power)
                    throw OpenMM::OpenMMException("Duplicate lattice power in serialized cuPO force");
            // Restore definitions directly: addLatticeSumInteraction applies
            // convenience policies that can overwrite saved switching/tails.
            force->latticeSums.emplace_back(kernel, weight,
                static_cast<LatticeSumInteraction::SplittingMethod>(splitting));
        }
        force->setName(node.getStringProperty("name"));
        force->setForceGroup(node.getIntProperty("group"));
        force->dispersionCorrectionFollowsParameters = node.getIntProperty("followParameters");
        force->cutoffInteraction = deserializeInteraction(node.getChildNode("Cutoff"));
        force->referenceLJInteraction = deserializeInteraction(node.getChildNode("ReferenceLJ"));
        for (const auto& particle : node.getChildNode("Particles").getChildren())
            force->particles.emplace_back(particle.getDoubleProperty("q"), particle.getDoubleProperty("s"),
                                          particle.getDoubleProperty("e"));
        for (const auto& exception : node.getChildNode("Exceptions").getChildren())
            force->exceptions.push_back({exception.getIntProperty("a"), exception.getIntProperty("b"),
                exception.getDoubleProperty("q"), exception.getDoubleProperty("s"), exception.getDoubleProperty("e")});
        force->globalParameters.clear();
        for (const auto& parameter : node.getChildNode("Globals").getChildren())
            force->globalParameters.push_back({parameter.getStringProperty("name"), parameter.getDoubleProperty("value")});
        for (const auto& offset : node.getChildNode("ParticleOffsets").getChildren()) {
            const int parameter = offset.getIntProperty("g"), index = offset.getIntProperty("index");
            if (parameter < 0 || parameter >= (int)force->globalParameters.size() ||
                    index < 0 || index >= (int)force->particles.size())
                throw OpenMM::OpenMMException("Invalid particle offset index in serialized cuPO force");
            force->particleOffsets.push_back({parameter, index,
                offset.getDoubleProperty("q"), offset.getDoubleProperty("s"), offset.getDoubleProperty("e")});
        }
        for (const auto& offset : node.getChildNode("ExceptionOffsets").getChildren()) {
            const int parameter = offset.getIntProperty("g"), index = offset.getIntProperty("index");
            if (parameter < 0 || parameter >= (int)force->globalParameters.size() ||
                    index < 0 || index >= (int)force->exceptions.size())
                throw OpenMM::OpenMMException("Invalid exception offset index in serialized cuPO force");
            force->exceptionOffsets.push_back({parameter, index,
                offset.getDoubleProperty("q"), offset.getDoubleProperty("s"), offset.getDoubleProperty("e")});
        }
        for (const auto& child : node.getChildren())
            if (child.getName() == "CutoffExpression")
                force->cutoffExpressionDefinition = deserializeDefinition(child);
        for (const auto& child : node.getChildNode("Tails").getChildren())
            force->tailDefinitions.push_back(deserializeDefinition(child));
        return force.release();
    }
};

namespace {
const bool registered = [] {
    OpenMM::SerializationProxy::registerProxy(typeid(NonbondedForce), new NonbondedForceProxy());
    return true;
}();
}  // namespace
}  // namespace NvOpenMM
