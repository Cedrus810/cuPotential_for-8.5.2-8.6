// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nvopenmm/LangevinMiddleIntegrator.h"
#include "nvopenmm/MonteCarloBarostat.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/serialization/SerializationProxy.h"
#include <memory>

namespace NvOpenMM {
namespace {
using OpenMM::SerializationNode;
using OpenMM::SerializationProxy;

class LangevinMiddleIntegratorProxy final : public SerializationProxy {
public:
    LangevinMiddleIntegratorProxy() : SerializationProxy("NvLangevinMiddleIntegrator") {}

    void serialize(const void* object, SerializationNode& node) const override {
        const auto& integrator = *static_cast<const LangevinMiddleIntegrator*>(object);
        node.setIntProperty("version", 1)
            .setDoubleProperty("temperature", integrator.getTemperature())
            .setDoubleProperty("friction", integrator.getFriction())
            .setDoubleProperty("stepSize", integrator.getStepSize())
            .setDoubleProperty("constraintTolerance", integrator.getConstraintTolerance())
            .setIntProperty("randomNumberSeed", integrator.getRandomNumberSeed())
            .setIntProperty("integrationForceGroups", integrator.getIntegrationForceGroups())
            .setIntProperty("comFrequency", integrator.getComFrequency());
    }

    void* deserialize(const SerializationNode& node) const override {
        if (node.getIntProperty("version") != 1)
            throw OpenMM::OpenMMException("NvLangevinMiddleIntegrator: unsupported serialization version");
        auto integrator = std::make_unique<LangevinMiddleIntegrator>(
            node.getDoubleProperty("temperature"), node.getDoubleProperty("friction"),
            node.getDoubleProperty("stepSize"), node.getIntProperty("comFrequency"));
        integrator->setConstraintTolerance(node.getDoubleProperty("constraintTolerance"));
        integrator->setRandomNumberSeed(node.getIntProperty("randomNumberSeed"));
        integrator->setIntegrationForceGroups(node.getIntProperty("integrationForceGroups"));
        return integrator.release();
    }
};

class MonteCarloBarostatProxy final : public SerializationProxy {
public:
    MonteCarloBarostatProxy() : SerializationProxy("NvMonteCarloBarostat") {}

    void serialize(const void* object, SerializationNode& node) const override {
        const auto& force = *static_cast<const MonteCarloBarostat*>(object);
        node.setIntProperty("version", 1)
            .setDoubleProperty("pressure", force.getDefaultPressure())
            .setDoubleProperty("temperature", force.getDefaultTemperature())
            .setIntProperty("frequency", force.getFrequency())
            .setIntProperty("randomNumberSeed", force.getRandomNumberSeed())
            .setIntProperty("forceGroup", force.getForceGroup())
            .setStringProperty("name", force.getName());
    }

    void* deserialize(const SerializationNode& node) const override {
        if (node.getIntProperty("version") != 1)
            throw OpenMM::OpenMMException("NvMonteCarloBarostat: unsupported serialization version");
        auto force = std::make_unique<MonteCarloBarostat>(
            node.getDoubleProperty("pressure"), node.getDoubleProperty("temperature"),
            node.getIntProperty("frequency"));
        force->setRandomNumberSeed(node.getIntProperty("randomNumberSeed"));
        force->setForceGroup(node.getIntProperty("forceGroup"));
        force->setName(node.getStringProperty("name"));
        return force.release();
    }
};

const bool registered = [] {
    SerializationProxy::registerProxy(typeid(LangevinMiddleIntegrator), new LangevinMiddleIntegratorProxy());
    SerializationProxy::registerProxy(typeid(MonteCarloBarostat), new MonteCarloBarostatProxy());
    return true;
}();
}  // namespace
}  // namespace NvOpenMM
