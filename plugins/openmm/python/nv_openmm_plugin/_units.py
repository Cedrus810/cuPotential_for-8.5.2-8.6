# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Return OpenMM quantities from thermodynamic parameter getters."""

from functools import wraps


def install(nvplugin):
    from openmm import unit

    for cls, name, target in (
        (nvplugin.LangevinMiddleIntegrator, "getTemperature", unit.kelvin),
        (nvplugin.LangevinMiddleIntegrator, "getFriction", unit.picosecond**-1),
        (nvplugin.MonteCarloBarostat, "getDefaultPressure", unit.bar),
        (nvplugin.MonteCarloBarostat, "getDefaultTemperature", unit.kelvin),
    ):
        original = getattr(cls, name)
        if getattr(original, "_cupo_unit_return", False):
            continue

        def wrap(method, expected):
            @wraps(method)
            def quantity(*args, **kwargs):
                value = method(*args, **kwargs)
                return value if unit.is_quantity(value) else value * expected

            quantity._cupo_unit_return = True
            return quantity

        setattr(cls, name, wrap(original, target))
