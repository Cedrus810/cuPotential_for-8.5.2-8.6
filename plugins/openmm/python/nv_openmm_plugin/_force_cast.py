# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore plugin types at OpenMM's generic SWIG return boundaries."""

from functools import wraps


_WRAPPED = "_cupo_force_cast_wrapped"


def install(mm, nvplugin):
    """Teach stock generic-returning wrappers about plugin forces and integrators.

    OpenMM's generated factory knows its built-in subclasses, but an independently
    compiled plugin arrives as a generic ``Force`` or ``Integrator``. Preserve its Python
    identity and SWIG ownership while replacing that generic pointer with a typed
    view. System and CompoundIntegrator references remain borrowed. Context
    returns its original integrator object with ownership unchanged.
    """
    def promote(force):
        if type(force) is mm.Force:
            classes = (nvplugin.NonbondedForce, nvplugin.MonteCarloBarostat)
        elif type(force) is mm.Integrator:
            classes = (nvplugin.LangevinMiddleIntegrator,)
        else:
            return force
        typed = None
        for cls in classes:
            typed = cls.cast(force)
            if typed is not None:
                break
        if typed is None:
            return force

        owns = force.thisown
        force.thisown = False
        typed.thisown = owns
        # Both proxies now share one typed SwigPyObject. When the temporary proxy
        # dies, `force` retains that pointer and its sole ownership flag. No owner
        # attributes or parent references are added to the Python force object.
        force.this = typed.this
        force.__class__ = cls
        return force

    def wrap(original):
        @wraps(original)
        def typed_result(*args, **kwargs):
            return promote(original(*args, **kwargs))

        setattr(typed_result, _WRAPPED, True)
        return typed_result

    for owner, name, is_static in (
        (mm.System, "getForce", False),
        (mm.Context, "getIntegrator", False),
        (mm.CompoundIntegrator, "getIntegrator", False),
        (mm.XmlSerializer, "deserialize", True),
        (mm.XmlSerializer, "clone", True),
    ):
        original = getattr(owner, name)
        if getattr(original, _WRAPPED, False):
            continue
        replacement = wrap(original)
        setattr(owner, name, staticmethod(replacement) if is_static else replacement)
