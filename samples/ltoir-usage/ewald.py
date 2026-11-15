# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ewald reciprocal-space Green's functions (Fourier kernels) and self-energy.

Backend-agnostic kernel cores plus their per-splitting builders: the Ewald 1/r
(p=1) kernel and its volume-baked / volume-differentiable variants, the Ewald
1/r⁶ (p=6, dispersion) kernel, and the Ewald self-energy correction.  The
reciprocal-energy convention (DC handling, doubling) is matched to the rest of
the reference layer.

These are correctness references, mostly NumPy (scipy.special.erfc for the p=6
forms).  Each kernel/self term is BARE (no coulomb / prefactor / V) and
value-returning — ``name(params…, k2)`` returns the value directly, mirroring
the C++ which returns the value; the reciprocal-energy assembly (×±prefactor/V,
or ×coulomb·Σqᵢ² for the self term) happens at the call site.
:func:`ewald_p1_fourier_kernel_value` takes an optional ``xp`` backend so the one
volume-differentiable autodiff path (the reciprocal stress) can evaluate it on
``jax.numpy`` arrays; everything else is NumPy.
"""
from __future__ import annotations

import math

import numpy as np


# ── Ewald 1/r (p=1) — reciprocal Fourier kernel (jax.numpy) ───────────────

def ewald_p1_fourier_kernel_value(alpha: float, k2):
    """BARE Ewald 1/r reciprocal Green's function value (jax.numpy); returns the value.

    ``K(k²) = (1/π)·exp(-π²·k²/α²)/k²`` with ``K(0)=0`` (charge-neutrality;
    v_hat(0) is divergent for p=1).  Mirrors cupo's CUDA
    EwaldFourierKernelValue<Real, 1> with the ``coulomb/V`` outer factor stripped
    — assemble at the call site with ``×(coulomb/V)``.

    ``jax.numpy`` throughout (so ``jax.grad`` differentiates it directly — see
    :func:`ewald_p1_fourier_kernel_grad` — and the volume-differentiable stress
    path works); the double-``where`` keeps the autodiff gradient finite at k²=0.
    NumPy/cupy consumers wrap with ``np.asarray``.  Requires jax x64 for fp64
    (conftest enables it for tests; scripts set ``JAX_ENABLE_X64``).
    """
    import jax.numpy as jnp
    g = math.pi ** 2 / alpha ** 2
    k2 = jnp.asarray(k2)
    safe = jnp.where(k2 == 0, 1.0, k2)
    return jnp.where(k2 == 0, 0.0, jnp.exp(-g * safe) / (math.pi * safe))


def ewald_p1_fourier_kernel_grad(alpha: float, k2):
    """BARE Ewald 1/r reciprocal kernel k²-derivative ``dK/dk²`` via autodiff; returns the value.

    ``jax.grad`` of :func:`ewald_p1_fourier_kernel_value` w.r.t. k² (the value's
    DC double-where keeps the autodiff finite and 0 at k²=0).  Assemble with
    ``×(coulomb/V)``.  Used by the reciprocal-space virial tensor.  Returns NumPy.
    """
    import jax
    import jax.numpy as jnp
    k2j = jnp.asarray(k2)
    g = jax.grad(lambda x: jnp.sum(ewald_p1_fourier_kernel_value(alpha, x)))(k2j)
    return np.asarray(g)


def ewald_p6_fourier_kernel_value(alpha: float, k2):
    """BARE Ewald 1/r⁶ (dispersion) reciprocal Green's function value (jax.numpy); returns the value.

        K(k²) = (π^{3/2}/2) · R_6(k²)
        R_6   = (4·α³·E - 4π²·k²·R_4) / 6
        R_4   = 2α·E - 2·π^{3/2}·√(k²)·erfc(π·√(k²)/α)
        E     = exp(-π²·k²/α²)

    BARE: the leading ``-prefactor/V`` is stripped — assemble at the call site
    with ``×(-prefactor/V)`` (the minus carries the attractive-dispersion sign;
    dispersion is the FT of a negative spatial function).  Finite and KEPT at
    k²=0 (no DC zeroing for p≥4): R_6(0)=2α³/3.  Mirrors cupo's CUDA
    EwaldFourierKernelValue<Real, 6> sans the outer factor.  ``jax.numpy``
    throughout (autodiff-ready — see :func:`ewald_p6_fourier_kernel_grad`).
    """
    import jax.numpy as jnp
    from jax.scipy.special import erfc
    pi32 = np.pi ** 1.5
    k2 = jnp.asarray(k2)
    k = jnp.sqrt(k2)
    t = np.pi * k / alpha
    E = jnp.exp(-t * t)
    C = erfc(t)
    R4 = 2.0 * alpha * E - 2.0 * pi32 * k * C
    R6 = (4.0 * alpha ** 3 * E - 4.0 * np.pi ** 2 * k2 * R4) / 6.0
    return 0.5 * pi32 * R6


def ewald_p6_fourier_kernel_grad(alpha: float, k2):
    """BARE Ewald 1/r⁶ reciprocal kernel k²-derivative ``dK/dk²`` via autodiff; returns the value.

    ``jax.grad`` of :func:`ewald_p6_fourier_kernel_value` w.r.t. k².  The value's
    bare ``√k²`` makes the autodiff singular only at the k²=0 DC bin, where the
    grad is masked to 0 — harmless because the virial weights it by ``k_a·k_b=0``
    there (the finite analytic DC value never contributes).  Assemble with
    ``×(-prefactor/V)``.  Returns NumPy.
    """
    import jax
    import jax.numpy as jnp
    k2j = jnp.asarray(k2)
    g = jax.grad(lambda x: jnp.sum(ewald_p6_fourier_kernel_value(alpha, x)))(k2j)
    g = jnp.where(k2j == 0, 0.0, g)
    return np.asarray(g)


def ewald_p1_self_value(alpha: float) -> float:
    """BARE Ewald p=1 self-energy correction value ``-α/√π``; returns the value.

    The grid-independent term that removes each particle's interaction with its
    own Gaussian in the Ewald decomposition.  BARE (per unit Σqᵢ²): assemble at
    the call site with the UNIFORM ``×prefactor·Σqᵢ²`` (here prefactor=coulomb;
    always a negative, attractive contribution).  Mirrors cupo's CUDA p1
    ``self_value``.
    """
    return float(-alpha / np.sqrt(np.pi))


def ewald_p6_self_value(alpha: float) -> float:
    """BARE Ewald p=6 (dispersion) self-energy correction value ``+α⁶/12``; returns the value.

    Positive (LJPME convention, E_self_disp > 0): it removes the i=j v_hat(0)
    overcount that PME's reciprocal sum implicitly adds for the finite p≥4
    dispersion kernel.  BARE (per unit Σqᵢ²): assemble at the call site with the
    UNIFORM ``×prefactor·Σqᵢ²`` (same assembly as p=1; the bare formula's
    intrinsic sign carries the physical asymmetry).  Mirrors cupo's CUDA
    EwaldSplitting<6>::self_value.
    """
    return float(alpha ** 6 / 12.0)


# ── direct-space BARE value terms (jax; autograd supplies the grads) ──────────
# These mirror the C++ EwaldP{1,6}{Short,Long}RangeValue structs exactly (BARE:
# no prefactor / qᵢqⱼ).  Each takes r² as its last positional arg and returns the
# value directly (mirrors the C++ which returns the value); the matching C++
# _grad is reproduced by jax.grad of these (d/d(r²)).  Identities:
#   p=1:  short + long ≡ 1/r          p=6:  short + long ≡ 1/r⁶

def ewald_p1_short_range_value(alpha: float, r2):
    """BARE Ewald p=1 short-range value ``S(r²) = erfc(α·r)/r`` (jnp); returns the value.

    Bind the splitting params with functools.partial(ewald_p1_short_range_value, alpha)
    for a 1-arg f(r²) (e.g. for jax.grad).
    """
    import jax.numpy as jnp
    from jax.scipy.special import erfc

    r2 = jnp.asarray(r2)
    r = jnp.sqrt(r2)
    return erfc(alpha * r) / r


def ewald_p1_long_range_value(alpha: float, r2):
    """BARE Ewald p=1 long-range value ``L(r²) = erf(α·r)/r`` (jnp); returns the value.

    erf is evaluated independently (not 1−erfc) so small-αr keeps full precision.

    Bind the splitting params with functools.partial(ewald_p1_long_range_value, alpha)
    for a 1-arg f(r²) (e.g. for jax.grad).
    """
    import jax.numpy as jnp
    from jax.scipy.special import erf

    r2 = jnp.asarray(r2)
    r = jnp.sqrt(r2)
    return erf(alpha * r) / r


def ewald_p6_short_range_value(alpha: float, r2):
    """BARE Ewald p=6 short-range value ``S(r²) = g₆(α r)/r⁶`` (jnp); returns the value.
    ``g₆(x) = exp(-x²)·(1 + x² + x⁴/2)``.

    Bind the splitting params with functools.partial(ewald_p6_short_range_value, alpha)
    for a 1-arg f(r²) (e.g. for jax.grad).
    """
    import jax.numpy as jnp
    a2 = alpha * alpha

    r2 = jnp.asarray(r2)
    y = a2 * r2                       # (α r)²
    g6 = jnp.exp(-y) * (1.0 + y + 0.5 * y * y)
    return g6 / (r2 ** 3)


def ewald_p6_long_range_value(alpha: float, r2):
    """BARE Ewald p=6 long-range value ``L(r²) = (1 − g₆(α r))/r⁶`` (jnp); returns the value.

    Uses the small-y Taylor of ``(1−g₆)/r⁶ = α⁶·(1/6 − y/8 + y²/20 − …)`` for
    ``y = (α r)² < 2`` to avoid cancellation in both value and derivative.
    The coefficients follow the integral ``1/2 ∫₀¹ t² exp(-yt) dt``:
    ``c_n=(-1)^n/[2*n!*(n+3)]``. Degree 24 also resolves FP64.
    The direct form is used elsewhere.

    Bind the splitting params with functools.partial(ewald_p6_long_range_value, alpha)
    for a 1-arg f(r²) (e.g. for jax.grad).
    """
    import jax.numpy as jnp
    a2 = alpha * alpha
    a6 = a2 ** 3

    r2 = jnp.asarray(r2)
    y = a2 * r2
    from math import factorial
    small = y < 2.0
    # Keep both jnp.where operands finite so autodiff also has a finite r=0
    # limit; a zero multiplier cannot hide NaN from an inactive division.
    small_y = jnp.where(small, y, 0.0)
    taylor = jnp.zeros_like(y)
    for n in range(24, -1, -1):
        coefficient = (-1.0)**n / (2*factorial(n)*(n+3))
        taylor = coefficient + small_y*taylor
    taylor *= a6
    safe_r2 = jnp.where(small, 1.0, r2)
    g6 = jnp.exp(-y) * (1.0 + y + 0.5 * y * y)
    direct = (1.0 - g6) / (safe_r2 ** 3)
    return jnp.where(small, taylor, direct)
