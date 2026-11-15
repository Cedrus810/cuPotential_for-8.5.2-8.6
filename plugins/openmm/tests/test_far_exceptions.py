# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A NonbondedForce exception may join two atoms that are nowhere near each other.

The obvious exceptions -- 1-2, 1-3, 1-4 -- are all bonded neighbours and so
always within a bond length or two. An NBFix is not. Overriding one atom-type
pair's Lennard-Jones means zeroing the standard LJ through an exception, keeping
the charges (chargeProd = q_i*q_j), and supplying the real LJ from a
CustomBondForce; and because it keys on atom TYPES rather than on bonded
neighbours, the resulting exceptions sit at arbitrary separation. In the system
this was found on, 2309 of 3024 such pairs were beyond the cutoff, out to 3 nm
in a 5 nm box.

cuda_towerplate_exclusion_cancel used to trap on any corrected exception whose
minimum-image cell delta differed from its raw one, which is every one of those
pairs as soon as the molecule diffuses past the half-box line -- so the run died,
non-deterministically, a few thousand steps in, over a difference of order 1e-13
kJ/mol. It also had the opposite failure in its other arm, silently charging a
contact interaction at whichever image the towerplate stencil happened to pick.

Direct cancellation reproduces the ordinary periodic pair geometry exactly;
reciprocal corrections use the coordinate minimum image in both arms.
Physical exception replacements independently obey the source force's
periodic-exception flag. The stock-parity tests below explicitly enable that flag
on both engines, covering NBFix-shaped and scaled exceptions in the mapped and
unmapped arms. Conversion also accepts and preserves OpenMM's default False.

Everything here runs IN-PROCESS. It used to run each case in a fresh interpreter,
with build() living inside a source string, because half of what was tested was a
device trap -- and a trap poisons the CUDA context for every later test in the
same process. Nothing traps now, so the isolation bought nothing and the string
cost a hundred lines of real logic that no linter, debugger or traceback could
reach.
"""

import math
import sys

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

from conftest import PLUGIN_PATH  # noqa: E402
sys.path[:0] = [p for p in PLUGIN_PATH if p not in sys.path]
import nv_openmm_plugin as nv  # noqa: E402

NM = unit.nanometer
EDGE, CUT = 4.0, 1.0
KJ = unit.kilojoule_per_mole


class Geom:
    """What the planted pair turned out to be, for a caller that has to predict
    what each engine should charge for it.

    Returned rather than printed. The mapped test builds the same box six times
    and differences the results, which is only meaningful if all six planted the
    SAME pair -- so the tests compare these instead of trusting that they did.
    """

    __slots__ = ("raw", "mic", "chg", "alpha")

    def __init__(self, raw, mic, chg, alpha):
        self.raw, self.mic, self.chg, self.alpha = raw, mic, chg, alpha

    def __eq__(self, other):
        return (self.raw, self.mic, self.chg, self.alpha) == \
               (other.raw, other.mic, other.chg, other.alpha)

    def __repr__(self):
        return (f"Geom(raw={self.raw:.6f}, mic={self.mic:.6f}, "
                f"chg={self.chg:.10f}, alpha={self.alpha:.6f})")


def build(scale, use_plugin=True, method="PME", mapped=False, exception=True):
    """Water box plus ONE exception between two far-apart atoms.

    scale == 1.0 reproduces the NBFix shape (chargeProd == q_i*q_j, no LJ).
    Anything else gives the exception a residual 1/r tail.

    method selects the splitting ("PME" or "Prolate"). Prolate is what the plugin
    actually ships (NonbondedForce.h: "THE DEFAULT"), so the agreement is checked
    under both, not just the one that is easy to compare against stock.

    mapped selects WHICH of the exclusion kernel's two arms the pair lands in, and
    is asserted below rather than assumed. False puts the two atoms beyond the
    cutoff under BOTH conventions. True puts them beyond the cutoff in raw
    coordinates but within it under the minimum image, so the towerplate evaluates
    the pair -- the arm that used to disagree silently.

    exception=False builds the same box with no exception at all, the baseline the
    mapped test differences against.

    Both plugin and stock builds explicitly enable periodic exceptions so that
    these parity tests compare all terms at the same minimum-image distance.

    Returns (context, integrator, Geom).
    """
    ff = app.ForceField("tip3p.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, model="tip3p", boxSize=mm.Vec3(EDGE, EDGE, EDGE) * NM)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=CUT * NM,
                             constraints=app.HBonds, rigidWater=True)
    pos = np.array(mod.positions.value_in_unit(NM))
    # Index, not the object: SWIG hands back a fresh proxy on every getForce, so
    # identity comparison never matches and the removal below has to go by position.
    nb_index = [i for i in range(system.getNumForces())
                if isinstance(system.getForce(i), mm.NonbondedForce)][0]
    nb = system.getForce(nb_index)

    oxy = [i for i in range(system.getNumParticles())
           if system.getParticleMass(i).value_in_unit(unit.dalton) > 15]
    if mapped:
        # Far apart in RAW x, close in y and z: the minimum image lands back inside
        # the cutoff, so the pair IS a towerplate neighbour.
        #
        # Choose the SMALLEST minimum image at or beyond CLASH, not the first
        # candidate that qualifies. Smallest, because what this configuration
        # measures is the short-range term stock drops, which goes like
        # erfc(alpha*r)/r -- a candidate sitting just inside the cutoff would leave
        # almost nothing to measure and make the signal depend on however
        # addSolvent packed the box. But not below CLASH: addSolvent over-packs
        # against the faces, so the very closest cross-face pair here is 0.18 nm,
        # two oxygens on top of each other -- not a configuration any real system
        # presents, and its LJ dwarfs the Coulomb term this isolates.
        #
        # Candidates are taken against the two x faces, found from the coordinates'
        # own range rather than [0, EDGE): addSolvent centres the box on the origin.
        # The minimum image is origin-free either way.
        CLASH = 0.25
        xlo, xhi = pos[oxy, 0].min(), pos[oxy, 0].max()
        near = np.array([i for i in oxy if pos[i][0] < xlo + 0.4])
        far = np.array([i for i in oxy if pos[i][0] > xhi - 0.4])
        assert len(near) and len(far), "box has no atoms against a face"
        best = None
        for a in near:
            dd = pos[far] - pos[a]
            rr = np.linalg.norm(dd - EDGE * np.round(dd / EDGE), axis=1)
            ok = np.flatnonzero(rr >= CLASH)
            if not len(ok):
                continue
            k = int(ok[np.argmin(rr[ok])])
            if best is None or rr[k] < best[0]:
                best = (float(rr[k]), int(a), int(far[k]))
        assert best is not None and best[0] < 0.7, (
            "no cross-face pair with a usable minimum image: %s" % (best,))
        lo, hi = best[1], best[2]
    else:
        # Pick two oxygens far enough apart that their cell indices differ by more
        # than half the grid on x -- exactly when the minimum image disagrees with
        # the raw delta. Necessary but not sufficient: the pair also has to miss
        # the towerplate, which is y and z doing the work and is what the assert
        # below checks.
        nc = int(np.floor(EDGE / CUT))
        cells = np.floor(pos[:, 0] / (EDGE / nc)).astype(int)
        lo = min(oxy, key=lambda i: cells[i])
        hi = max(oxy, key=lambda i: cells[i])
        assert cells[hi] - cells[lo] > nc // 2, (cells[lo], cells[hi], nc)

    d = pos[hi] - pos[lo]
    mic = d - EDGE * np.round(d / EDGE)
    # Which arm the pair lands in is the whole point of `mapped`, and neither
    # picker guarantees it by construction -- the unmapped one constrains only the
    # x cell delta, and on a 4-cell grid a delta of 3 is adjacent under the minimum
    # image, so it relies on y and z to push the pair apart. Check it here rather
    # than infer it from a test passing.
    assert (np.linalg.norm(mic) < CUT) == mapped, (
        "minimum image %.3f nm against cutoff %.3f puts this pair in the wrong arm "
        "for mapped=%r" % (np.linalg.norm(mic), CUT, mapped))

    qi = nb.getParticleParameters(lo)[0].value_in_unit(unit.elementary_charge)
    qj = nb.getParticleParameters(hi)[0].value_in_unit(unit.elementary_charge)
    # alpha comes off the stock NonbondedForce, derived the way OpenMM derives it
    # when none was set. Stock's is the only alpha a prediction needs -- the
    # plugin's half is splitting-independent.
    alpha = nb.getPMEParameters()[0].value_in_unit(NM ** -1)
    if alpha == 0.0:   # 0 means "choose from the tolerance", as OpenMM does
        alpha = float(np.sqrt(-np.log(2 * nb.getEwaldErrorTolerance())) / CUT)
    geom = Geom(float(np.linalg.norm(d)), float(np.linalg.norm(mic)),
                float(qi * qj), alpha)

    if exception:
        nb.addException(lo, hi, scale * qi * qj, 0.3, 0.0)
    nb.setExceptionsUsePeriodicBoundaryConditions(True)

    if use_plugin:
        system.addForce(nv.NonbondedForce(nb, 1.0, getattr(nv.NonbondedForce, method)))
        system.removeForce(nb_index)
    integ = mm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                     {"Precision": "mixed", "DeterministicForces": "true"})
    ctx.setPositions(pos * NM)
    return ctx, integ, geom


def energy(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_nbfix_shaped_far_exception_does_not_trap(method):
    """The regression. A full-Coulomb, no-LJ exception across half the box is an
    ordinary nonbonded pair, and must simply run -- under either splitting, since
    Prolate is the default this plugin actually ships.

    This configuration used to reach the exclusion kernel's __trap(), which takes
    the CUDA context with it; the assertion that matters is that 50 steps complete.
    """
    ctx, integ, _ = build(scale=1.0, method=method)
    try:
        integ.step(50)
        assert energy(ctx) < 0.0
    finally:
        del ctx, integ


# PME is numerically what stock OpenMM's own PME does (same alpha, same mesh --
# NonbondedForce.h), so it is held to fp32-round-off tightness. Prolate is a
# genuinely different splitting of the same 1/r sum -- ~2.5e-5 energy / ~5e-4
# force RMSE truncation error against PME, by that same header's own numbers,
# independent of anything this file is testing -- so its tolerance is wider on
# purpose. Both bound the SAME question, the far exception's minimum-image
# correction; they are not re-litigating the splittings' documented difference.
_AGREEMENT_TOL = {"PME": 1e-4, "Prolate": 2e-3}


def _plugin_vs_stock(method, scale, steps=0):
    """Both engines on the same box and the same exception, energies returned."""
    out = {}
    for tag, plug in (("plugin", True), ("stock", False)):
        ctx, integ, _ = build(scale=scale, use_plugin=plug, method=method)
        try:
            if steps:
                integ.step(steps)
            out[tag] = energy(ctx)
        finally:
            del ctx, integ
    return out["plugin"], out["stock"]


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_nbfix_shaped_far_exception_agrees_with_stock(method):
    """Not trapping is only worth having if the number is right: minimum image
    has to give what OpenMM gives, or the guard was silenced rather than fixed."""
    plugin, stock = _plugin_vs_stock(method, scale=1.0)
    rel = abs(plugin - stock) / abs(stock)
    tol = _AGREEMENT_TOL[method]
    assert rel < tol, (
        f"plugin {plugin:.4f} vs stock {stock:.4f} kJ/mol, relative {rel:.2e} "
        f"(tol {tol:.0e}) under {method} -- minimum image does not reproduce "
        "OpenMM for this pair")


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_scaled_far_exception_agrees_with_stock(method):
    """The case that used to be fatal.

    Give the same far pair a residual 1/r tail. Both forces explicitly enable
    periodic exceptions, so their physical replacements use the minimum image.
    Compare the resulting energy against stock under both splitting choices.
    """
    plugin, stock = _plugin_vs_stock(method, scale=0.5, steps=50)
    rel = abs(plugin - stock) / abs(stock)
    tol = _AGREEMENT_TOL[method]
    assert rel < tol, (
        f"plugin {plugin:.4f} vs stock {stock:.4f} kJ/mol, relative {rel:.2e} "
        f"(tol {tol:.0e}) under {method}")


def test_mapped_far_exception_closes_against_the_closed_form():
    """The mapped pair must no longer contain the historical wrong-image term.

    The pair crosses a box face: its raw separation exceeds the cutoff, while
    its minimum image is a close contact. Both forces request periodic physical
    replacements. Difference each engine against its own no-exception baseline
    to remove the fixed PME-versus-Prolate offset, then bound the remaining gap
    against the closed-form error from mixing raw and minimum-image distances.
    """
    base, geoms = {}, []
    for tag, plug in (("plugin", True), ("stock", False)):
        ctx, integ, g = build(scale=1.0, use_plugin=plug, method="Prolate",
                              mapped=True, exception=False)
        try:
            base[tag] = energy(ctx)
        finally:
            del ctx, integ
        geoms.append(g)

    gaps = {}
    for scale in (1.0, 0.0):
        d = {}
        for tag, plug in (("plugin", True), ("stock", False)):
            ctx, integ, g = build(scale=scale, use_plugin=plug,
                                  method="Prolate", mapped=True)
            try:
                d[tag] = energy(ctx) - base[tag]
            finally:
                del ctx, integ
            geoms.append(g)
        gaps[scale] = d["plugin"] - d["stock"]

    # Six builds, differenced against each other -- only meaningful if all six
    # planted the same pair. addSolvent is seeded, so they should, but "should"
    # is what this is here to stop being.
    assert all(g == geoms[0] for g in geoms), f"builds disagree on the pair: {geoms}"
    raw, mic, q, alpha = geoms[0].raw, geoms[0].mic, geoms[0].chg, geoms[0].alpha
    assert raw > 3.0 and mic < 0.7, (
        f"the planted pair is not in the mapped regime: raw {raw:.3f} nm, "
        f"minimum image {mic:.3f} nm, cutoff {CUT} nm")

    C = 138.935456  # kJ/mol * nm / e^2

    def predicted(s):
        return C * q * ((s - 1.0) / mic - s / raw
                        + math.erf(alpha * raw) / raw
                        + math.erfc(alpha * mic) / mic)

    # Guard the test's own signal. The check below is relative, so a prediction
    # near zero turns it into an absolute test far tighter than two independently
    # computed energies can meet -- a spurious failure, not a spurious pass. At
    # the geometry the picker actually returns the gap is ~135 kJ/mol, so there is
    # a wide margin; this only bites if someone raises CLASH far enough to push
    # the pair out where erfc has taken the short-range term away.
    assert abs(predicted(1.0)) > 0.2, (
        f"minimum image {mic:.3f} nm leaves only {predicted(1.0):.4f} kJ/mol to "
        "measure -- the relative check below cannot resolve that; pick a tighter "
        "pair")

    # `predicted` is the error this arm USED to make, kept rather than deleted:
    # asserting against it, instead of against zero, is what distinguishes a fix
    # from an implementation that merely moved the disagreement somewhere else.
    # Better than 1% of it left means the wrong-image term is gone, not shrunk.
    for scale, got in sorted(gaps.items()):
        want = predicted(scale)
        assert abs(got) <= 0.01 * abs(want), (
            f"scale {scale}: the two engines still differ by {got:.4f} kJ/mol. "
            f"Applying the exception at the wrong image would cost {want:.4f}, so "
            "the correction is not being evaluated at the minimum image after all.")


def test_conversion_preserves_forcefield_exception_default():
    """A normal force-field source converts without changing its distance flag."""
    ff = app.ForceField("tip3p.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, model="tip3p", boxSize=mm.Vec3(2.5, 2.5, 2.5) * unit.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    assert not nb.getExceptionsUsePeriodicBoundaryConditions()
    before = mm.XmlSerializer.serialize(nb)
    force = nv.NonbondedForce(nb)
    assert not force.getExceptionsUsePeriodicBoundaryConditions()
    assert force.getNumExceptions() == nb.getNumExceptions()
    assert mm.XmlSerializer.serialize(nb) == before
    force.setExceptionsUsePeriodicBoundaryConditions(True)
    assert force.getExceptionsUsePeriodicBoundaryConditions()
    assert mm.XmlSerializer.serialize(nb) == before
    force.setExceptionsUsePeriodicBoundaryConditions(False)
    assert not force.getExceptionsUsePeriodicBoundaryConditions()
