# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Coulomb + C6-dispersion single-point references.

The default, ``--system water_10k``, uses the bundled OpenMM-generated water
structure in ``data/water_10k/``. With ``--pdb``, a caller's PDB or PDB.gz supplies
an amber14-all + tip3pfb system (water or apo protein + ions). Custom systems must
have zero net charge: the independent JAX reference omits the charged-system
neutralizing background term.
Two independent OpenMM single-point evaluations are stored side-by-side:

  * Coulomb-only PME (charges intact, LJ ε zeroed on all particles).
  * C6-only dispersion PME (−Σ C6/r⁶, geometric combining, no R¹²).

Bonded 1-2/1-3 pairs are excluded; 1-4 (and beyond) are full-strength nonbonded
pairs (no Amber 1-4 scaling).  The exclusion list is the 1-2/1-3 bond-graph subset,
so 1-4 stay ordinary PME pairs in OpenMM, jax and cupo alike.

Both passes use the Reference platform (uniformly fp64) so the CUDA
fp64-PME grid-corruption ceiling (G ≳ 600) doesn't apply.  The blob is
consumed by ``usage.cu`` in this directory, which runs the
corresponding prolate P=1 and P=6 splittings and compares.

Output (binary, little-endian, fp64 + i32):

    +---------------------------------------------------------------+
    | u32     n_atoms                                               |
    | u32     n_excl                                                |
    | f64     box[3]                  (nm)                          |
    | f64     cutoff_nm               (used by the C++ side)        |
    | f64     coulomb_constant        (kJ·nm / mol·e², OpenMM k_e)  |
    | f64     coords[n_atoms][3]      (nm; molecules made whole)   |
    | f64     charges[n_atoms]        (e)                           |
    | f64     sigma[n_atoms]          (nm)                          |
    | f64     epsilon[n_atoms]        (kJ/mol)                      |
    | i32     exclusions[n_excl][2]   (0-indexed atom pairs)        |
    | f64     ref_E_coulomb           (kJ/mol)                      |
    | f64     ref_F_coulomb[n_atoms][3]  (kJ/mol/nm)                |
    | f64     ref_E_dispersion        (kJ/mol; C6-only, geometric)  |
    | f64     ref_F_dispersion[n_atoms][3]  (kJ/mol/nm)             |
    | f64     ref_W_coulomb[6]        (kJ/mol; virial, Voigt)       |
    | f64     ref_W_dispersion[6]     (kJ/mol; C6-only virial)      |
    +---------------------------------------------------------------+

The "dispersion" fields are now the C6-only PME dispersion (−Σ C6/r⁶, geometric
combining, no R¹²) — matching the cupo p6 + the virial; the former full-LJPME
(R¹²+R⁶) reference is gone.

The two virial tensors are analytical (jax-autodiff) reference pressure tensors
W = −∂E/∂ε under an affine box strain, Voigt order (xx, yy, zz, xy, xz, yz).  They
use an EWALD splitting with an exact reciprocal (direct k-sum, no spreading, so no
aliasing), a fixed 1.8 nm real-space cutoff (capped below half the shortest box
edge), and an extended k-range,
so the Ewald floor sits far below the prolate/c2order3 bandwidth floor that
usage.cu's cupo virial is tested against.  Coulomb is p=1 (charges); dispersion is
the C6-only p=6 term (√C₆ weights, no R¹²).

Usage (from this directory, requires OpenMM + jax):
    python gen_reference_bin.py
    python gen_reference_bin.py --system water_10k
    python gen_reference_bin.py --pdb my_system.pdb --system my_system
"""
from __future__ import annotations

import argparse
import gzip
import io
import struct
import sys
from pathlib import Path

import numpy as np
import openmm
from openmm import unit
from openmm.app import ForceField, PDBFile

import jax
jax.config.update("jax_enable_x64", True)   # fp64 — the FD self-check needs it
import jax.numpy as jnp

# ewald.py beside this file: the reference Ewald functions.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from ewald import (                                                # noqa: E402
    ewald_p1_fourier_kernel_value, ewald_p6_fourier_kernel_value,
    ewald_p1_short_range_value, ewald_p1_long_range_value,
    ewald_p6_short_range_value, ewald_p6_long_range_value,
)

DEFAULT_SYSTEM = "water_10k"
DEFAULT_CUTOFF_NM = 0.9
DEFAULT_REF_CUTOFF_NM = 2.0
_WATER_DIR = _SCRIPT_DIR.parents[1] / "data" / "water_10k"
_BUNDLED_PDBS = {"water_10k": _WATER_DIR / "solvated.pdb.gz"}


def _system_out(system: str) -> Path:
    if system == "water_10k":
        return _WATER_DIR / "water_10k_ref.bin"
    return _SCRIPT_DIR / f"{system}_ref.bin"


def _load_pdb(pdb_path: Path):
    """Read a (gzipped) PDB → (pdb, coords nm, orthorhombic box nm, n_atoms).

    This loader preserves the stored coordinates. The caller makes bonded
    molecules whole before the OpenMM evaluations and binary serialization;
    integer box shifts preserve the periodic Coulomb/C6 configuration."""
    opener = gzip.open if pdb_path.suffix == ".gz" else open
    with opener(pdb_path, "rt") as f:
        pdb = PDBFile(io.StringIO(f.read()))
    n_atoms = pdb.topology.getNumAtoms()
    vectors = pdb.topology.getPeriodicBoxVectors()
    if vectors is None:
        raise ValueError("the input PDB must specify a periodic box")
    box_vecs = np.asarray(vectors.value_in_unit(unit.nanometer), dtype=np.float64)
    if not np.allclose(box_vecs, np.diag(np.diag(box_vecs)), rtol=0.0, atol=1e-12):
        raise ValueError("this sample requires an orthorhombic PDB box")
    box = np.array([box_vecs[0][0], box_vecs[1][1], box_vecs[2][2]],
                   dtype=np.float64)
    coords = np.array(pdb.positions.value_in_unit(unit.nanometer),
                      dtype=np.float64)
    assert coords.shape == (n_atoms, 3)
    return pdb, coords, box, n_atoms


def _extract_params(nbf, n_atoms):
    """Pull per-atom (q, σ, ε) from a NonbondedForce."""
    charges = np.empty(n_atoms, dtype=np.float64)
    sigmas = np.empty(n_atoms, dtype=np.float64)
    epsilons = np.empty(n_atoms, dtype=np.float64)
    for i in range(n_atoms):
        q, s, e = nbf.getParticleParameters(i)
        charges[i] = q.value_in_unit(unit.elementary_charge)
        sigmas[i] = s.value_in_unit(unit.nanometer)
        epsilons[i] = e.value_in_unit(unit.kilojoule_per_mole)
    return charges, sigmas, epsilons


def _bond_neighbors(system, n_atoms):
    """Adjacency (list of sets) from the force-field bond graph — the system's
    HarmonicBondForce, built with constraints=None so every bond (incl. flexible
    water O-H) is present.  Ions / lone atoms have no neighbours."""
    nbrs = [set() for _ in range(n_atoms)]
    for f in system.getForces():
        if isinstance(f, openmm.HarmonicBondForce):
            for k in range(f.getNumBonds()):
                i, j, *_ = f.getBondParameters(k)
                nbrs[i].add(j)
                nbrs[j].add(i)
    return nbrs


def _exclusions_12_13(nbrs):
    """1-2 and 1-3 atom pairs from the bond graph: the pairs whose full short+long
    interaction the splitting removes (the blob's exclusion list).  1-4 and beyond
    stay ordinary full-strength nonbonded pairs.  For pure water this is exactly
    the 3 intramolecular pairs (O-H, O-H, H-H), matching the old behaviour."""
    excl = set()
    for i, ni in enumerate(nbrs):
        for j in ni:
            excl.add((min(i, j), max(i, j)))                 # 1-2
            for k in nbrs[j]:
                if k != i:
                    excl.add((min(i, k), max(i, k)))         # 1-3
    return (np.array(sorted(excl), dtype=np.int32)
            if excl else np.zeros((0, 2), dtype=np.int32))


def _make_molecules_whole(coords, box, nbrs):
    """Unwrap each bonded molecule so it is contiguous (no atom split across a
    periodic face), then wrap each molecule's centroid into [0, L).  Required so
    OpenMM's PME exclusion correction sees 1-2/1-3 pairs at their true (small)
    intramolecular distance — a torn molecule makes that correction under-subtract
    the reciprocal part and corrupts the energy.  The reference is otherwise
    wrap-invariant (jax/cupo use the integer minimum image), so this only matters
    for the OpenMM cross-check; it reproduces the old per-water-triplet wrap."""
    coords = coords.astype(np.float64).copy()
    box = np.asarray(box, np.float64)
    seen = np.zeros(len(coords), dtype=bool)
    for root in range(len(coords)):
        if seen[root]:
            continue
        comp = [root]
        seen[root] = True
        head = 0
        while head < len(comp):
            i = comp[head]; head += 1
            for j in nbrs[i]:
                if not seen[j]:
                    d = coords[j] - coords[i]
                    coords[j] = coords[i] + (d - np.round(d / box) * box)
                    seen[j] = True
                    comp.append(j)
        idx = np.asarray(comp)
        coords[idx] -= np.floor(coords[idx].mean(0) / box) * box
    return coords


def _eval(system, coords, *, label):
    """Reference-platform fp64 single-point evaluation."""
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(coords * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    E = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    F = np.array(
        state.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
        dtype=np.float64)
    nbf = next(f for f in system.getForces()
               if isinstance(f, openmm.NonbondedForce))
    a, gx, gy, gz = nbf.getPMEParametersInContext(context)
    parts = [f"α_C={float(a):.3f}  G_C=({gx},{gy},{gz})"]
    if nbf.getNonbondedMethod() == openmm.NonbondedForce.LJPME:
        a_lj, gx_lj, gy_lj, gz_lj = nbf.getLJPMEParametersInContext(context)
        parts.append(f"α_LJ={float(a_lj):.3f}  G_LJ=({gx_lj},{gy_lj},{gz_lj})")
    print(f"  [{label}] " + "  ".join(parts))
    print(f"  [{label}] E = {E:.6f} kJ/mol"
          f"   |F|_max = {np.max(np.linalg.norm(F, axis=1)):.3f}")
    return E, F


def _build_coulomb_system(box_diag, ref_cutoff, ewald_tol, charges, excl):
    """Coulomb-only PME (LJ zeroed) built directly with the 1-2/1-3 exclusion
    list, so 1-4 (and beyond) pairs are ordinary full-strength PME pairs — not
    OpenMM "exceptions" (a full-strength exception would compute bare 1/r and miss
    the pair's periodic contribution, breaking the jax cross-test)."""
    N = len(charges)
    L = np.asarray(box_diag, np.float64)
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(L[0], 0, 0) * unit.nanometer,
        openmm.Vec3(0, L[1], 0) * unit.nanometer,
        openmm.Vec3(0, 0, L[2]) * unit.nanometer)
    for _ in range(N):
        system.addParticle(1.0)
    nbf = openmm.NonbondedForce()
    nbf.setNonbondedMethod(openmm.NonbondedForce.PME)
    nbf.setCutoffDistance(ref_cutoff * unit.nanometer)
    nbf.setEwaldErrorTolerance(ewald_tol)
    nbf.setUseDispersionCorrection(False)
    sigma_unused = 1.0 * unit.nanometer
    eps_zero = 0.0 * unit.kilojoule_per_mole
    for i in range(N):
        nbf.addParticle(charges[i] * unit.elementary_charge, sigma_unused, eps_zero)
    qq_zero = 0.0 * unit.elementary_charge ** 2
    for a, b in np.asarray(excl, int):
        nbf.addException(int(a), int(b), qq_zero, sigma_unused, eps_zero)
    system.addForce(nbf)
    return system


# Force groups for the C6-only dispersion decomposition (queried selectively).
_GRP_NBF_DIRECT = 1     # NonbondedForce real-space + exceptions — IGNORED (LB)
_GRP_RECIP      = 2     # NonbondedForce LJPME reciprocal (geometric C6) — used
_GRP_REAL       = 3     # CustomNonbondedForce real-space (geometric, damped) — used
_GRP_BOND       = 4     # CustomBondForce exclusion correction — used
_DISP_GROUPS    = {_GRP_RECIP, _GRP_REAL, _GRP_BOND}

# LJPME split for the dispersion reference: α small enough that the damped
# real-space decays well inside ref_cutoff (g₆(α·rc)≈0), with a grid that
# resolves the reciprocal (the recip converges quickly — flat past a modest G).
_DISP_ALPHA_LJ = 2.0    # nm⁻¹
_DISP_GRID     = 64


def _build_dispersion_system(coords, box_diag, sigmas, epsilons, excl,
                             *, cutoff, alpha_lj, grid):
    """C6-only dispersion system with GEOMETRIC combining (for proteins).

    The total over force groups {RECIP, REAL, BOND} is exactly the geometric C6
    lattice sum −Σ_{nonbonded} C6_ij/r⁶ (C6_ij = √C6_i·√C6_j), with NO repulsive
    R¹².  Decomposition (α = `alpha_lj`, g₆(x)=e^{−x²}(1+x²+x⁴/2)):
      * CustomNonbondedForce (REAL): −C6_ij·g₆(αr)/r⁶ over non-excluded pairs in
        cutoff — the PME real-space short part with explicit geometric combining.
      * NonbondedForce LJPME (RECIP only): the geometric-C6 reciprocal complement
        −Σ_all C6_ij·[1−g₆]/r⁶ (its LB direct group is left in NBF_DIRECT, unused).
      * CustomBondForce (BOND): +C6_ij·[1−g₆]/r⁶ over the exclusion list, removing
        the bonded pairs' reciprocal contribution (protein-correct; 0 on water).
    Net: real + recip + bond = −Σ_nonbonded C6_ij/r⁶ (see gen-script docstring).
    `c6_i = √C6_i = 2√ε_i·σ_i³` is a single per-particle parameter."""
    N = len(sigmas)
    L = np.asarray(box_diag, np.float64)
    c6 = 2.0 * np.sqrt(np.maximum(epsilons, 0.0)) * sigmas ** 3   # √C₆ per atom
    a2 = float(alpha_lj) ** 2

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(L[0], 0, 0) * unit.nanometer,
        openmm.Vec3(0, L[1], 0) * unit.nanometer,
        openmm.Vec3(0, 0, L[2]) * unit.nanometer)
    for _ in range(N):
        system.addParticle(1.0)

    # NonbondedForce LJPME — used ONLY for its geometric-C6 reciprocal.
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.LJPME)
    nb.setCutoffDistance(cutoff * unit.nanometer)
    nb.setUseDispersionCorrection(False)
    nb.setLJPMEParameters(alpha_lj, grid, grid, grid)
    for i in range(N):
        nb.addParticle(0.0 * unit.elementary_charge,
                       float(sigmas[i]) * unit.nanometer,
                       float(epsilons[i]) * unit.kilojoule_per_mole)
    for a, b in np.asarray(excl, int):
        nb.addException(int(a), int(b), 0.0 * unit.elementary_charge ** 2,
                        1.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    nb.setForceGroup(_GRP_NBF_DIRECT)
    nb.setReciprocalSpaceForceGroup(_GRP_RECIP)
    system.addForce(nb)

    # CustomNonbondedForce — geometric, damped PME real-space dispersion.
    # (per-particle parameter `sc6` = √C₆ is referenced as sc61 / sc62 for the pair.)
    g6 = "exp(-a2*r^2)*(1 + a2*r^2 + 0.5*a2*a2*r^4)"
    custom = openmm.CustomNonbondedForce(f"-(sc61*sc62)*{g6}/r^6")
    custom.addPerParticleParameter("sc6")
    custom.addGlobalParameter("a2", a2)
    custom.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    custom.setCutoffDistance(cutoff * unit.nanometer)
    custom.setUseSwitchingFunction(False)
    custom.setUseLongRangeCorrection(False)
    for i in range(N):
        custom.addParticle([float(c6[i])])
    for a, b in np.asarray(excl, int):
        custom.addExclusion(int(a), int(b))
    custom.setForceGroup(_GRP_REAL)
    system.addForce(custom)

    # CustomBondForce — remove the bonded pairs' reciprocal (long-range) part.
    bond = openmm.CustomBondForce(f"c6prod*(1 - {g6})/r^6")
    bond.addPerBondParameter("c6prod")
    bond.addGlobalParameter("a2", a2)
    for a, b in np.asarray(excl, int):
        bond.addBond(int(a), int(b), [float(c6[int(a)] * c6[int(b)])])
    bond.setForceGroup(_GRP_BOND)
    system.addForce(bond)
    return system


def _eval_dispersion(system, coords, box_diag):
    """Evaluate the C6 dispersion (E, F) over the {RECIP, REAL, BOND} groups
    (NBF's LB direct group is excluded)."""
    integ = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    L = np.asarray(box_diag, np.float64)
    ctx.setPeriodicBoxVectors(openmm.Vec3(L[0], 0, 0) * unit.nanometer,
                              openmm.Vec3(0, L[1], 0) * unit.nanometer,
                              openmm.Vec3(0, 0, L[2]) * unit.nanometer)
    ctx.setPositions(coords * unit.nanometer)
    groups = sum(1 << g for g in _DISP_GROUPS)
    state = ctx.getState(getEnergy=True, getForces=True, groups=groups)
    E = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    F = np.array(state.getForces().value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer), dtype=np.float64)
    return E, F


# ════════════════════════════════════════════════════════════════════════════
#  Analytical reference virial (pressure tensor) via jax-autodiff Ewald PME.
# ════════════════════════════════════════════════════════════════════════════
#
# W = −∂E_total/∂ε at ε=0 under the affine deformation x→(I+ε)x, B→(I+ε)B
# (coords co-deform with the box, so fractional coords — and the structure
# factor — are strain-invariant).  E_total is the full Ewald PME decomposition
# with an EXACT reciprocal (direct k-sum, NO spreading) so there is no aliasing.
# Self-energy is strain-independent ⇒ 0 virial (omitted).  Voigt: (xx,yy,zz,xy,xz,yz).
#
# Reuses the jax Ewald kernels of ewald.py; the
# DC (k=0) bin is guarded so the affine-strain autodiff stays finite (∂k²/∂ε=0
# there, and the p6 √k² kernel would otherwise be 0/0).

_EWALD_ALPHA = 2.5          # nm⁻¹ split parameter — small enough that the
                            # reciprocal Gaussian exp(−π²k²/α²) decays in few
                            # k-vectors, while erfc(α·rc) stays ≪1 at rc below.
_EWALD_MMAX  = 22           # reciprocal k-vector radius
_EWALD_RC    = 1.8          # fixed real-space cutoff (nm), independent of --cutoff,
                            # capped at box/2·0.999 for the minimum image.
_CHUNK_M     = 4096         # reciprocal m-chunk for the lax.scan (the checkpoint
                            # bounds the backward-pass memory to one chunk).


def _recip_p1(alpha, k2, is_dc):
    safe = jnp.where(is_dc, 1.0, k2)
    return jnp.where(is_dc, 0.0, ewald_p1_fourier_kernel_value(alpha, safe))


def _recip_p6(alpha, k2, is_dc):
    safe = jnp.where(is_dc, 1.0, k2)
    dc = 0.5 * np.pi ** 1.5 * (2.0 * alpha ** 3 / 3.0)   # K_p6(0) = π^{3/2}α³/3
    return jnp.where(is_dc, dc, ewald_p6_fourier_kernel_value(alpha, safe))


def _pair_list(coords, box, rc, excl):
    """Non-excluded i<j pairs within rc (min-image): vectorised per atom.
    Returns (dr (P,3), pair_i (P,), pair_j (P,))."""
    N = coords.shape[0]
    partners = {}
    for a, b in np.asarray(excl, int):
        partners.setdefault(int(a), []).append(int(b))
        partners.setdefault(int(b), []).append(int(a))
    partners = {k: np.asarray(v, int) for k, v in partners.items()}
    rc2 = rc * rc
    drs, iis, jjs = [], [], []
    for i in range(N - 1):
        d = coords[i] - coords[i + 1:]
        d -= np.round(d / box) * box
        r2 = np.einsum("kc,kc->k", d, d)
        js = np.arange(i + 1, N)
        sel = r2 < rc2
        if not sel.any():
            continue
        jsel, dsel = js[sel], d[sel]
        ex_i = partners.get(i)
        if ex_i is not None and ex_i.size:
            keep = ~np.isin(jsel, ex_i)
            jsel, dsel = jsel[keep], dsel[keep]
        if jsel.size:
            drs.append(dsel)
            iis.append(np.full(jsel.size, i))
            jjs.append(jsel)
    if not drs:
        return (np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int))
    return (np.concatenate(drs), np.concatenate(iis), np.concatenate(jjs))


def _ewald_static(coords, box, excl, *, m_max, rc, chunk=_CHUNK_M):
    """Coords-independent data shared by E, forces and the virial.

    The reciprocal m-grid is padded with zero-weight rows to a whole number of
    `chunk`-sized blocks and reshaped (n_chunks, chunk, ·) for the lax.scan;
    `vc` masks the padding.
    
    Plus the real/exclusion pair lists with their integer min-image vectors
    (fixed at the reference config so the energy is differentiable in coords and box)."""
    box = np.asarray(box, np.float64)
    ms = np.arange(-m_max, m_max + 1)
    M = np.stack(np.meshgrid(ms, ms, ms, indexing="ij"), -1).reshape(-1, 3).astype(np.float64)
    M = M[(M ** 2).sum(1) <= m_max * m_max]            # k-space cutoff
    is_dc = np.all(M == 0.0, axis=1)
    valid = np.ones(len(M))
    pad = (-len(M)) % chunk                            # pad to full chunks
    if pad:
        M = np.vstack([M, np.zeros((pad, 3))])
        is_dc = np.concatenate([is_dc, np.ones(pad, bool)])
        valid = np.concatenate([valid, np.zeros(pad)])
    nch = len(M) // chunk
    _dr, ri, rj = _pair_list(coords, box, rc, excl)
    n_real = (np.round((coords[ri] - coords[rj]) / box)
              if ri.size else np.zeros((0, 3)))
    ex = np.asarray(excl, int)
    n_excl = np.round((coords[ex[:, 0]] - coords[ex[:, 1]]) / box)
    return {"Mc": jnp.asarray(M.reshape(nch, chunk, 3)),
            "dcc": jnp.asarray(is_dc.reshape(nch, chunk)),
            "vc": jnp.asarray(valid.reshape(nch, chunk)),
            "ri": ri, "rj": rj, "n_real": jnp.asarray(n_real),
            "exi": ex[:, 0], "exj": ex[:, 1], "n_excl": jnp.asarray(n_excl)}


# ── differentiable energy pieces (functions of coords AND box matrix B) ──────
def _recip_chunk(coords, B, Mc, dc_c, vc, w, alpha, p, prefactor):
    """Reciprocal energy from one m-chunk (S(m) computed from coords ⇒
    differentiable for forces; box B carries the strain for the virial).  `vc`
    masks the padding rows that fill the chunk."""
    Binv = jnp.linalg.inv(B)
    V = jnp.abs(jnp.linalg.det(B))
    frac = coords @ Binv.T
    kv = Mc @ Binv
    k2 = jnp.sum(kv * kv, axis=1)
    ker = (_recip_p1 if p == 1 else _recip_p6)(alpha, k2, dc_c)
    ph = (2.0 * jnp.pi) * (Mc @ frac.T)
    Cc = jnp.cos(ph) @ w
    Dc = jnp.sin(ph) @ w
    S2 = Cc * Cc + Dc * Dc
    sign = 1.0 if p == 1 else -1.0
    return sign * (prefactor / (2.0 * V)) * jnp.sum(vc * ker * S2)


def _pair_term(coords, B, ii, jj, nimg, w, kern, alpha, sign, prefactor):
    """Pair-sum energy with a fixed integer min-image vector nimg: the
    displacement coords_i − coords_j − B·nimg is differentiable in coords & B."""
    dr = coords[ii] - coords[jj] - nimg @ B.T
    r2 = jnp.sum(dr * dr, axis=1)
    return sign * prefactor * jnp.sum((w[ii] * w[jj]) * kern(alpha, r2))


def _ewald_energy(coords, B, w, st, *, p, prefactor, alpha):
    """Total Ewald PME energy (real + recip + self − excl), a single jnp scalar.

    Differentiable in `coords` (⇒ forces) and in `B` (⇒ virial via affine
    strain), so `_ewald_forces`/`_ewald_virial` are one `jax.grad` of this.  The
    reciprocal m-sum is a jax.checkpoint'd lax.scan: the backward pass
    rematerialises one chunk at a time, so a single grad over all ~tens-of-k
    vectors stays memory-bounded.  Self-energy is constant under both gradients."""
    short_k = ewald_p1_short_range_value if p == 1 else ewald_p6_short_range_value
    long_k = ewald_p1_long_range_value if p == 1 else ewald_p6_long_range_value
    self_v = (-alpha / np.sqrt(np.pi)) if p == 1 else (alpha ** 6 / 12.0)
    rs = 1.0 if p == 1 else -1.0

    @jax.checkpoint
    def body(acc, chunk):
        Mc, dc_c, vc = chunk
        return acc + _recip_chunk(coords, B, Mc, dc_c, vc, w, alpha, p, prefactor), None

    E, _ = jax.lax.scan(body, 0.0, (st["Mc"], st["dcc"], st["vc"]))
    E = E + prefactor * self_v * jnp.sum(w ** 2)            # self (const grad)
    if st["ri"].size:
        E = E + _pair_term(coords, B, st["ri"], st["rj"], st["n_real"], w,
                           short_k, alpha, rs, prefactor)
    if st["exi"].size:                                      # −E_excl correction
        E = E - _pair_term(coords, B, st["exi"], st["exj"], st["n_excl"], w,
                           long_k, alpha, rs, prefactor)
    return E


def _ewald_forces(coords, B, w, st, *, p, prefactor, alpha):
    """F = −∂E/∂coords — a single jax.grad of `_ewald_energy`."""
    g = jax.grad(lambda c: _ewald_energy(c, B, w, st, p=p, prefactor=prefactor,
                                         alpha=alpha))
    return -np.asarray(g(coords))


def _ewald_virial(coords, B0, w, st, *, p, prefactor, alpha):
    """W = −∂E/∂ε (Voigt) under affine strain x→(I+ε)x, B→(I+ε)B — a single
    jax.grad of `_ewald_energy` w.r.t. the strain tensor."""
    def E_eps(eps):
        Fm = jnp.eye(3) + eps
        return _ewald_energy(coords @ Fm.T, Fm @ B0, w, st, p=p,
                             prefactor=prefactor, alpha=alpha)
    W = -np.asarray(jax.grad(E_eps)(jnp.zeros((3, 3))))
    W = 0.5 * (W + W.T)
    return np.array([W[0, 0], W[1, 1], W[2, 2], W[0, 1], W[0, 2], W[1, 2]])


def _ewald_reference(coords, box, weights, excl, *, p, prefactor,
                     alpha=_EWALD_ALPHA, m_max=_EWALD_MMAX, rc=None):
    """jax Ewald reference (E, F, W6) for a charge species — the single
    `_ewald_energy` differentiated for forces (∂coords) and virial (∂strain)."""
    box = np.asarray(box, np.float64)
    rc = rc if rc is not None else min(_EWALD_RC, float(np.min(box)) * 0.5 * 0.999)
    st = _ewald_static(np.asarray(coords, np.float64), box, excl, m_max=m_max, rc=rc)
    cj = jnp.asarray(coords, jnp.float64)
    Bj = jnp.asarray(np.diag(box))
    wj = jnp.asarray(np.asarray(weights, np.float64))
    E = float(_ewald_energy(cj, Bj, wj, st, p=p, prefactor=prefactor, alpha=alpha))
    F = _ewald_forces(cj, Bj, wj, st, p=p, prefactor=prefactor, alpha=alpha)
    W6 = _ewald_virial(cj, Bj, wj, st, p=p, prefactor=prefactor, alpha=alpha)
    return E, F, W6


def _openmm_energy_at(system, coords, box_diag):
    """Single fp64 potential energy at (coords, box) on the Reference platform."""
    integ = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    L = np.asarray(box_diag, np.float64)
    ctx.setPeriodicBoxVectors(openmm.Vec3(L[0], 0, 0) * unit.nanometer,
                              openmm.Vec3(0, L[1], 0) * unit.nanometer,
                              openmm.Vec3(0, 0, L[2]) * unit.nanometer)
    ctx.setPositions(coords * unit.nanometer)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)


def _openmm_virial_trace_fd(system, coords, box_diag, h=2e-5):
    """OpenMM Tr(W) = −dE/d(lnL) via isotropic box+coords scaling (central diff)."""
    box = np.asarray(box_diag, np.float64)
    Ep = _openmm_energy_at(system, coords * (1.0 + h), box * (1.0 + h))
    Em = _openmm_energy_at(system, coords * (1.0 - h), box * (1.0 - h))
    return -(Ep - Em) / (2.0 * h)


def _jax_reference_and_crosstest(coords, box, charges, sigmas, epsilons, excl, k_e,
                                 sys_coul, E_coul_omm, F_coul_omm,
                                 E_disp_omm, F_disp_omm, *, enforce=False):
    """jax Ewald references (Coulomb p1, C6-dispersion p6) + cross-tests vs OpenMM.

    Returns (W_coul, W_disp) Voigt virials for the blob.  Cross-checks: (1) jax
    Coulomb E/F vs OpenMM, (2) jax C6 dispersion E/F vs the geometric OpenMM
    dispersion system, (3) jax autodiff virial vs an OpenMM box-strain FD."""
    sqrtC6 = 2.0 * np.sqrt(np.maximum(epsilons, 0.0)) * sigmas ** 3

    def _relF(Fa, Fb):
        d = Fa - Fb
        return (float(np.sum(d ** 2)) / max(float(np.sum(Fb ** 2)), 1e-60)) ** 0.5

    print(f"  [jax Ewald ref] α={_EWALD_ALPHA} nm⁻¹, |m|≤{_EWALD_MMAX} "
          f"(exact reciprocal); E, F=−∂E/∂x, W=−∂E/∂ε via one jax.grad each")

    # ── (1) Coulomb (p=1): E + F + W ──
    E_c, F_c, W_c = _ewald_reference(coords, box, charges, excl, p=1, prefactor=k_e)
    print(f"  [xtest Coulomb E/F]  jax E={E_c:+.6e}  OpenMM {E_coul_omm:+.6e}  "
          f"relE={abs(E_c - E_coul_omm) / max(abs(E_coul_omm), 1e-30):.2e}  "
          f"relF={_relF(F_c, F_coul_omm):.2e}")

    # ── (2) C6 dispersion (p=6): E + F + W, vs the geometric OpenMM system ──
    E_d, F_d, W_d = _ewald_reference(coords, box, sqrtC6, excl, p=6, prefactor=1.0)
    print(f"  [xtest Disp   E/F]   jax E={E_d:+.6e}  OpenMM {E_disp_omm:+.6e}  "
          f"relE={abs(E_d - E_disp_omm) / max(abs(E_disp_omm), 1e-30):.2e}  "
          f"relF={_relF(F_d, F_disp_omm):.2e}")

    # ── (3) virial: jax autodiff trace vs OpenMM box-strain FD (Coulomb) ──
    trW_fd = _openmm_virial_trace_fd(sys_coul, coords, box)
    trW = float(W_c[0] + W_c[1] + W_c[2])
    print(f"  [xtest virial]       jax Tr(W)={trW:+.6e}  OpenMM-FD {trW_fd:+.6e}  "
          f"rel={abs(trW - trW_fd) / max(abs(trW_fd), 1e-30):.2e}")
    if enforce:
        # Bundled references must pass these independent checks before they are
        # written. Custom PDBs retain the existing reported comparisons:
        # their size/charge can require a larger Ewald shell or different cutoff.
        checks = {
            "Coulomb energy": (abs(E_c - E_coul_omm) / max(abs(E_coul_omm), 1e-30), 2e-6),
            "Coulomb forces": (_relF(F_c, F_coul_omm), 2e-6),
            "dispersion energy": (abs(E_d - E_disp_omm) / max(abs(E_disp_omm), 1e-30), 2e-6),
            "dispersion forces": (_relF(F_d, F_disp_omm), 2e-6),
            "Coulomb virial trace": (abs(trW - trW_fd) / max(abs(trW_fd), 1e-30), 1e-5),
        }
        for name, (error, tolerance) in checks.items():
            if not np.isfinite(error) or error > tolerance:
                raise RuntimeError(f"bundled reference {name} cross-check failed: "
                                   f"relative error {error:g} > {tolerance:g}")
        print("  [xtest] bundled reference cross-checks PASS")
    return W_c, W_d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", default=None,
                    help="Bundled system: water_10k (default). "
                         "With --pdb, an output label (default: input filename "
                         "without extensions).")
    ap.add_argument("--pdb", type=Path, default=None,
                    help="Neutral input PDB or PDB.gz using amber14-all + tip3pfb; "
                         "omit for the bundled water_10k system.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Override the output blob path. Bundled water goes to "
                         "data/water_10k/water_10k_ref.bin; custom "
                         "PDB references go beside this script.")
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF_NM,
                    help="Cutoff (nm) stored in the blob — used by the "
                         "C++ side (default 0.9).")
    ap.add_argument("--ref-cutoff", type=float, default=DEFAULT_REF_CUTOFF_NM,
                    help="Cutoff (nm) used by the OpenMM reference run "
                         "(default 2.0).  Larger lets `--ewald-tol` go "
                         "tighter without blowing up the PME grid size.")
    ap.add_argument("--ewald-tol", type=float, default=1e-9,
                    help="Ewald error tolerance (default 1e-9).  Tighter "
                         "is slow on the Reference platform (G scales as "
                         "√(-log tol)); 1e-7 already converges energies "
                         "to <1e-6 relative — well past the splittings' "
                         "prolate truncation floor (~1e-5 at c=10 / c=14).")
    args = ap.parse_args()
    if args.system is None:
        args.system = (Path(args.pdb.name.removesuffix(".gz")).stem
                       if args.pdb is not None else DEFAULT_SYSTEM)
    if args.pdb is None and args.system not in _BUNDLED_PDBS:
        ap.error("choose --system water_10k, or supply --pdb")
    pdb_path = args.pdb if args.pdb is not None else _BUNDLED_PDBS[args.system]
    if args.output is not None:
        out_path = args.output
    elif args.pdb is not None:
        out_path = _SCRIPT_DIR / f"{args.system}_ref.bin"
        bundled_outputs = {_system_out(name).resolve()
                           for name in _BUNDLED_PDBS}
        if out_path.resolve() in bundled_outputs:
            ap.error("the custom PDB output would overwrite a bundled reference; "
                     "choose --output explicitly")
    else:
        out_path = _system_out(args.system)
    pdb, coords, box, n_atoms = _load_pdb(pdb_path)
    print(f"  system:  {args.system}  ({pdb_path})")
    # Probe molecular parameters and 1-2/1-3 exclusions.
    # 1-4 and beyond retain full-strength nonbonded interactions.
    ff_probe = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    sys_probe = ff_probe.createSystem(
        pdb.topology, nonbondedMethod=openmm.app.PME,
        nonbondedCutoff=args.ref_cutoff * unit.nanometer,
        constraints=None, rigidWater=False)
    nbf_probe = next(f for f in sys_probe.getForces()
                     if isinstance(f, openmm.NonbondedForce))
    charges, sigmas, epsilons = _extract_params(nbf_probe, n_atoms)
    net_charge = float(np.sum(charges))
    if args.pdb is not None and abs(net_charge) > 1e-6:
        ap.error(f"the custom PDB must be neutral (net charge {net_charge:+.9g} e); "
                 "the JAX Ewald reference does not include the charged-system "
                 "neutralizing background term")
    nbrs = _bond_neighbors(sys_probe, n_atoms)
    coords = _make_molecules_whole(coords, box, nbrs)
    excl = _exclusions_12_13(nbrs)
    if not (0.0 < args.cutoff < np.min(box) / 2):
        ap.error("--cutoff must be positive and less than half the shortest box edge")
    if not (0.0 < args.ref_cutoff < np.min(box) / 2):
        ap.error("--ref-cutoff must be positive and less than half the shortest box edge")
    print(f"  atoms:   {n_atoms}")
    print(f"  box:     {box} nm")
    print(f"  cutoff:  {args.cutoff} nm  (ref: {args.ref_cutoff} nm)")

    print(f"  excl(1-2/1-3): {len(excl)}  charges_nz={int((charges != 0).sum())}  "
          f"lj_nz={int((epsilons != 0).sum())}")

    # ── Pass 1: Coulomb-only ────────────────────────────────────────────
    sys_coul = _build_coulomb_system(box, args.ref_cutoff, args.ewald_tol,
                                     charges, excl)
    E_coul, F_coul = _eval(sys_coul, coords, label="Coulomb")

    # ── Pass 2: C6-only dispersion (geometric combining, no R¹²) ───────────
    sys_disp = _build_dispersion_system(
        coords, box, sigmas, epsilons, excl,
        cutoff=args.ref_cutoff, alpha_lj=_DISP_ALPHA_LJ, grid=_DISP_GRID)
    E_disp, F_disp = _eval_dispersion(sys_disp, coords, box)
    print(f"  [Disp   ] E = {E_disp:.6f} kJ/mol   |F|_max = "
          f"{np.max(np.linalg.norm(F_disp, axis=1)):.3f}  (C6-only, geometric)")

    k_e = 138.935458  # OpenMM Coulomb constant, kJ·nm/(mol·e²)

    # ── jax-autodiff Ewald references (E/F/virial) + cross-tests vs OpenMM ─
    W_coul, W_disp = _jax_reference_and_crosstest(
        coords, box, charges, sigmas, epsilons, excl, k_e,
        sys_coul, E_coul, F_coul, E_disp, F_disp, enforce=args.pdb is None)

    # ── Write binary blob ───────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", n_atoms, len(excl)))
        f.write(box.tobytes())
        f.write(struct.pack("<dd", float(args.cutoff), float(k_e)))
        f.write(coords.tobytes())
        f.write(charges.tobytes())
        f.write(sigmas.tobytes())
        f.write(epsilons.tobytes())
        f.write(excl.tobytes())
        f.write(struct.pack("<d", float(E_coul)))
        f.write(F_coul.tobytes())
        f.write(struct.pack("<d", float(E_disp)))
        f.write(F_disp.tobytes())
        f.write(np.ascontiguousarray(W_coul, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(W_disp, dtype=np.float64).tobytes())
    size_kib = out_path.stat().st_size / 1024
    print(f"  wrote:   {out_path}  ({size_kib:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
