#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine stock PME and alchemical LJ in one plugin NonbondedForce.

The prepared TYK2 ejm_31 -> ejm_42 system contains a stock NonbondedForce
(ordinary LJ, PME charges and parameter offsets) and a CustomNonbondedForce
(OpenFE's Gapsys softcore LJ). combine_nonbonded() takes those two forces and
returns one nv.NonbondedForce. Bonded forces, including the sparse alchemical
exception bonds, remain in the System.

The important API calls are at the end of combine_nonbonded(): construct from
the stock NonbondedForce and attach the COMPLETE LJ expression with
setCutoffInteraction(..., tailCorrection=True). The constructor defaults to
Prolate electrostatics and automatic box safety. The expression includes
ordinary LJ because setCutoffInteraction replaces it. Interaction groups become
per-particle membership flags; globals and stock parameter offsets stay live.
The built-in switching function applies once to the combined LJ expression.
One homogeneous tail correction integrates that complete expression using
ungrouped pair averaging. This intentionally differs from preserving the original
CustomNonbondedForce's interaction-group tail counts, so total energies can differ
slightly from the original two-force System even when pair forces agree.

This example accepts stock periodic PME plus one periodic-cutoff expression with
matching cutoffs, exclusions, switching and tail-enablement settings. Particle
values and group membership are copied at conversion time: edit the sources and
reconvert if those change. Source force groups are combined into the stock
NonbondedForce's group. Energy parameter derivatives and tabulated/computed custom
values are outside this example's scope.

Only OpenMM, NumPy and the plugin are needed; data/ contains a prepared stock
System and coordinates. No OpenFE installation or network access is required.
Run ./run to compare energies and forces at the prepared window, after changing
a charge-offset global, and after changing softcore globals, then run short NVT
dynamics. These perturbations demonstrate live updates, not an RBFE protocol.
"""

import gzip
from pathlib import Path
import re

import numpy as np
import openmm as mm
from openmm import unit
import nv_openmm_plugin as nv


def scalar(value, units):
    return value.value_in_unit(units) if unit.is_quantity(value) else float(value)


def rename(expression, names):
    """Rename whole identifiers, including the 1/2 particle suffixes."""
    return re.sub(r"\b[A-Za-z_]\w*\b", lambda m: names.get(m[0], m[0]), expression)


def group_mask(stock_custom_nonbonded, add_field):
    """Replace interaction groups with symmetric 0/1 membership expressions."""
    fields, terms = {}, []

    def membership(atoms):
        atoms = frozenset(atoms)
        if atoms not in fields:
            fields[atoms] = add_field(f"group{len(fields)}",
                                     [int(i in atoms) for i in range(stock_custom_nonbonded.getNumParticles())])
        return fields[atoms]

    for i in range(stock_custom_nonbonded.getNumInteractionGroups()):
        left, right = map(frozenset, stock_custom_nonbonded.getInteractionGroupParameters(i))
        a, b = membership(left), membership(right)
        ab, ba = f"{a}1*{b}2", f"{a}2*{b}1"
        if left == right:
            terms.append(ab)
        elif left.isdisjoint(right):
            terms.append(f"({ab}+{ba})")
        else:
            # Count an overlapping pair once within a group, but retain its
            # multiplicity when it appears in several different groups.
            terms.append(f"({ab}+{ba}-({ab})*({ba}))")
    return "+".join(terms) if terms else "1"


def combine_nonbonded(stock_nonbonded, stock_custom_nonbonded):
    """Return one plugin NonbondedForce from two stock force definitions."""
    if (stock_nonbonded.getNonbondedMethod() != stock_nonbonded.PME or
            stock_custom_nonbonded.getNonbondedMethod() != stock_custom_nonbonded.CutoffPeriodic):
        raise ValueError("Expected stock NonbondedForce PME and custom CutoffPeriodic")
    n = stock_nonbonded.getNumParticles()
    if (stock_custom_nonbonded.getNumParticles() != n or
            stock_custom_nonbonded.getCutoffDistance() != stock_nonbonded.getCutoffDistance()):
        raise ValueError("The two forces must have matching particle counts and cutoffs")
    use_switching = stock_nonbonded.getUseSwitchingFunction()
    if (stock_custom_nonbonded.getUseSwitchingFunction() != use_switching or
            (use_switching and stock_custom_nonbonded.getSwitchingDistance() != stock_nonbonded.getSwitchingDistance())):
        raise ValueError("The two forces must have matching switching settings")
    if stock_nonbonded.getUseDispersionCorrection() != stock_custom_nonbonded.getUseLongRangeCorrection():
        raise ValueError("The two forces must have matching tail-correction settings")
    exceptions = {tuple(sorted(stock_nonbonded.getExceptionParameters(i)[:2]))
                  for i in range(stock_nonbonded.getNumExceptions())}
    exclusions = {tuple(sorted(stock_custom_nonbonded.getExclusionParticles(i)))
                  for i in range(stock_custom_nonbonded.getNumExclusions())}
    if exclusions != exceptions:
        raise ValueError("Custom exclusions must match stock NonbondedForce exception pairs")
    if (stock_custom_nonbonded.getNumTabulatedFunctions() or stock_custom_nonbonded.getNumEnergyParameterDerivatives()
            or (hasattr(stock_custom_nonbonded, "getNumComputedValues") and
                stock_custom_nonbonded.getNumComputedValues())):
        raise ValueError("This sample does not lower tabulated/computed values or energy derivatives")

    global_parameters = {}
    for force in (stock_nonbonded, stock_custom_nonbonded):
        for i in range(force.getNumGlobalParameters()):
            name, value = force.getGlobalParameterName(i), force.getGlobalParameterDefaultValue(i)
            if name in global_parameters and global_parameters[name] != value:
                raise ValueError(f"Conflicting default for global {name}")
            global_parameters[name] = value

    # Prefix all generated identifiers so the stock expression remains intact.
    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", stock_custom_nonbonded.getEnergyFunction())) | set(global_parameters)
    prefix = "fused_"
    while any(name.startswith(prefix) for name in identifiers):
        prefix = "_" + prefix
    fields, columns = [], []

    def add_field(name, values):
        name = prefix + name
        fields.append(name)
        columns.append(values)
        return name

    values = [stock_nonbonded.getParticleParameters(i) for i in range(n)]
    sigma = add_field("sigma", [scalar(v[1], unit.nanometer) for v in values])
    epsilon = add_field("epsilon", [scalar(v[2], unit.kilojoule_per_mole) for v in values])
    sigma_terms, epsilon_terms = [sigma], [epsilon]
    offsets = {}
    for i in range(stock_nonbonded.getNumParticleParameterOffsets()):
        name, atom, charge, ds, de = stock_nonbonded.getParticleParameterOffset(i)
        ds, de = scalar(ds, unit.nanometer), scalar(de, unit.kilojoule_per_mole)
        if ds or de:
            offsets.setdefault(name, np.zeros((n, 2)))[atom] += (ds, de)
    # TYK2's stock particle offsets change only charges. For callers with LJ
    # offsets too, the new expression must explicitly reproduce their laws.
    for i, (name, changes) in enumerate(offsets.items()):
        ds = add_field(f"sigmaOffset{i}", changes[:, 0])
        de = add_field(f"epsilonOffset{i}", changes[:, 1])
        sigma_terms.append(f"{name}*{ds}")
        epsilon_terms.append(f"{name}*{de}")
    def particle(terms, suffix):
        return rename("+".join(terms), {name: name+suffix for name in fields})
    s1, s2 = (particle(sigma_terms, suffix) for suffix in ("1", "2"))
    e1, e2 = (particle(epsilon_terms, suffix) for suffix in ("1", "2"))
    sr6 = prefix + "sr6"
    stock_lj_energy = f"4*sqrt(({e1})*({e2}))*{sr6}*({sr6}-1)"
    definitions = [f"{sr6}=((({s1})+({s2}))/(2*r))^6"]

    # Copy the stock custom expression and parameters, with renamed identifiers
    # to avoid collisions with the ordinary LJ definitions above.
    names = {}
    rows = [stock_custom_nonbonded.getParticleParameters(i) for i in range(n)]
    for i in range(stock_custom_nonbonded.getNumPerParticleParameters()):
        source = stock_custom_nonbonded.getPerParticleParameterName(i)
        target = add_field(f"custom{i}", [row[i] for row in rows])
        names.update({source+suffix: target+suffix for suffix in ("1", "2")})
    pieces = [piece.strip() for piece in stock_custom_nonbonded.getEnergyFunction().split(";") if piece.strip()]
    for i, definition in enumerate(pieces[1:]):
        names[definition.split("=", 1)[0].strip()] = prefix + f"local{i}"
    custom_energy = rename(pieces[0], names)
    definitions.extend(rename(piece, names) for piece in pieces[1:])
    selected = prefix + "selected"
    definitions.append(f"{selected}=" + group_mask(stock_custom_nonbonded, add_field))
    # select also suppresses undefined softcore branches for excluded regions.
    energy = f"({stock_lj_energy})+{selected}*select({selected},({custom_energy}),0)"

    # One force, one direct pair traversal. Copy charges, exception laws, and
    # parameter offsets, using the default Prolate splitting and box safety.
    nv_nonbonded = nv.NonbondedForce(stock_nonbonded)
    # Inherit the cutoff, exclusions and shared switch from the stock force.
    # The switch and one ungrouped tail correction cover the sum of both LJ terms.
    nv_nonbonded.setCutoffInteraction(
        expression=energy + ";" + ";".join(definitions),
        parameterNames=fields,
        particleParameters=list(zip(*columns)),
        globalParameters=global_parameters,
        tailCorrection=stock_nonbonded.getUseDispersionCorrection(),
    )
    return nv_nonbonded


def replace_nonbonded(system):
    """Replace only the two pair forces; retain bonded and sparse exception terms."""
    stock_nonbonded = [(i, f) for i, f in enumerate(system.getForces()) if isinstance(f, mm.NonbondedForce)]
    stock_custom_nonbonded = [(i, f) for i, f in enumerate(system.getForces()) if isinstance(f, mm.CustomNonbondedForce)]
    if len(stock_nonbonded) != 1 or len(stock_custom_nonbonded) != 1:
        raise ValueError("Expected one stock NonbondedForce and one CustomNonbondedForce")
    nv_nonbonded = combine_nonbonded(stock_nonbonded[0][1], stock_custom_nonbonded[0][1])
    for i in sorted((stock_nonbonded[0][0], stock_custom_nonbonded[0][0]), reverse=True):
        system.removeForce(i)
    system.addForce(nv_nonbonded)
    return nv_nonbonded


def energy_and_forces(context):
    state = context.getState(getEnergy=True, getForces=True)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))


def main():
    data = Path(__file__).resolve().parent / "data"
    with gzip.open(data / "system.xml.gz", "rt") as stream:
        xml = stream.read()
    stock = mm.XmlSerializer.deserialize(xml)
    plugin = mm.XmlSerializer.deserialize(xml)
    stock_nonbonded = next(f for f in stock.getForces() if isinstance(f, mm.NonbondedForce))
    stock_custom_nonbonded = next(f for f in stock.getForces() if isinstance(f, mm.CustomNonbondedForce))
    atoms = sorted({stock_nonbonded.getParticleParameterOffset(i)[1]
                    for i in range(stock_nonbonded.getNumParticleParameterOffsets())})
    defaults = {f.getGlobalParameterName(i): f.getGlobalParameterDefaultValue(i)
                for f in (stock_nonbonded, stock_custom_nonbonded) for i in range(f.getNumGlobalParameters())}
    nv_nonbonded = replace_nonbonded(plugin)
    print(f"TYK2 RBFE: {stock.getNumParticles()} atoms; stock NonbondedForce + "
          "CustomNonbondedForce -> one nv.NonbondedForce", flush=True)
    print(f"Preserved {nv_nonbonded.getNumParticleParameterOffsets()} particle offsets, "
          f"{nv_nonbonded.getNumExceptionParameterOffsets()} exception offsets; "
          f"lowered {stock_custom_nonbonded.getNumInteractionGroups()} interaction groups.", flush=True)
    with np.load(data / "positions.npz", allow_pickle=False) as archive:
        positions = archive["positions_nm"]
        velocities = archive["velocities_nm_per_ps"]
        box = archive["box_vectors_nm"]

    integrators, contexts = [], []
    for system in (stock, plugin):
        integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond,
                                               .004*unit.picoseconds)
        integrator.setRandomNumberSeed(20261025)
        integrator.setConstraintTolerance(1e-6)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                             {"Precision": "mixed", "DeterministicForces": "true"})
        context.setPeriodicBoxVectors(*(mm.Vec3(*v) for v in box))
        context.setPositions(positions)
        context.setVelocities(velocities)
        integrators.append(integrator)
        contexts.append(context)

    windows = [("prepared", {}), ("charge offset", {"lambda_electrostatics_core": .6}),
               ("softcore", {"lambda_sterics_insert": .65, "lambda_sterics_delete": .35}),
               ("prepared again", {})]
    print("Stock uses PME; the plugin uses default Prolate electrostatics. "
          "Energy differences also include the change from the original grouped "
          "tail correction to one ungrouped correction.", flush=True)
    print("window             delta E (kJ/mol)    energy error     force error    ligand force error", flush=True)
    energies = []
    for label, changes in windows:
        for context in contexts:
            for name, value in (defaults | changes).items():
                context.setParameter(name, value)
        es, fs = energy_and_forces(contexts[0])
        ep, fp = energy_and_forces(contexts[1])
        if not np.isfinite(ep) or not np.isfinite(fp).all():
            raise RuntimeError(f"Nonfinite plugin result at {label}")
        e_err = abs(ep-es)/max(abs(es), 1.)
        f_err = np.linalg.norm(fp-fs)/np.linalg.norm(fs)
        ligand_err = np.linalg.norm((fp-fs)[atoms])/np.linalg.norm(fs[atoms])
        print(f"{label:18s} {ep-es:15.6f} {e_err:15.2e} {f_err:15.2e} {ligand_err:21.2e}", flush=True)
        # PME and Prolate have different numerical approximation errors. Include
        # those errors, mixed-precision arithmetic, and the changed tail
        # averaging convention in this comparison budget. Use the existing TYK2
        # Prolate benchmark's full-system force bound; keep the tighter energy
        # and ligand-force checks for this prepared example.
        if e_err > 1e-5 or f_err > 1e-3 or ligand_err > 5e-4:
            raise RuntimeError(f"Stock/plugin comparison failed at {label}")
        energies.append((es, ep))
    for i in (1, 2):
        ds, dp = np.asarray(energies[i]) - energies[0]
        if abs(ds) < .1 or abs(dp-ds) > .05:
            raise RuntimeError(f"Energy change disagrees with stock at {windows[i][0]}")
    if not np.allclose(energies[-1], energies[0], rtol=0, atol=1e-5):
        raise RuntimeError("Returning to the prepared window changed its energy")

    integrators[1].step(200)
    energy, forces = energy_and_forces(contexts[1])
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise RuntimeError("Nonfinite result after NVT dynamics")
    print("PASS: live charge/softcore updates agree with stock; 200 NVT steps completed.")


if __name__ == "__main__":
    main()
