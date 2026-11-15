# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen TYK2 alchemical Hamiltonians and their one-traversal cuPo lowering.

Loading uses only NumPy and OpenMM. OpenFE and OpenFF are generation-time tools;
no force-field assignment, atom mapping, or network access occurs here. The stock
System is the numerical reference. Its sparse bonded terms remain stock OpenMM
when the nonbonded pair terms are fused for cuPo.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .datasets import extra_dirs

_DATA_DIR = Path(__file__).resolve().parent / "tyk2_alchemical"
# Extra cases, each directory with its own manifest: tyk2_alchemical/ in each extra
# dataset directory, from cupo-repo.ini at the repository root when present.
_EXTRA_DIRS = [d / "tyk2_alchemical" for d in extra_dirs()]


def _fixture_dirs():
    """The fixture directories there are, each with a manifest.json."""
    return [d for d in [_DATA_DIR] + _EXTRA_DIRS if (d / "manifest.json").is_file()]


def _cases():
    """The cases on disk, in manifest order: rbfe and abfe, then the extra cases,
    when there are any."""
    out = []
    for folder in _fixture_dirs():
        out += [c for c in json.loads((folder / "manifest.json").read_text())["systems"]
                if c not in out]
    return tuple(out)


CASES = _cases()
SYSTEM_NAMES = tuple("tyk2_" + case + "_dodec" for case in CASES)


def _require(condition, message):
    if not condition:
        raise ValueError("TYK2 alchemical fixture: " + message)


def _scalar(value, target):
    from openmm import unit
    return float(value.value_in_unit(target)) if unit.is_quantity(value) else float(value)


@dataclass(frozen=True)
class Tyk2AlchemicalDataset:
    case: str
    system_xml: str
    positions: np.ndarray
    box: np.ndarray
    velocities: np.ndarray | None
    globals: dict[str, float]
    metadata: dict
    manifest_sha256: str

    @property
    def name(self):
        return "tyk2_" + self.case + "_dodec"

    def create_system(self):
        """Deserialize a new stock System at the manifest's explicit midpoint."""
        import openmm as mm
        system = mm.XmlSerializer.deserialize(self.system_xml)
        found = set()
        for force in system.getForces():
            if not hasattr(force, "getNumGlobalParameters"):
                continue
            for index in range(force.getNumGlobalParameters()):
                name = force.getGlobalParameterName(index)
                if name in self.globals:
                    force.setGlobalParameterDefaultValue(index, self.globals[name])
                    found.add(name)
        _require(found == set(self.globals), "window names globals absent from the System")
        _require(system.getNumParticles() == len(self.positions), "System/position particle counts differ")
        return system

    def benchmark_metadata(self):
        return dict(self.metadata, fixture=self.name, globals=dict(self.globals),
                    manifest_sha256=self.manifest_sha256,
                    scope="complete fixed-window MD, including neighbor/cell maintenance")


def load_tyk2_alchemical(case: str) -> Tyk2AlchemicalDataset:
    """Load a short case key or tyk2_<case>_dodec without parameterization."""
    if case in SYSTEM_NAMES:
        case = case[len("tyk2_"):-len("_dodec")]
    folder = manifest = None
    for candidate in _fixture_dirs():
        manifest_bytes = (candidate / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if case in manifest.get("systems", {}):
            folder = candidate
            break
    _require(folder is not None, f"unknown case {case!r}; choose {', '.join(CASES)}")
    _require(manifest.get("schema_version") == 1, "unsupported manifest schema")
    entry = manifest["systems"][case]
    payloads = {}
    for kind in ("system", "positions"):
        relative = entry["files"][kind]
        path = (folder / relative).resolve()
        _require(path.is_relative_to(folder.resolve()), "file path leaves the fixture directory")
        payloads[kind] = path.read_bytes()
        _require(hashlib.sha256(payloads[kind]).hexdigest() == manifest["file_sha256"][relative],
                 f"{relative} checksum mismatch")
    xml_bytes = payloads["system"]
    if entry["files"]["system"].endswith(".gz"):
        xml_bytes = gzip.decompress(xml_bytes)
    import io
    with np.load(io.BytesIO(payloads["positions"]), allow_pickle=False) as archive:
        positions = np.asarray(archive["positions_nm"], dtype=np.float64)
        box = np.asarray(archive["box_vectors_nm"], dtype=np.float64)
        velocities = (np.asarray(archive["velocities_nm_per_ps"], dtype=np.float64)
                      if "velocities_nm_per_ps" in archive else None)
    metadata = dict(entry["metadata"])
    _require(positions.shape == (metadata["atoms"], 3), "invalid position array shape")
    _require(box.shape == (3, 3), "invalid periodic box array shape")
    _require(np.isfinite(positions).all() and np.isfinite(box).all(), "nonfinite coordinates or box")
    _require(np.linalg.det(box) > 0, "nonpositive periodic volume")
    if velocities is not None:
        _require(velocities.shape == positions.shape and np.isfinite(velocities).all(),
                 "invalid velocity array")
    globals_ = {str(name): float(value) for name, value in entry["globals"].items()}
    _require(all(np.isfinite(value) for value in globals_.values()), "nonfinite window global")
    _require(float(metadata["timestep_fs"]) > 0, "nonpositive timestep")
    return Tyk2AlchemicalDataset(case, xml_bytes.decode(), positions, box, velocities, globals_, metadata,
                                 hashlib.sha256(manifest_bytes).hexdigest())


def _rename_expression(expression, rename):
    return re.sub(r"\b[A-Za-z_]\w*\b", lambda match: rename.get(match[0], match[0]), expression)


def _switch_expression(expression, start, cutoff):
    # OpenMM's automatic Custom switch multiplies its complete pair expression.
    # Preserve each source term's switch when combining several terms into one
    # expression; a switch on the fused expression would multiply every term.
    x = f"((r-({start:.17g}))/({cutoff-start:.17g}))"
    return f"({expression})*select(step(r-({start:.17g})),1-10*{x}^3+15*{x}^4-6*{x}^5,1)"


@dataclass
class FusedPairDefinition:
    """One expression plus source tail policies, before binding to the plugin."""
    expression: object
    native_index: int
    custom_indices: tuple[int, ...]
    native_tail: bool
    custom_tails: tuple[object, ...]


def _interaction_group_class_plan(n, groups):
    """Private fixture lowering: exact membership classes, at most 24 codes."""
    if not n or not groups:
        return None
    endpoints = [frozenset(atoms) for group in groups for atoms in group]
    if any(any(atom < 0 or atom >= n for atom in atoms) for atoms in endpoints):
        raise ValueError("interaction-group atom outside System")
    signatures = [tuple(int(atom in atoms) for atoms in endpoints) for atom in range(n)]
    classes = sorted(set(signatures))
    # Largest distinct sum is 2**23 + 2**22; all class-pair sums, including
    # diagonal 2**24, are distinct and exactly representable in float32.
    if len(classes) > 24:
        return None
    lookup = {signature: index for index, signature in enumerate(classes)}
    ids = np.array([lookup[signature] for signature in signatures], dtype=np.int32)
    return dict(codes=np.ldexp(np.ones(n, dtype=np.float32), ids), signatures=classes)


def _interaction_group_class_table(plan, offset, count):
    signatures = plan["signatures"]
    table = []
    for i, first in enumerate(signatures):
        for j in range(i, len(signatures)):
            second = signatures[j]
            multiplicity = 0
            for group in range(offset, offset+count):
                a, b = 2*group, 2*group+1
                multiplicity += bool((first[a] and second[b]) or (first[b] and second[a]))
            table.append(((1 << i)+(1 << j), multiplicity))
    return table


def _interaction_group_class_expression(plan, offset, count, field):
    table = _interaction_group_class_table(plan, offset, count)
    if not table:
        return "0"
    # Minimize the number of delta exceptions, not a guessed runtime population.
    values = [value for code, value in table]
    default = min(set(values), key=lambda value: (-values.count(value), value))
    terms = [str(default)] if default else []
    pair = f"({field}1+{field}2)"
    for code, value in table:
        coefficient = value-default
        if coefficient:
            terms.append(f"({coefficient})*delta({pair}-({code}))")
    return "+".join(terms) if terms else "0"


def _interaction_group_mask_costs(plan, groups, ranges):
    """Estimate predicate work; this is not a compiler/GPU instruction model.

    Count unique group terms plus their sums, and shared class-code comparisons
    plus sparse exceptions. Ignore deeper compiler algebra and record traffic.
    """
    terms, fields, codes, products = {}, set(), set(), set()
    simple_adds = encoded_adds = 0
    for offset, count in ranges:
        if not count:
            continue
        simple_adds += count-1
        for left, right in groups[offset:offset+count]:
            left, right = frozenset(left), frozenset(right)
            fields.update((left, right))
            terms[frozenset((left, right))] = (1 if left == right else
                                              3 if left.isdisjoint(right) else 5)
        table = _interaction_group_class_table(plan, offset, count)
        values = [value for code, value in table]
        default = min(set(values), key=lambda value: (-values.count(value), value))
        exceptions = [(code, value-default) for code, value in table if value != default]
        codes.update(code for code, coefficient in exceptions)
        products.update((code, coefficient) for code, coefficient in exceptions if coefficient != 1)
        encoded_adds += max(0, len(exceptions)+bool(default)-1)
    # A class-code sum is shared; each delta requires one subtraction and test.
    encoded_ops = bool(codes)+2*len(codes)+len(products)+encoded_adds
    return (sum(terms.values())+simple_adds, len(fields)), (encoded_ops, int(bool(codes)))


def _choose_interaction_group_classes(n, groups, ranges):
    # At most 300 class-pair entries; multiplicities must be exact in float32.
    if any(count > 2**24 for offset, count in ranges):
        return None
    plan = _interaction_group_class_plan(n, groups)
    if plan is None:
        return None
    simple, encoded = _interaction_group_mask_costs(plan, groups, ranges)
    # Prefer ordinary masks on ties; fewer fields alone do not justify more math.
    return plan if encoded[0] < simple[0] and encoded[1] <= simple[1] else None


def fuse_pair_definition(system) -> FusedPairDefinition:
    """Lower stock pair terms and interaction-group membership to one expression.

    This is fixture preparation, not a general OpenMM conversion API. It preserves
    duplicate interaction-group multiplicity and leaves native exception/1–4 and
    sparse bonded terms with their original force and distance convention. All custom
    exclusions must match the native exceptions, so one cancellation contract suffices.
    The benchmark evaluates all force groups: the fused pair force adopts the
    native force group, so source pair terms cannot be queried independently by
    their original groups after conversion. Generated region/class fields encode
    fixed topology and particle parameter snapshots; changing source groups or
    particle parameters requires reconversion. Global parameters remain live.
    """
    import openmm as mm
    from openmm import unit

    forces = list(system.getForces())
    native_indices = [i for i, force in enumerate(forces) if isinstance(force, mm.NonbondedForce)]
    _require(len(native_indices) == 1, "expected exactly one stock NonbondedForce")
    native_index = native_indices[0]
    native = forces[native_index]
    _require(native.getNonbondedMethod() == native.PME, "the native force must use PME")
    cutoff = _scalar(native.getCutoffDistance(), unit.nanometer)
    n = system.getNumParticles()
    exceptions = {tuple(sorted(native.getExceptionParameters(i)[:2]))
                  for i in range(native.getNumExceptions())}
    custom_indices = tuple(i for i, force in enumerate(forces) if isinstance(force, mm.CustomNonbondedForce))
    _require(custom_indices, "expected an alchemical CustomNonbondedForce")
    names, columns, terms, definitions, globals_ = [], [], [], [], {}
    tail_definitions = []
    membership_fields = {}

    # This converter takes a fixed parameter snapshot. Preserve complete double
    # representations, including signed zero, when sharing identical columns.
    # Source parameter changes require reconversion; live global laws are kept.
    field_snapshots, requested_names = {}, set()

    def add_field(name, values):
        _require(name not in requested_names, f"duplicate lowered particle field {name}")
        requested_names.add(name)
        column = np.asarray(values, dtype=np.float64)
        key = column.tobytes()
        if key in field_snapshots:
            return field_snapshots[key]
        field_snapshots[key] = name
        names.append(name)
        columns.append(column)
        return name

    def add_globals(force):
        for index in range(force.getNumGlobalParameters()):
            name = force.getGlobalParameterName(index)
            value = float(force.getGlobalParameterDefaultValue(index))
            _require(name not in globals_ or globals_[name] == value,
                     f"conflicting default for global {name}")
            globals_[name] = value

    add_globals(native)
    particle_values = np.asarray([
        (_scalar(sigma, unit.nanometer), _scalar(epsilon, unit.kilojoule_per_mole))
        for charge, sigma, epsilon in (native.getParticleParameters(i) for i in range(n))])
    native_lj_is_zero = not np.any(particle_values[:, 1] != 0)
    finite_sigmas = np.all(np.isfinite(particle_values[:, 0]))
    offsets = {}
    for index in range(native.getNumParticleParameterOffsets()):
        parameter, atom, charge, sigma, epsilon = native.getParticleParameterOffset(index)
        sigma = _scalar(sigma, unit.nanometer)
        epsilon = _scalar(epsilon, unit.kilojoule_per_mole)
        native_lj_is_zero &= epsilon == 0
        finite_sigmas &= np.isfinite(sigma)
        if sigma or epsilon:
            offsets.setdefault(parameter, np.zeros((n, 2)))[atom] += (sigma, epsilon)
    # A fixed zero-strength native LJ term stays zero for every live global.
    # Inspect each original epsilon offset, not its value at today's lambda or
    # an aggregate that happens to cancel. Source exceptions and tails remain.
    if not native_lj_is_zero or not finite_sigmas:
        sigma_name = add_field("nativeSigma", particle_values[:, 0])
        epsilon_name = add_field("nativeEpsilon", particle_values[:, 1])
        sigma_terms = [sigma_name]
        epsilon_terms = [epsilon_name]
        for index, (parameter, values) in enumerate(offsets.items()):
            sig = add_field(f"nativeSigmaOffset{index}", values[:, 0])
            eps = add_field(f"nativeEpsilonOffset{index}", values[:, 1])
            sigma_terms.append(f"{parameter}*{sig}")
            epsilon_terms.append(f"{parameter}*{eps}")
        def per_atom(parts, suffix):
            rename = {name: name+suffix for name in names}
            return _rename_expression("+".join(parts), rename)
        sig1, sig2 = (per_atom(sigma_terms, suffix) for suffix in ("1", "2"))
        eps1, eps2 = (per_atom(epsilon_terms, suffix) for suffix in ("1", "2"))
        native_lj = (f"4*sqrt(({eps1})*({eps2}))*nativeSr6*(nativeSr6-1)")
        definitions.append(f"nativeSr6=((({sig1})+({sig2}))/(2*r))^6")
        if native.getUseSwitchingFunction():
            native_lj = _switch_expression(native_lj, _scalar(native.getSwitchingDistance(), unit.nanometer), cutoff)
        terms.append(native_lj)

    all_groups = []
    group_offsets = {}
    for force_index in custom_indices:
        force = forces[force_index]
        group_offsets[force_index] = len(all_groups)
        all_groups.extend(force.getInteractionGroupParameters(i)
                          for i in range(force.getNumInteractionGroups()))
    ranges = [(group_offsets[index], forces[index].getNumInteractionGroups())
              for index in custom_indices]
    class_plan = _choose_interaction_group_classes(n, all_groups, ranges)
    class_field = (add_field("interactionClass", class_plan["codes"])
                   if class_plan is not None else None)

    for force_index in custom_indices:
        force = forces[force_index]
        prefix = f"pair{force_index}_"
        _require(force.getNumParticles() == n, "Custom/native particle counts differ")
        _require(force.getNonbondedMethod() == force.CutoffPeriodic, "custom terms must use CutoffPeriodic")
        _require(_scalar(force.getCutoffDistance(), unit.nanometer) == cutoff, "pair cutoffs differ")
        _require(force.getNumTabulatedFunctions() == 0, "tabulated pair functions are not supported")
        _require(not hasattr(force, "getNumComputedValues") or force.getNumComputedValues() == 0,
                 "computed pair values are not supported")
        _require(force.getNumEnergyParameterDerivatives() == 0,
                 "energy parameter derivatives are not supported")
        excluded = {tuple(sorted(force.getExclusionParticles(i))) for i in range(force.getNumExclusions())}
        _require(excluded == exceptions, "Custom exclusions must equal native exception pairs")
        add_globals(force)
        rename = {}
        parameters = np.asarray([force.getParticleParameters(i) for i in range(n)], dtype=np.float64)
        for index in range(force.getNumPerParticleParameters()):
            source = force.getPerParticleParameterName(index)
            name = add_field(prefix + source, parameters[:, index])
            rename.update({source+suffix: name+suffix for suffix in ("1", "2")})
        pieces = [piece.strip() for piece in force.getEnergyFunction().split(";") if piece.strip()]
        for definition in pieces[1:]:
            name = definition.split("=", 1)[0].strip()
            rename[name] = prefix + name
        energy = _rename_expression(pieces[0], rename)
        local_definitions = [_rename_expression(piece, rename) for piece in pieces[1:]]
        if force.getUseSwitchingFunction():
            energy = _switch_expression(energy, _scalar(force.getSwitchingDistance(), unit.nanometer), cutoff)
        masks = []
        if class_plan is not None and force.getNumInteractionGroups():
            masks.append(_interaction_group_class_expression(class_plan,
                group_offsets[force_index], force.getNumInteractionGroups(), class_field))
        else:
            for index in range(force.getNumInteractionGroups()):
                left, right = force.getInteractionGroupParameters(index)
                def membership(atoms):
                    key = frozenset(atoms)
                    if key not in membership_fields:
                        membership_fields[key] = add_field(
                            f"region{len(membership_fields)}", [int(atom in key) for atom in range(n)])
                    return membership_fields[key]
                a, b = membership(left), membership(right)
                ab, ba = f"{a}1*{b}2", f"{a}2*{b}1"
                if set(left) == set(right):
                    masks.append(ab)
                elif set(left).isdisjoint(right):
                    masks.append(f"({ab}+{ba})")
                else:
                    masks.append(f"({ab}+{ba}-({ab})*({ba}))")
        if masks:
            mask = prefix+"selected"
            local_definitions.append(mask+"="+"+".join(masks))
            energy = f"{mask}*select({mask},({energy}),0)"
        terms.append(energy)
        definitions.extend(local_definitions)
        if force.getUseLongRangeCorrection():
            # Keep source corrections separate: a partial LRC policy must not
            # accidentally integrate pair terms whose source correction is off.
            tail_definitions.append(force)

    result = mm.CustomNonbondedForce("+".join(f"({term})" for term in terms) + ";" + ";".join(definitions))
    result.setName("TYK2 fused ordinary and alchemical pair expression")
    result.setNonbondedMethod(result.CutoffPeriodic)
    result.setCutoffDistance(cutoff)
    result.setForceGroup(native.getForceGroup())
    for name in names:
        result.addPerParticleParameter(name)
    for name, value in globals_.items():
        result.addGlobalParameter(name, value)
    particle_rows = np.column_stack(columns) if columns else np.empty((n, 0))
    for row in particle_rows:
        result.addParticle(row.tolist())
    for first, second in sorted(exceptions):
        result.addExclusion(first, second)
    return FusedPairDefinition(result, native_index, custom_indices,
                               native.getUseDispersionCorrection(), tuple(tail_definitions))


def convert_to_plugin(system, box_safety_factor=1.0, method="Prolate"):
    """Replace all stock pair forces with one cuPo force; retain sparse terms."""
    import openmm as mm
    import nv_openmm_plugin as nv

    definition = fuse_pair_definition(system)
    native = system.getForce(definition.native_index)
    # The native analytic reference tail and each stock Custom correction retain
    # their own semantics. Grouped LRC excludes self pairs, whereas an ungrouped
    # expression-mask LRC includes them; integrating the fused expression would
    # silently change those weights. Tail-only definitions schedule no traversal.
    base = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(native))
    combined = nv.NonbondedForce(base, box_safety_factor, getattr(nv.NonbondedForce, method))
    # This benchmark converter preserves the stock Hamiltonian. Stock's native
    # LJ tail uses global defaults, while cuPo normally follows live parameters.
    # Keep that native policy explicitly; custom expression tails remain live.
    combined.setDispersionCorrectionFollowsParameters(False)
    combined.setCutoffInteraction(definition.expression, definition.native_tail)
    for correction in definition.custom_tails:
        combined.addExpressionTailCorrection(correction)
    removed = sorted((definition.native_index, *definition.custom_indices), reverse=True)
    for index in removed:
        system.removeForce(index)
    system.addForce(combined)
    return combined
