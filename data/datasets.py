# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark dataset loader.

Each subfolder of the repository ``data/`` directory holds one solvated
molecular system.  The vendored input every system ships with is
``solvated.pdb.gz`` — a gzip-compressed PDB carrying the topology,
coordinates, and (via its ``CRYST1`` record) the periodic box.

Two systems are not in ``data/``: apoa1 and dhfr are OpenMM's own benchmark
files, read from the installed OpenMM or downloaded from OpenMM's repository
(``nv_openmm_plugin._benchmark_systems``, see _benchmark_systems()), and loaded
by the same names.

``load_dataset(name)`` parses that file into a :class:`Dataset`, which
exposes the coordinates, box vectors, and the underlying OpenMM
``PDBFile``.  :meth:`Dataset.parameterize_nonbonded` then runs the system
through an OpenMM ``ForceField`` to recover per-atom charges, Lennard-Jones
``sigmas``/``epsilons``, and the 1-2 / 1-3 exclusion pairs — defaulting to the
TIP3P-FB water model and the AMBER99SB-ILDN protein force field.

Units follow OpenMM's MD convention throughout: nanometers for lengths,
elementary charge for charges, kJ/mol for energies.
"""
from __future__ import annotations

import gzip
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# This module lives in the data/ directory itself; the per-system folders
# (water_100k/, p38/, ...) are its siblings.
_DATA_DIR = Path(__file__).resolve().parent
_REPO_DIR = _DATA_DIR.parent

# The plugin's module that finds OpenMM's benchmark files, in this tree: the
# repository and the plugin's source tarball both have it here.
_BENCHMARK_SYSTEMS_FILE = (_REPO_DIR / "plugins" / "openmm" / "python" / "nv_openmm_plugin"
                           / "_benchmark_systems.py")


def _benchmark_systems():
    """The plugin's _benchmark_systems module, loaded from its file in this tree.

    By path, and not with `import nv_openmm_plugin._benchmark_systems`, for two
    reasons. The package's __init__ loads the plugin's libraries into OpenMM, which
    cupo's own tests must not depend on and need not have built; the module itself
    imports only the standard library. And an installed plugin may be another
    version than this tree's: the tests read the files this tree's module names.
    The sample, which needs the plugin anyway, imports the installed module."""
    name = "_cupo_data_benchmark_systems"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _BENCHMARK_SYSTEMS_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


_benchmarks = _benchmark_systems()
# OpenMM's benchmark systems, by name: apoa1 and dhfr.
OPENMM_SYSTEMS = _benchmarks.SYSTEMS


def repo_config(repo_dir=None):
    """cupo-repo.ini at the repository root, read with configparser; an empty
    configuration when there is no such file."""
    import configparser
    config = configparser.ConfigParser()
    config.read(Path(repo_dir or _REPO_DIR) / "cupo-repo.ini")
    return config


def extra_dirs(repo_dir=None):
    """Extra dataset directories, from cupo-repo.ini at the repository root when
    present: [data] extra_dirs, one path per line, relative to the file."""
    repo_dir = Path(repo_dir or _REPO_DIR)
    value = repo_config(repo_dir).get("data", "extra_dirs", fallback="")
    return [repo_dir / line.strip() for line in value.splitlines() if line.strip()]


# Where the benchmark files are downloaded to, for a run from this tree.
_OPENMM_BENCHMARKS_CACHE = _REPO_DIR / ".tmp" / "openmm-benchmarks"


def _roots():
    """data/, then the extra dataset directories (extra_dirs)."""
    return [_DATA_DIR] + [d for d in extra_dirs() if d.is_dir()]


@dataclass(frozen=True)
class NonbondedParameters:
    """Per-atom nonbonded parameters plus the short-range exclusion list.

    Attributes:
        charges:    (N,) fp64 — partial charges in elementary charge (e).
        sigmas:     (N,) fp64 — Lennard-Jones σ in nanometers.
        epsilons:   (N,) fp64 — Lennard-Jones ε in kJ/mol.
        exclusions: (M, 2) int32 — fully-excluded (1-2 and 1-3) atom-index
                    pairs, i.e. the bonded / angle pairs OpenMM removes from
                    the nonbonded sum entirely.  Scaled 1-4 interactions are
                    NOT included here.
    """
    charges: np.ndarray
    sigmas: np.ndarray
    epsilons: np.ndarray
    exclusions: np.ndarray


@dataclass(frozen=True)
class Dataset:
    """A solvated molecular system loaded by :func:`load_dataset`.

    Attributes:
        name:   the name it was loaded by: a folder name under ``data/`` (or an
                extra dataset directory), or one of OpenMM's benchmark systems.
        coords: (N, 3) fp64 — atom positions in nanometers, as stored in the
                PDB (not wrapped into the primary cell).
        box:    (3, 3) fp64 — periodic box vectors in nanometers (rows are the
                lattice vectors a, b, c).
        pdb:    the OpenMM ``app.PDBFile`` the coordinates and topology came
                from; kept so callers can reach the topology, elements, etc.
    """
    name: str
    coords: np.ndarray
    box: np.ndarray
    pdb: "object"  # openmm.app.PDBFile (imported lazily in load_dataset)

    @property
    def topology(self):
        """The OpenMM ``Topology`` (atoms, residues, bonds, box)."""
        return self.pdb.topology

    @property
    def n_atoms(self) -> int:
        return self.coords.shape[0]

    @property
    def positions(self):
        """Positions as a numpy-backed ``Quantity``, ready for setPositions.

        Use this rather than ``dataset.pdb.getPositions()``. The two carry the
        same coordinates, but the PDBFile hands back a Quantity wrapping a
        LIST of one Vec3 object per atom, and SWIG converts those one at a time
        on the way into the C++ layer:

            Quantity(list of Vec3)   95.89 ms      92224 atoms (apoa1)
            Quantity(numpy array)     0.24 ms      same coordinates

        That is not just slow. A benchmark that calls setPositions before every
        timed sample precedes each measurement with ~100 ms of Python object
        allocation, which evicts the cache and TLB entries the measured call
        then needs -- OpenMM's getForces ends by scattering N atoms in permuted
        order. On the CI machine that turned a clean measurement into a bimodal
        one: slow-sample fraction 0.32, round-to-round spread 19.9%, versus
        0.007 and 0.9% once the positions were numpy-backed.

        Built from `coords`, which load_dataset already read with asNumpy=True,
        so this costs nothing beyond attaching the unit.
        """
        from openmm import unit
        return self.coords * unit.nanometer

    def parameterize_nonbonded(
        self,
        protein_forcefield: str = "amber99sbildn.xml",
        water_model: str = "tip3pfb.xml",
        *,
        nonbonded_cutoff: float = 0.9,
    ) -> NonbondedParameters:
        """Assign nonbonded parameters via an OpenMM ``ForceField``.

        Builds a PME ``System`` from this dataset's topology and reads back the
        per-atom charges / σ / ε and the auto-generated exclusion list.

        Args:
            protein_forcefield: force-field XML for the (bio)polymer and ions.
                Defaults to AMBER99SB-ILDN.
            water_model: force-field XML for water.  Defaults to TIP3P-FB.
            nonbonded_cutoff: real-space cutoff in nm passed to
                ``createSystem`` — it only affects which method-specific
                parameters OpenMM computes, not the per-atom values returned.

        Returns:
            A :class:`NonbondedParameters` with ``charges`` (e), ``sigmas`` (nm),
            ``epsilons`` (kJ/mol), and the (M, 2) int32 1-2 / 1-3 ``exclusions``.
        """
        # Imported here (not at module top) so callers that only need
        # coords/box from load_dataset don't pay the OpenMM import cost.
        import openmm
        from openmm import app, unit

        ff = app.ForceField(protein_forcefield, water_model)
        system = ff.createSystem(
            self.pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=nonbonded_cutoff * unit.nanometer,
            constraints=None,
            rigidWater=False,
            residueTemplates=template_pins(self.pdb.topology),
        )
        nb = next(f for f in system.getForces()
                  if isinstance(f, openmm.NonbondedForce))

        n = nb.getNumParticles()
        charges = np.empty(n, dtype=np.float64)
        sigmas = np.empty(n, dtype=np.float64)
        epsilons = np.empty(n, dtype=np.float64)
        for i in range(n):
            q, sig, eps = nb.getParticleParameters(i)
            charges[i] = q.value_in_unit(unit.elementary_charge)
            sigmas[i] = sig.value_in_unit(unit.nanometer)
            epsilons[i] = eps.value_in_unit(unit.kilojoule_per_mole)

        # OpenMM's createSystem auto-generates one exception per 1-2, 1-3,
        # and 1-4 pair.  The 1-2 / 1-3 pairs are *fully* excluded
        # (chargeprod == 0 and ε == 0); scaled 1-4 pairs keep non-zero
        # parameters.  Select the fully-excluded set — these are exactly the
        # pairs dropped from the real-space nonbonded sum.
        qq_unit = unit.elementary_charge ** 2
        excl = []
        for k in range(nb.getNumExceptions()):
            i, j, qq, _sig, eps = nb.getExceptionParameters(k)
            if (qq.value_in_unit(qq_unit) == 0.0
                    and eps.value_in_unit(unit.kilojoule_per_mole) == 0.0):
                excl.append((i, j))
        exclusions = np.asarray(excl, dtype=np.int32).reshape(-1, 2)

        return NonbondedParameters(
            charges=charges, sigmas=sigmas, epsilons=epsilons,
            exclusions=exclusions,
        )


# Residue names whose force-field template cannot be recovered from connectivity
# and has to be taken from the residue name instead.
#
# OpenMM matches templates by graph isomorphism and treats the residue name as a
# hint only.  GLYCAM's sugars are isomorphic to one another — ``4GB`` has the
# same connectivity as ``4EA``, ``4KA``, ``4LA`` and 25 others — so the matcher
# cannot choose between them and raises "Multiple non-identical matching
# templates".  The name is the only thing that separates them, and the PDB
# carries it.
#
# Enumerated per residue rather than "pin every residue to its own name":
# protein termini are named ``ALA``/``GLY`` while their templates are
# ``NALA``/``CALA``, and the matcher resolves those correctly from the graph.
# Pinning those by name would actively break them.
_AMBIGUOUS_TEMPLATE_RESIDUES = frozenset({"0GB", "4GB", "ROH"})


def template_pins(topology) -> dict:
    """``residueTemplates`` for ``createSystem``, covering only what needs it.

    Empty for every system whose residues the matcher can resolve on its own, so
    it is safe to pass unconditionally.
    """
    return {r: r.name for r in topology.residues()
            if r.name in _AMBIGUOUS_TEMPLATE_RESIDUES}


def available_datasets() -> list[str]:
    """Return the sorted names load_dataset accepts: the folder names under
    ``data/`` and the extra dataset directories (extra_dirs), and OpenMM's
    benchmark systems, which are always available (found or downloaded when
    loaded)."""
    return sorted({p.name for root in _roots() if root.is_dir() for p in root.iterdir()
                   if p.is_dir() and (p / "solvated.pdb.gz").is_file()}
                  | set(OPENMM_SYSTEMS))


def dataset_dir(name: str) -> Path:
    """The folder of dataset ``name``: ``data/<name>``, or ``<dir>/<name>`` in an
    extra dataset directory (extra_dirs). For a name that is in neither,
    ``data/<name>``, where a new dataset goes."""
    for root in _roots():
        if (root / name / "solvated.pdb.gz").is_file():
            return root / name
    return _DATA_DIR / name


def dataset_file(name: str) -> Path:
    """The PDB file of dataset ``name``: OpenMM's benchmark file for apoa1 and
    dhfr (openmm_benchmark_file, which may download it), else
    ``dataset_dir(name)/solvated.pdb.gz``, which may not exist."""
    if name in OPENMM_SYSTEMS:
        return _benchmarks.openmm_benchmark_file(name, _OPENMM_BENCHMARKS_CACHE)
    return dataset_dir(name) / "solvated.pdb.gz"


def load_dataset(name: str) -> Dataset:
    """Load a solvated system: ``data/<name>/solvated.pdb.gz``, or for apoa1 and
    dhfr OpenMM's own file (:func:`dataset_file`).

    Args:
        name: a subfolder of the repo ``data/`` directory, e.g. ``"water_100k"``
            or ``"p38"``, or ``"apoa1"`` or ``"dhfr"``.  See
            :func:`available_datasets`.

    Returns:
        A :class:`Dataset` with coordinates (nm), box vectors (nm), and the
        parsed OpenMM ``PDBFile``.
    """
    from openmm import app, unit

    pdb_path = dataset_file(name)
    if not pdb_path.is_file():
        choices = available_datasets()
        hint = f"  Available: {', '.join(choices)}" if choices else ""
        raise FileNotFoundError(
            f"No dataset {name!r}: {pdb_path} does not exist.{hint}")

    opener = gzip.open if pdb_path.suffix == ".gz" else open
    with opener(pdb_path, "rt") as fh:
        pdb = app.PDBFile(fh)

    coords = np.asarray(
        pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )

    box_vectors = pdb.topology.getPeriodicBoxVectors()
    if box_vectors is None:
        raise ValueError(
            f"Dataset {name!r}: {pdb_path} has no periodic box (missing "
            f"CRYST1 record); box vectors are required.")
    box = np.asarray(
        box_vectors.value_in_unit(unit.nanometer), dtype=np.float64)

    return Dataset(name=name, coords=coords, box=box, pdb=pdb)
