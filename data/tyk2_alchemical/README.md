# Frozen TYK2 alchemical benchmarks

These are parameterized, solvated TYK2 protein–ligand systems, not an alchemical
label placed on water or a synthetic BACE topology. Loading and benchmarking
requires only OpenMM and NumPy. OpenFE, OpenFF, RDKit and force-field files are
used **only to regenerate** the checked-in artifacts.

The reference for each benchmark is its stock OpenMM `system.xml.gz`, including
its original sparse bonded and exception terms, switches, interaction groups,
and dispersion corrections. `positions.npz` contains prepared coordinates, box
vectors, and velocities in nm and ps. `manifest.json` records checksums, complete
global-parameter values, atom counts, force policies, and the ligand/dummy indices.
`expressions.json` exposes the stock pair expressions without decompressing XML.
The benchmark names are `tyk2_rbfe_dodec` and `tyk2_abfe_dodec`; `_dodec`
identifies their rhombic dodecahedral boxes.

## Physical system and provenance

The protein, ligand coordinates and antechamber AM1-BCC charges come from the
[OpenFE benchmark TYK2 dataset](https://github.com/OpenFreeEnergy/openfe-benchmarks/tree/f85850eb2c4e53b6bbff563d882f4b6707a69d18/openfe_benchmarks/data/benchmark_systems/jacs_set/tyk2).
The network edge **ejm_31 → ejm_42** maps 31 atoms, replacing one old-only hydrogen
with a four-atom methyl group. This modest dummy region avoids changing the
length of a mapped constrained X–H bond. The generator recovers the actual
network mapping and mapper settings from the pinned network; it does not infer
a new map. [PROVENANCE.md](PROVENANCE.md) records source permalinks, hashes,
protein preparation, the MIT notice and the generation settings.
Raw external sources are downloaded into the build directory, not vendored.

Parameterization uses Amber14 protein, TIP3P water and OpenFF Sage 2.2.1 ligand
parameters, retaining the dataset's AM1-BCC partial charges exactly. The complex
is solvated with 1 nm padding in a rhombic dodecahedron, neutralized, and given
0.15 M salt. Both endpoint systems share the same protein, water and ion atoms.
The ABFE system has 34,078 atoms; the hybrid RBFE system has 34,082 atoms,
including five unique atoms across its endpoints.

All systems use periodic PME, a 0.9 nm pair cutoff, and a 0.8–0.9 nm LJ switch.
The cutoff follows OpenFE's default adopted in
[PR #1523](https://github.com/OpenFreeEnergy/openfe/pull/1523), with a 0.1 nm
switching width. The 1 nm solvation padding is an independent setting.
Hydrogen masses are repartitioned to 3 amu except rigid-water hydrogens. H-bond
constraints and rigid water allow a 4 fs benchmark step. Native exceptions use
the default nonperiodic convention, as do the sparse factory-generated
CustomBondForce terms. Reciprocal exclusion corrections in cuPo always use
minimum-image distances, independently of the native exception flag. The
converter preserves either source flag; these fixtures exercise its `False`
default and the raw-distance bonded kernels. The generated force groups
are preserved. Full-system comparisons evaluate **all force groups**; the fused
cuPo conversion cannot retain separate queries of the original pair force groups.
No barostat is serialized, and each system has one CMMotionRemover at frequency
100. The benchmark selects NVT or adds its own NPT barostat.

The exception-distance revision changes only the serialized native flag and
its metadata. Positions, velocities and historical preparation checkpoints are
retained from preparation with periodic native exceptions. Every native
exception has identical raw and minimum-image displacement in each stored
frame; preparation was not rerun. See [PROVENANCE.md](PROVENANCE.md).

## The Hamiltonians and their midpoint definitions

“Lambda 0.5” describes the selected alchemical window. It does **not** mean setting
every independent global parameter to 0.5. The complete vectors are in the
manifest and the XML defaults. In particular, partially charging a dummy atom
while softening its repulsion can make the unsoftened Coulomb attraction singular.
The fixtures use physically staged schedules with active softcore sterics.

| Case | Pair model | Selected window |
| --- | --- | --- |
| `rbfe` | Actual pinned OpenFE HybridTopologyFactory, Gapsys v2 softcore, alpha 0.85 | OpenFE's built-in **namd** schedule at master lambda 0.5: steric insertion 0.75, deletion 0.25; unique inserted/deleted charges off; core and bonded lambdas 0.5 |
| `abfe` | Actual pinned openmmtools AbsoluteAlchemicalFactory, exact PME, electrostatic annihilation and steric decoupling, alpha 0.5 | Steric coupling 0.5 **after charge annihilation**, electrostatic coupling 0.0; intraligand sterics stay coupled |

RBFE uses the user's pinned [OpenFE factory and lambda protocol](https://github.com/OpenFreeEnergy/openfe/tree/1d4610ba845f6fd7b9a107b262a345d1a063d89a/src/openfe/protocols/openmm_rfe/_rfe_utils).
It explicitly sets `interpolate_old_and_new_14s=True`, matching the pinned
protocol's `turn_off_core_unique_exceptions=True` setting. The lower-level
factory defaults this option to false. For this edge, the protocol setting
creates **13 sparse interpolated exception bonds** and zeroes the corresponding
native exception interactions. The generator and fixture tests require this
nonempty coverage.
The built-in `namd` schedule exercises softcore at master 0.5; the default OpenFE
schedule has fully inserted/not-yet-deleted sterics at that master lambda.
ABFE uses the user's pinned [openmmtools factory](https://github.com/choderalab/openmmtools/blob/ba4031da49981fe9e8b0bcec010ab53e18a03ba2/openmmtools/alchemy/alchemy.py).
Its staged steric midpoint follows OpenFE's ABFE convention: the OpenFE decoupling
value for electrostatics is 1, corresponding to a factory coupling value of 0.

Native dispersion corrections and CustomNonbondedForce long-range corrections
are enabled. **RBFE explicitly selects `use_dispersion_correction=True`, overriding
OpenFE's default of false** to exercise lambda-dependent tail corrections; the
stock reference and cuPo use the same selected policy. ABFE retains its factory's
default correction policy. The custom
force groups are retained in the stock reference and lowered to per-particle
region flags/expression predicates for cuPo's single traversal. Tail corrections
retain their original group and switch policies independently of that lowering.

The fixture converter simplifies identical or disjoint group masks. When cheaper,
it replaces membership flags with one per-particle class code and a sparse table
of pair multiplicities, expressed using `delta()`. This preserves overlapping and
duplicate groups without a separate traversal. Class codes describe fixed topology;
changing the source groups requires reconversion. The compiler treats these codes
as ordinary parameters and makes no Boolean assumptions about arbitrary user data.
The frozen stock XML and the source-specific tail definitions remain the reference.

These are **fixed-window Hamiltonian and complete-MD performance benchmarks**.
They are not complete ABFE/RBFE production protocols: there is no Boresch restraint
setup, standard-state correction, endpoint sampling, or free-energy estimator.
The short preparation provides finite, MD-ready starting coordinates; it does
not establish thermodynamic equilibration or binding free-energy convergence.

## Loading and benchmarks

To load the frozen reference or its fused cuPo equivalent from the repository:

```python
from data.tyk2_alchemical import load_tyk2_alchemical, convert_to_plugin

fixture = load_tyk2_alchemical("tyk2_rbfe_dodec")  # also accepts the short case key "rbfe"
stock_system = fixture.create_system()
cupo_system = fixture.create_system()
combined_force = convert_to_plugin(cupo_system, method="Prolate")
# Use fixture.positions, fixture.box and fixture.velocities (nm and ps).
# The System defaults already contain the complete selected lambda vector.
```

The plugin's test stages, `openmm_unit` and `openmm_bench`, collect these fixtures
by themselves. To run just their correctness checks and end-to-end benchmarks
with the repository test runner:

```sh
./run_test openmm_unit -k tyk2_alchemical
./run_test openmm_bench -k 'tyk2_rbfe_dodec or tyk2_abfe_dodec'
```

The benchmark covers single and mixed precision, NVT and NPT. It compares the
cuPo force/integrator/barostat with stock OpenMM using the same prepared state;
the measured MD steps include neighbor-list and cell-list maintenance.

## Preparation and regeneration

Each system was minimized at its specified window (10 kJ/mol/nm tolerance,
maximum 1,000 iterations), then run for 5 ps at 2 fs and 20 ps at 4 fs with
LangevinMiddle at 300 K, friction 1/ps, and constraint tolerance 1e-6. Seeds are
20261025 and 20261026. The recorded stock CUDA mixed-precision runs
have finite energies and forces throughout; `preparation.json` records energy
and maximum force at each checkpoint. GPU preparation is reproducible in its
inputs and random seeds, but is not promised to be bit-identical across devices
or OpenMM/CUDA versions.

The fixtures were made by a generator script, not part of this tree, which
downloads and verifies the pinned inputs and factory/protocol files; neither is
copied into the fixtures. See [PROVENANCE.md](PROVENANCE.md) for the inputs,
per-system settings, source permalinks and the original generation/preparation
versions, which are also retained in `manifest.json`.
