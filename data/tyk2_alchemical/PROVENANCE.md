<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TYK2 alchemical fixture provenance

The `abfe/` and `rbfe/` directories contain the prepared OpenMM systems,
coordinates, velocities and expression metadata used by the tests and the
benchmarks. The original protein, ligand and network files are **not vendored**:
loading uses only the frozen fixtures and needs neither those inputs nor
OpenFE/OpenFF. Their benchmark identifiers are `tyk2_rbfe_dodec` and
`tyk2_abfe_dodec`, identifying the rhombic dodecahedral box geometry.

The fixtures were made by a generator script that downloads the pinned inputs
and factories below, checks their hashes, builds and prepares each system and
writes the files here with [`manifest.json`](manifest.json). The script is not
part of this tree; this file records everything it takes and every setting it
uses.

## Molecular inputs

All molecular inputs come from **OpenFreeEnergy/openfe-benchmarks**, commit
`f85850eb2c4e53b6bbff563d882f4b6707a69d18`:

| Input | Immutable source |
| --- | --- |
| Prepared TYK2 protein | [protein.pdb](https://github.com/OpenFreeEnergy/openfe-benchmarks/blob/f85850eb2c4e53b6bbff563d882f4b6707a69d18/openfe_benchmarks/data/benchmark_systems/jacs_set/tyk2/protein.pdb) |
| Ligand coordinates and AM1-BCC charges | [ligands_antechamber_am1bcc.sdf](https://github.com/OpenFreeEnergy/openfe-benchmarks/blob/f85850eb2c4e53b6bbff563d882f4b6707a69d18/openfe_benchmarks/data/benchmark_systems/jacs_set/tyk2/ligands_antechamber_am1bcc.sdf) |
| Selected RBFE edge and atom mapping | [industry_benchmarks_network.json](https://github.com/OpenFreeEnergy/openfe-benchmarks/blob/f85850eb2c4e53b6bbff563d882f4b6707a69d18/openfe_benchmarks/data/benchmark_systems/jacs_set/tyk2/industry_benchmarks_network.json) |
| Upstream protein preparation | [PREPARATION_DETAILS.md](https://github.com/OpenFreeEnergy/openfe-benchmarks/blob/f85850eb2c4e53b6bbff563d882f4b6707a69d18/openfe_benchmarks/data/benchmark_systems/jacs_set/tyk2/PREPARATION_DETAILS.md) |
| Dataset license | [MIT LICENSE](https://github.com/OpenFreeEnergy/openfe-benchmarks/blob/f85850eb2c4e53b6bbff563d882f4b6707a69d18/LICENSE) |

The generator selects **ejm_31 → ejm_42**, the existing `mol0 → mol3` edge in
the network, rather than running a new mapper. The ligand-local map is
`0..29 → 0..29` and `31 → 30`: 31 mapped atoms, one old-only atom (index 30)
and four new-only atoms (indices 31–34). The original mapper annotations are
recovered from the network. The ABFE region is the 32-atom ejm_31 ligand; the
hybrid RBFE region is the 36-atom union of both ligands.

Ligand atom order, coordinates and `atom.dprop.PartialCharge` values are retained.
The generator does not rerun AM1-BCC. Upstream preparation renamed the protein
caps to ACE/NME and repaired their adjacent bonds; regeneration starts with that
already prepared protein.

SHA256 values below apply to the original downloaded bytes, before compression
or ligand extraction. Download URLs and hashes are also recorded in
[`manifest.json`](manifest.json), under `provenance.source_files`, and are checked
by the generator, including when reusing cached downloads.

```text
protein.pdb
  eb5b4eb0170997d781e878c25a69f5f4ecdd90f5c70797e6d32f0d44024c203b
ligands_antechamber_am1bcc.sdf
  7cd476fa7f2aa3dfb618c10fd802e637eaf7a2b708ef0961d2ed83f67d1ccc2c
industry_benchmarks_network.json
  cc06319903df6d1ec02d2d1a476491ff00aa82b1185624b6760b52583d370fc3
PREPARATION_DETAILS.md
  d9c02e4d69554e91aec68a8423c01dc78524cfe9e434edfabf5263320fef3bc2
LICENSE
  bcf849f59c95a7f80b50f38b34c1d64278a37a13e4b874ac5f7eaf1aff57b21f
```

## Alchemical definitions

The generator downloads the three factory/protocol files below at
exact commits into `--build-dir/factories/` and verifies their SHA256 values before
import. Their contents are not vendored either.

| Definition | Immutable source | SHA256 |
| --- | --- | --- |
| RBFE hybrid topology factory | [OpenFE relative.py](https://github.com/OpenFreeEnergy/openfe/blob/1d4610ba845f6fd7b9a107b262a345d1a063d89a/src/openfe/protocols/openmm_rfe/_rfe_utils/relative.py) | `2a407297038ea90663f03a0d1aec1907076d51a5d67bb202317c741b8c06de9d` |
| RBFE lambda protocol | [OpenFE lambdaprotocol.py](https://github.com/OpenFreeEnergy/openfe/blob/1d4610ba845f6fd7b9a107b262a345d1a063d89a/src/openfe/protocols/openmm_rfe/_rfe_utils/lambdaprotocol.py) | `1a5c4b4a359b511027c52490b4de1d2b52e6d51eac3e1428e1931345c3747597` |
| ABFE factory | [openmmtools alchemy.py](https://github.com/choderalab/openmmtools/blob/ba4031da49981fe9e8b0bcec010ab53e18a03ba2/openmmtools/alchemy/alchemy.py) | `557edff2c0e4006892bf46fdd1db720ba3ce4d7f9bc48d4d535977a96711f91b` |

RBFE's exception policy follows the pinned OpenFE
[settings](https://github.com/OpenFreeEnergy/openfe/blob/1d4610ba845f6fd7b9a107b262a345d1a063d89a/src/openfe/protocols/openmm_rfe/equil_rfe_settings.py)
and [protocol setup](https://github.com/OpenFreeEnergy/openfe/blob/1d4610ba845f6fd7b9a107b262a345d1a063d89a/src/openfe/protocols/openmm_rfe/hybridtop_units.py).
The ABFE staged midpoint follows `LambdaSettings` in
[OpenFE 1.13.0](https://github.com/OpenFreeEnergy/openfe/blob/858783fa1d8c99f482cee5f4ac20985c0b58668b/src/openfe/protocols/openmm_afe/equil_afe_settings.py):
electrostatics are annihilated before the steric decoupling midpoint.

The generator creates the following systems from the shared endpoints:

| System | Construction and selected window |
| --- | --- |
| **RBFE** | OpenFE `HybridTopologyFactory`; Gapsys v2 sterics with alpha 0.85; `interpolate_old_and_new_14s=True`; explicit `use_dispersion_correction=True`. The `namd` protocol at master lambda 0.5 gives steric insertion/deletion 0.75/0.25, unique-atom charges off, and core/bonded lambdas 0.5. |
| **ABFE** | openmmtools `AbsoluteAlchemicalFactory`, exact PME; electrostatic annihilation and steric decoupling of ejm_31, softcore alpha 0.5. Factory coupling values are `lambda_electrostatics=0` and `lambda_sterics=0.5`; intraligand sterics stay coupled. |

RBFE retains 13 interpolated sparse exception bonds. These selected windows are
not obtained by setting every global parameter to 0.5. Full global vectors are in
the manifest and serialized System defaults.

## Generation and preparation settings

Generation uses Amber14 protein, TIP3P water and Sage 2.2.1 ligand parameters,
1 nm dodecahedral padding, neutralization and 0.15 M salt. Both endpoints share
the environment. PME uses a 0.9 nm cutoff and LJ switching from 0.8 to 0.9 nm,
following the cutoff default adopted in
[OpenFE PR #1523](https://github.com/OpenFreeEnergy/openfe/pull/1523).
The 1 nm solvation padding is independent of that cutoff.
Hydrogen masses are 3 amu except rigid-water hydrogens; H-bond constraints and
rigid water support the 4 fs benchmark timestep. Native exceptions use the
default nonperiodic convention; sparse custom bonded exception terms retain their
factory nonperiodic convention. Correction policies and other deliberate departures
from default OpenFE protocols are described in the [fixture README](README.md).

For each case, preparation minimizes with tolerance 10 kJ/mol/nm and at most
1,000 iterations, then runs 5 ps at 2 fs and 20 ps at 4 fs with LangevinMiddle,
300 K, friction 1/ps and constraint tolerance 1e-6. The generation seed is
20261025; preparation seeds in RBFE/ABFE order are
20261025/20261026. Checkpoints are recorded in each `preparation.json`.

The cutoff revision reused the original endpoint parameters, atom mapping and
solvated box, changing only the cutoff and switch before rebuilding the two
alchemical Hamiltonians through the pinned factories. Both preparations
were repeated at the new settings. The generator rejects cached endpoint or
prepared systems with stale cutoff/switch or native exception-distance settings
rather than publishing metadata that disagrees with their XML.

The subsequent exception-distance revision changes the serialized native flag
from `True` to `False` in both systems, with matching metadata and checksums.
Positions, velocities, force parameters and preparation checkpoints are retained;
the original preparation used `True` and was not rerun. A CPU audit of the stored
frames found exactly identical raw and reduced-box minimum-image displacements
for all 55,094 RBFE and 55,066 ABFE exception pairs. This verifies
the saved-state geometry, without asserting identity of the original preparation
trajectories under the revised setting. Future generation uses the default
nonperiodic convention and rejects old periodic caches.

The original CPU generation used Python 3.12, OpenFE 1.13.0, OpenMM 8.4.0,
openmmtools 0.26.0, openmmforcefields 0.15.1, OpenFF Toolkit 0.19.0,
RDKit 2025.09.6 and NumPy 2.5.4. Original GPU preparation used OpenMM 8.4,
CUDA mixed precision with deterministic forces, and an RTX 5090. Version records
are retained in `manifest.json`.
The optional environment pins the main generation tools; it is not a full
transitive dependency lock. GPU trajectories and artifact bytes are not promised
to match across devices or software builds.

After replacing a reviewed set of fixtures, run:

```sh
./run_test openmm_unit -k tyk2_alchemical
./run_test openmm_bench -k 'tyk2_rbfe_dodec or tyk2_abfe_dodec'
```

These fixtures cover fixed-window Hamiltonians and MD performance. They do not
include Boresch restraints, standard-state corrections, endpoint sampling or
free-energy estimators.

## Dataset license

The upstream dataset is MIT licensed. Its notice is retained here for the
derived fixtures even though the raw source files are no longer vendored:

```text
MIT License

Copyright (c) 2022 Irfan Alibay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
