<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepared TYK2 RBFE system

`system.xml.gz` contains a stock OpenMM hybrid System for **ejm_31 → ejm_42**
bound to TYK2. `positions.npz` contains `positions_nm`, `box_vectors_nm` and
`velocities_nm_per_ps`. Loading these files requires OpenMM and NumPy; OpenFE
and the parameterization tools are unnecessary at runtime.

This is a prepared alchemical window for force comparisons and short dynamics.
It does not provide a complete relative binding free-energy calculation,
endpoint sampling or a free-energy estimator.

## Molecular inputs and construction

Protein, ligand coordinates, AM1-BCC charges and the existing ligand atom map
come from the MIT-licensed [OpenFE TYK2 benchmark dataset](https://github.com/OpenFreeEnergy/openfe-benchmarks/tree/f85850eb2c4e53b6bbff563d882f4b6707a69d18/openfe_benchmarks/data/benchmark_systems/jacs_set/tyk2).
The [network edge](https://github.com/OpenFreeEnergy/openfe-benchmarks/blob/f85850eb2c4e53b6bbff563d882f4b6707a69d18/openfe_benchmarks/data/benchmark_systems/jacs_set/tyk2/industry_benchmarks_network.json)
maps 31 ligand atoms, with one old-only hydrogen and four new-only atoms.
Dataset partial charges are retained without recalculation.

The 34,082-atom system uses Amber14 protein, TIP3P water and OpenFF Sage 2.2.1
ligand parameters. It has a rhombic dodecahedral box, 1 nm solvation padding,
neutralization and 0.15 M salt. PME uses a 0.9 nm cutoff, with LJ switching
from 0.8 to 0.9 nm. Hydrogen masses are repartitioned to 3 amu except in rigid
water; H-bond constraints and rigid water support a 4 fs timestep.

The system was constructed with the pinned OpenFE
[HybridTopologyFactory](https://github.com/OpenFreeEnergy/openfe/blob/1d4610ba845f6fd7b9a107b262a345d1a063d89a/src/openfe/protocols/openmm_rfe/_rfe_utils/relative.py)
and [lambda protocol](https://github.com/OpenFreeEnergy/openfe/blob/1d4610ba845f6fd7b9a107b262a345d1a063d89a/src/openfe/protocols/openmm_rfe/_rfe_utils/lambdaprotocol.py).
It uses Gapsys v2 softcore sterics with alpha 0.85 and
`interpolate_old_and_new_14s=True`, producing 13 interpolated exception bonds.
Native dispersion and custom long-range corrections are enabled; enabling
dispersion correction explicitly overrides the factory's default.

The XML defaults select the `namd` schedule at master lambda 0.5:

| Global parameters | Value |
| --- | --- |
| `lambda_bonds`, `lambda_angles`, `lambda_torsions` | 0.5 |
| `lambda_electrostatics_core`, `lambda_sterics_core` | 0.5 |
| `lambda_electrostatics_delete`, `lambda_electrostatics_insert` | 1.0, 0.0 |
| `lambda_sterics_insert`, `lambda_sterics_delete` | 0.75, 0.25 |
| `softcore_alpha` | 0.85 |

Thus both unique-atom charge sets are off while softcore sterics remain active.
Preparation minimized at this window, then ran 5 ps at 2 fs and 20 ps at 4 fs
using LangevinMiddle at 300 K and friction 1/ps. This short preparation supplies
an MD starting state and does not establish thermodynamic equilibration.
The saved System has no barostat. Native and custom bonded exceptions use
nonperiodic distances. The native flag was changed after preparation; the
coordinates were retained because all native exception displacements in this
saved frame are identical with raw and minimum-image distances.

## Payload checksums

SHA256 of the files shipped beside this README:

```text
5e1c2312d6c547883acb9e678d81e22c7e2b4082f5455a638edca7e62af84b64  system.xml.gz
03ed0c2f47e2ef6cad0f8fbd74623aeb8708624ae082700ef32ea907edeb15ee  positions.npz
```

## Dataset license notice

The following [upstream MIT notice](https://github.com/OpenFreeEnergy/openfe-benchmarks/blob/f85850eb2c4e53b6bbff563d882f4b6707a69d18/LICENSE)
is retained for these derived molecular fixtures:

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
