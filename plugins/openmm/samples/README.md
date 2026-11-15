<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Samples

Each directory is one sample with its own `run`. `run_docker` and `common.sh` are
shared: `run_docker` builds and enters the container, and `common.sh` is what every
`run` sources to make `nv_openmm_plugin` importable, from a wheel, a conda package
or an already-built `python/` tree beside this directory.

| sample | what it shows |
|---|---|
| `solvated_systems` | the plugin end to end on apoa1, bace and dhfr, in NVT and NPT |
| `alchemical` | TYK2 RBFE: stock PME and custom softcore LJ combined in one plugin `NonbondedForce`, with live globals and particle/exception offsets |

From the repository's root, after building the plugin with
`./run_docker plugins/openmm/run_build`:

```bash
./run_docker                                              # solvated_systems, the default
./run_docker plugins/openmm/samples/alchemical/run
```

The [alchemical sample](alchemical/sample.py) takes a stock `NonbondedForce` and
`CustomNonbondedForce` and returns one plugin `NonbondedForce`. It combines
ordinary LJ with the source softcore expression, converts interaction groups to
particle flags, and retains live offsets. The default constructor selects Prolate
electrostatics and automatic box safety. Matching source switching and
tail-correction settings are required; the built-in switch and one ungrouped
tail correction apply to the combined LJ expression. Sparse exception bonds and
other bonded forces remain in OpenMM.

It checks full-system and transforming-ligand forces and total energies against
stock OpenMM, allowing for the numerical differences between PME and Prolate and
the energy difference from changing the original grouped tail correction.
It changes charge and softcore globals, revisits the prepared
window, and runs 200 NVT steps. The prepared TYK2 RBFE inputs are bundled in
[alchemical/data](alchemical/data/README.md); OpenFE and network access are not
needed. The updates demonstrate the API rather than a complete sampling protocol.
Expressions compile through the plugin; users need no CUDA source code.
