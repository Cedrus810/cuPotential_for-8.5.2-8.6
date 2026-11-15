<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Documentation commands for cuPotential 0.3.5

Generated from the [cuPotential documentation](https://docs.nvidia.com/cuda/cupotential) for this release. CI runs these commands to check installation, source builds, samples, tests, and benchmarks.

## Get the source from GitHub

[Documentation](https://docs.nvidia.com/cuda/cupotential/install)

```bash
git clone --branch v0.3.5 https://github.com/NVIDIA/cuPotential
cd cuPotential
```

## Unpack the source archive

[Documentation](https://docs.nvidia.com/cuda/cupotential/install)

```bash
tar -xzf cupo-nv-openmm-plugin-src-0.3.5+*.tar.gz
cd cupo-nv-openmm-plugin-src-0.3.5+*/
```

## Install with pip from PyPI

[Documentation](https://docs.nvidia.com/cuda/cupotential/install)

```bash
pip install nv-openmm-plugin
```

## Install with pip from the GitHub release

[Documentation](https://docs.nvidia.com/cuda/cupotential/install)

```bash
./run_docker bash
pip install https://github.com/NVIDIA/cuPotential/releases/download/v0.3.5/nv_openmm_plugin-0.3.5-cp312-cp312-manylinux_2_38_x86_64.whl
./plugins/openmm/samples/solvated_systems/run
```

## Install with conda

[Documentation](https://docs.nvidia.com/cuda/cupotential/install)

```bash
curl -fL https://github.com/NVIDIA/cuPotential/releases/download/v0.3.5/nv-openmm-plugin-conda-channel-0.3.5.tar.gz | tar -xz
./run_docker bash
micromamba create -y -n cupotential -c ./nv-openmm-plugin-conda-channel-0.3.5 -c conda-forge nv-openmm-plugin
micromamba run -n cupotential ./plugins/openmm/samples/solvated_systems/run
```

## Build the plugin

[Documentation](https://docs.nvidia.com/cuda/cupotential/install)

```bash
./run_docker plugins/openmm/run_build
```

## Run the samples

[Documentation](https://docs.nvidia.com/cuda/cupotential/install)

```bash
./run_docker                                              # solvated_systems, the default
./run_docker plugins/openmm/samples/alchemical/run
```

## Tests and benchmarks

[Documentation](https://docs.nvidia.com/cuda/cupotential/tests)

```bash
./run_test openmm_unit -k "test_energy and dhfr-fp64"
./run_test openmm_bench -k "apoa1-mixed-nvt"
```

## Test a pip installation

[Documentation](https://docs.nvidia.com/cuda/cupotential/tests)

```bash
./run_docker bash
pip install <the release's wheel>
CUPO_TEST_INSTALLED=1 ./run_test openmm_unit -k test_unit_conversion
```

## Test a conda installation

[Documentation](https://docs.nvidia.com/cuda/cupotential/tests)

```bash
./run_docker bash
micromamba create -y -n cupotential -c ./<the release's conda channel> -c conda-forge nv-openmm-plugin
micromamba install -y -n cupotential -c conda-forge pytest
CUPO_TEST_INSTALLED=1 micromamba run -n cupotential ./run_test openmm_unit -k test_unit_conversion
```
