<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# water_10k

NVIDIA generated this periodic water system with OpenMM on 2026-04-06 and
equilibrated it at 300 K and 1 bar on 2026-11-07. The shipped PDB is the final
NPT snapshot: 3,375 water molecules (10,125 atoms, all `HOH`) in a cubic box
with edge length 4.6716 nm after PDB rounding. Each water molecule is stored
whole, with its centroid in the primary box.

`solvated.pdb.gz`, `water_10k_ref.bin`, this documentation, and the cuPO sample
and reference-generator source are distributed under Apache-2.0.

The `solvated.pdb.gz` convention includes this system in the OpenMM plugin's
automatic benchmark dataset discovery. The same coordinates and box supply
the independent Coulomb/C6 reference used by the LTO-IR example.

## Snapshot preparation

OpenMM 8.4 with `amber14-all.xml` and `amber14/tip3pfb.xml` supplies the full
Coulomb and Lennard-Jones potential during preparation and dynamics:

1. Make molecules whole across periodic boundaries and remove initial close
   contacts with `LocalEnergyMinimizer`, using a 10 kJ/mol/nm tolerance and no
   iteration limit. This is preparation for dynamics.
2. Thermalize for 100 ps at fixed volume and 300 K, with the barostat disabled.
3. Run 1,000 ps of NPT dynamics at 300 K and 1 bar and retain the final state.
   No minimization is applied after dynamics.

Dynamics uses `LangevinMiddleIntegrator` with a 2 fs step, 1/ps friction,
`HBonds` constraints, rigid water, and constraint tolerance `1e-8`.
`MonteCarloBarostat` attempts an isotropic volume change every 25 steps.
PME uses a 0.9 nm cutoff, Ewald tolerance `1e-6`, and OpenMM's default
Lennard-Jones dispersion correction. Seeds are 20261107 for the integrator,
20261108 for the barostat, and 20261109 for initial velocities. Minimization
uses CUDA double precision; dynamics uses CUDA mixed precision with
deterministic forces on an RTX 5090.

Density, temperature, and potential energy were sampled every 10 ps.
Late NPT block averages are:

| NPT interval (ps) | Density (g/mL) | Temperature (K) | Potential energy (kJ/mol) |
|---|---:|---:|---:|
| 250–500 | 0.99375 | 300.95 | -166004.56 |
| 500–750 | 0.99363 | 299.94 | -165914.14 |
| 750–1000 | 0.99427 | 299.78 | -165971.59 |

The final instantaneous density is 0.99032 g/mL; it need not equal the ensemble
average. The saved PDB contains the final NPT box and whole-molecule positions.
Reference generation rereads this PDB, including its coordinate and box
rounding, so benchmarks and the reference binary use the same configuration.

## Run the particle-mesh example

The cuPO runtime archive, the OpenMM source tarball and the GitHub tree include
this PDB and its precomputed reference. The
[LTO-IR sample's README](../../samples/ltoir-usage/README.md#usage) says how to
run the example in each. It compares Coulomb and attractive C6 dispersion
energies, forces and virials for exact reciprocal, host particle-mesh, and
supported GPU compact convolution modes. This water system is the example's
default input; running it uses Python 3's standard library, with no NumPy,
OpenMM or JAX dependency.

## Reference generation

The reference generator assigns `amber14-all.xml` and `amber14/tip3pfb.xml`
parameters to the PDB. OpenMM's FP64 Reference platform supplies energies and
forces; independent JAX Ewald sums supply virials through differentiation of
box strain. The generator reports cross-checks between these calculations and
does not call cuPO to produce expected values.

With NumPy, OpenMM and JAX installed, regenerate from the repository or archive
root:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python3 \
    samples/ltoir-usage/gen_reference_bin.py
```

By default (also selectable with `--system water_10k`), this reads
`data/water_10k/solvated.pdb.gz` and writes
`data/water_10k/water_10k_ref.bin`. Use `--output /path/to/water_10k_ref.bin`
to write elsewhere. The blob stores coordinates, particle parameters,
exclusions, and reference energies, forces and virials in the format documented
by the generator.

For another net-neutral, orthorhombic periodic system, supply
`--pdb /path/to/system.pdb.gz --output /path/to/system_ref.bin`. Charged systems
are rejected because the independent JAX reference does not include a
charged-system background correction.

The shipped reference contains 10,125 atoms and 10,125 excluded pairs, with a
sample splitting cutoff of 0.9 nm. Reference settings are:

- OpenMM Coulomb PME: 2.0 nm real-space cutoff and `1e-9` Ewald tolerance.
- OpenMM C6 dispersion: geometric combining, `alpha = 2.0 nm^-1`, and a `64³`
  reciprocal mesh with damped real-space and exclusion terms; no repulsive C12.
- JAX Ewald: `alpha = 2.5 nm^-1`, 1.8 nm real-space cutoff, and a spherical
  reciprocal sum through integer radius 22. Virials differentiate affine box
  strain and use `(xx, yy, zz, xy, xz, yz)` order.

Independent relative differences for the shipped file are:

| Cross-check | Coulomb | C6 dispersion |
|---|---:|---:|
| JAX versus OpenMM energy | 2.55e-9 | 2.07e-9 |
| JAX versus OpenMM forces | 2.78e-9 | 6.05e-9 |
| JAX virial trace versus OpenMM finite difference | 2.69e-8 | — |

The generator rejects the bundled reference if energy/force differences exceed
`2e-6` or the virial-trace difference exceeds `1e-5`. Energy references are
`-198199.1030934021 kJ/mol` for Coulomb and `-53694.206359164506 kJ/mol` for C6.
The file was generated with Python 3.12.12, NumPy 2.5.1, OpenMM 8.4.0, and
JAX/jaxlib 0.9.2, using OpenMM's Reference platform and the CUDA JAX backend
on an RTX 5090. Numerical agreement is the portable reproducibility check;
exact bytes can vary with library versions and execution backend.

The reference is 1,053,160 bytes with SHA-256:

```text
a67441dc55279620b95e999aad30dc3b70a601e247c21a4e33964c69c7e88522
```

The compressed PDB is 162,008 bytes with SHA-256:

```text
8ac94a4e9c5870a78c673879070d226b0ed0163bfca30d2f35f6dee573d802da
```
