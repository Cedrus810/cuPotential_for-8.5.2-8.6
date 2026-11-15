<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# LTO-IR: the ltoir-usage sample

The cupo runtime tarball, OpenMM source tarball and GitHub source tree include
the complete particle-mesh example in
`samples/ltoir-usage/`, the `water_10k` PDB and independent reference, and
the reference generator. It assembles Coulomb and dispersion energies, forces and
virials using the scalar LTO-IR math object and `libcupo.so`.

Both tarballs include the runtime and scalar math objects. A GitHub clone does
not: the sample's `run` downloads them from the release first (`./fetch_cupo`).
See the [sample README](../samples/ltoir-usage/README.md#usage) for build and
run commands; no OpenMM plugin build is needed.

## Files

Paths are relative to the unpacked cupo tarball:

```
samples/
  common.sh                     — What the samples' run scripts share
  ltoir-usage/
    run                         — End-to-end build+run script (requires compact convolution)
    usage.cu                    — Full particle-mesh pipeline (coulomb + dispersion):
                                  energy, forces and the virial/pressure
                                  tensor, vs analytical references.
                                  Takes <system>_ref.bin; --particle-mesh / --grid N,
                                  --compact GPU convolution; links libcupo.
    gen_reference_bin.py        — OpenMM energy/forces + JAX Ewald virial generator
    ewald.py                    — Ewald functions used by the generator
data/water_10k/
  solvated.pdb.gz               — NVIDIA-generated 3,375-water NPT snapshot (300 K, 1 bar)
  water_10k_ref.bin             — Independent reference for the water system
  README.md                    — Provenance and regeneration
```

## End-to-End Particle-Mesh Assembly (`usage.cu`)

`usage.cu` shows how to wire the returns of the public API ([docs/ltoir.md](ltoir.md)) into a
complete particle-mesh implementation that matches an analytical reference to prolate
truncation precision.  It tests both coulomb + dispersion energies, forces, virials.
The packaged `water_10k` system and reference are the default input, so running
the example needs no external structure file. The optional reference generator
also accepts user-supplied periodic PDB structures.

The water snapshot is NPT-equilibrated at 300 K and 1 bar and included in
automatic benchmark dataset discovery. The sample evaluates its Coulomb and attractive C6 terms;
the dataset README records preparation and reference generation.

The exact reciprocal and `--particle-mesh` modes execute on the host. Their
scalar API calls resolve to the host side of the `__host__ __device__` symbols
in the selected variant of `cupo_math.cu.o`, which also supplies all P1/P6 and
spreading device functions. The `--compact` mode uses device buffers and
the compact GPU convolution API while keeping spreading, gathering and
energy/virial reductions on the host.

## The sample's compact mode

The full particle sample requires compact convolution in `libcupo.so`.
`samples/ltoir-usage/run` always links `libcupo.so` and
fails early if the library or compact architecture metadata is missing.
With a GPU listed in the package metadata it runs `--compact` after the exact
and CPU particle-mesh comparisons, using the packaged reference by default.
The executable uses `$ORIGIN/../../build` to locate the shared library after
relocation; an optional `BUILD_DIR` also records the selected runtime directory.
A build without the compact convolution feature cannot build this sample.

The [sample README](../samples/ltoir-usage/README.md#usage) gives the commands:
both math variants, another reference with `REF_BIN`, and the compact mode run by
hand, `usage_prolate <ref.bin> --compact [--grid N] [--cutoff NM]`.

`--compact` implies particle mesh: spreading and gathering run on the host,
while `cupo_convolution_execute` performs FP32 GPU transforms and influence
multiplication. The sample requests the unweighted forward spectrum `S` along
with the real output and computes energy and virial on the host. It assembles
the influence values, physical scaling and normalization itself.

Automatic dimensions come from the runtime's `cupo_plan_grid` with a Coulomb
Prolate P1 splitting plan and `CUPO_GRID_COMPACT`, using the combined-force
sizing policy [ltoir.md](ltoir.md) describes. Plan creation and sizing run on
the CPU and link `libcupo.so`; no `plan_grid.cu.o` is shipped.
The sample's `--compact` path
explicitly truncates at angular `|k|*cutoff <= c`, with `c=10.75` for P1 and
`c=12.764031957789484` for P6. Its per-axis support radii are clamped to Nyquist;
both H and its Fourier gradient are masked outside the sphere, including the
retained box's corners. The public scalar functions remain untruncated, as do
the ordinary host mesh's represented modes. This compact-path truncation is an
additional approximation. Its virial assumes fixed mode membership, away from
box changes that move modes across the band boundary.
Both H and S use contiguous Z-fastest FFT order with shape
`[min(Gx,2*sx+1), min(Gy,2*sy+1), sz+1]`. Explicit `--grid N` selects a cubic
mesh with a supported axis length; the combined-force target does not establish
an independent P6 accuracy guarantee.

## Reference data

`usage.cu` compares against a reference blob, `<system>_ref.bin`. The archive
includes `data/water_10k/water_10k_ref.bin`, which the launcher uses by default.
The launcher reads build metadata with Python 3's standard library; no NumPy,
OpenMM, JAX or reference generation is needed to run it. The launcher accepts
only the optional math variant, `standard` or `fast_math`.

The NVIDIA-generated water system contains 3,375 waters (10,125 atoms) in a
cubic periodic box. Originally generated with OpenMM 8.4 on 2026-04-06, it was
subsequently NPT-equilibrated at 300 K and 1 bar after initial energy minimization.
The reference was regenerated from the final NPT snapshot and box. The water
dataset, reference, sample, reference generator and documentation are Apache-2.0. See
[the dataset README](../data/water_10k/README.md) for provenance.

To regenerate it, install NumPy, OpenMM and JAX in a Python environment and run:

```bash
cd samples/ltoir-usage
python gen_reference_bin.py                   # writes ../../data/water_10k/water_10k_ref.bin
python gen_reference_bin.py --pdb /path/to/system.pdb.gz --output system_ref.bin
```

`--pdb` accepts PDB or gzipped PDB files for net-neutral, orthorhombic periodic
systems and assigns `amber14-all.xml` and `amber14/tip3pfb.xml` parameters.
Non-neutral systems are rejected because the independent JAX reference does
not include a charged-system background correction. The generator runs two single-point
evaluations on OpenMM's Reference platform (FP64 PME at `ewald_tol=1e-9`) and
packs coordinates, charges, σ/ε, exclusions, energies and per-atom forces into
the blob. Coulomb uses the particle charges; dispersion contains only the
attractive geometric-C6 term, without the repulsive R¹² term. Bonded 1–2 and
1–3 pairs are excluded; 1–4 pairs use full nonbonded strength.

JAX autodiff of an independent Ewald sum supplies the virials,
`W = −∂E/∂ε` under affine box strain, in `(xx, yy, zz, xy, xz, yz)` order. The
reference uses an extended real-space cutoff and reciprocal range to reduce
its truncation error. It does not call cuPO to produce expected values.

Set `REF_BIN=/path/to/system_ref.bin` to run against another blob. A missing
reference is an error.
