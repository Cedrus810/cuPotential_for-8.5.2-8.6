<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ltoir-usage — LTO-IR Prolate Splitting Sample

End-to-end example that links the shipped LTO-IR math object and unified runtime,
and assembles Coulomb and dispersion energies, forces and virials. The API it
uses is described in [docs/ltoir.md](../../docs/ltoir.md), and the sample in
[docs/ltoir-usage.md](../../docs/ltoir-usage.md).

The cupo runtime tarball, OpenMM source tarball and GitHub source tree include
this complete example, the NVIDIA-generated water
system and independent OpenMM/Ewald reference in `data/water_10k/`, and the
reference generator. See [the dataset README](../../data/water_10k/README.md)
for provenance and reference construction.
The water snapshot is NPT-equilibrated at 300 K and 1 bar and included in
automatic benchmark dataset discovery. This sample evaluates its Coulomb and attractive C6 terms.
Running the example uses Python 3's standard library to read build metadata.
It needs no NumPy, OpenMM, JAX, external molecular data or OpenMM plugin build.

## Prerequisites

- A cupo build with compact convolution enabled (the default), containing
  `build/libcupo.so`, `include/cupo/convolution.h` and `build/runtime_build.json`.
- NVIDIA GPU with an architecture included in the shipped math object;
  executing `--compact` requires an architecture in the `compact_architectures`
  array in `build/runtime_build.json`.
- Python 3 (standard library only).
- A compatible CUDA Toolkit installed locally (for `run` directly).
  Match the package's producer version recorded in `.cuda-version`;
  compact convolution requires CUDA 13.1 or newer.
- An OpenMP-enabled host compiler and its runtime library (GCC/libgomp in the
  provided CUDA container).

## Usage

From the root of a GitHub clone or of the unpacked OpenMM source tarball, in the
tree's CUDA Docker environment, run either math variant. A clone has no `build/`,
so the first run downloads the matching runtime and math objects from the
release with `./fetch_cupo`, as the plugin's `run_build` does:

```bash
./run_docker samples/ltoir-usage/run              # standard variant, water_10k
./run_docker samples/ltoir-usage/run fast_math    # fast_math variant, water_10k
```

The cupo runtime tarball has no `run_docker`. From the root of the unpacked
tarball, with the compatible CUDA Toolkit, run the same script directly:

```bash
./samples/ltoir-usage/run              # standard variant, water_10k
./samples/ltoir-usage/run fast_math    # fast_math variant, water_10k
```

The optional argument is `standard` (the default) or `fast_math`. Both variants
use the packaged `water_10k` reference, containing 3,375 waters (10,125 atoms).
Use a different reference or place build output outside a read-only archive:

```bash
REF_BIN=/path/to/system_ref.bin ./samples/ltoir-usage/run
BUILD_DIR=/tmp/cupo-usage ./samples/ltoir-usage/run
```

## What it does

1. Compiles `usage.cu` with `-dc` (relocatable device code, LTO target).
2. Device-links it against `build/ltoir/<variant>/cupo_math.cu.o`, which contains
   all Prolate P1/P6 splitting and C²-order-3 spreading functions. The selected
   variant is `standard` or `fast_math`.
3. Runs the host PME demo — Coulomb (P=1) + C6 dispersion (P=6)
   energy, forces and the virial/pressure tensor, exact and `--particle-mesh`
   reciprocal — comparing against an analytical reference blob
   (`data/water_10k/water_10k_ref.bin` by default). Regenerate it or supply another system with
   `gen_reference_bin.py` (see
   [reference data](../../docs/ltoir-usage.md#reference-data)); a missing blob
   is an error.
4. Always links `build/libcupo.so`, which supplies splitting-plan creation,
   `cupo_plan_grid`, and compact convolution. Grid planning runs on the CPU
   without CUDA execution; there is no separate `plan_grid.cu.o`. The sample
   checks the runtime ABI before using its APIs. On a compiled target, runs
   `--compact` after the exact and CPU mesh modes. The binary finds the library through
   `$ORIGIN/../../build`, so the tree can be relocated. A custom `BUILD_DIR`
   also records the selected runtime directory. Missing or invalid
   `compact_architectures` metadata is an error; an empty array identifies a
   build without the convolution feature, which cannot run this sample.

The exact reciprocal and ordinary particle-mesh paths run on the CPU. Their
scalar calls use the host implementations in `cupo_math.cu.o`; the compact
path performs the FFT and convolution on the GPU, with host spreading,
interpolation, energy and virial reductions. Build and numerical failures
return a nonzero exit status.

`detect_architecture` in `../common.sh` selects the CUDA-visible device for
compilation and compact target checks, respecting `CUDA_VISIBLE_DEVICES`.

## Compact particle-mesh mode

The `compact_architectures` integer array in `build/runtime_build.json` records
the compiled compact targets; `[]` means the backend is disabled. The build takes
the runtime's architectures unless `CUPO_COMPACT_ARCHS` names others.
`cupo_convolution_check_mesh` checks a mesh on the current CUDA device.
Only SM120 has measured correctness/performance qualification so far.
Scalar-object targets are independent.

From `samples/ltoir-usage/`, after building with `./run`, the manual command is:

```text
./usage_prolate <ref.bin> --compact [--grid N] [--cutoff NM]
```

For example:

```bash
./usage_prolate ../../data/water_10k/water_10k_ref.bin --compact --cutoff 0.9
```

`--compact` implies `--particle-mesh`. The sample spreads and gathers on the
host, and calls the public `cupo_convolution_execute` API for the FP32 GPU
transforms and influence multiplication. The application computes energy and
virial from the exported unweighted forward spectrum `S`; those reductions
and the physical prefactors remain sample code.

Without `--grid`, the sample creates a Prolate P1 splitting plan and calls
`cupo_plan_grid(plan, box, CUPO_GRID_COMPACT, mesh)`, then destroys the plan.
This selects supported dimensions using the existing empirical p1/C²-order-3
heuristic. This is not a separate p6 accuracy calibration. An explicit `--grid N` uses the same supported length
on all three axes.

Each p1/p6 evaluation prints its real-space grid dimensions, compact reciprocal
grid dimensions, and retained support radii.

The scalar Fourier functions remain untruncated. For `--compact`, the sample
explicitly truncates at each model's design bandwidth: `c=10.75` for P1 and
`c=12.764031957789484` for P6. `compact_support_radius` retains axis modes with
`2*pi*abs(k)*cutoff <= c`, clamped to the mesh Nyquist limit; frequencies passed
to `cupo/math/splitting.h` are in cycles per length. Both the influence and its gradient are
also zeroed outside this radial band, including corners of the retained box.
This is an approximation to the full transform, not a claim of exact support.
The direct structure-factor and ordinary `--particle-mesh` paths retain the
full kernel on their represented modes. Virials for the clipped path describe
fixed mode membership; a changing box can move modes across the sharp cutoff.

Influence values are packed into exactly
`[min(Gx,2*sx+1), min(Gy,2*sy+1), sz+1]`, contiguous with Z fastest and X/Y in
FFT order. The optional raw forward spectrum uses the same shape and order.
The sample uses automatic workspace ownership:

```cpp
cupo_convolution_plan plan = nullptr;
cupo_status result = cupo_convolution_plan_create(
    gx, gy, gz, sx, sy, sz,
    {CUPO_GRID_FP32, 0}, {CUPO_GRID_FP32, 0}, &plan);
```

Compare each `cupo_status` with `CUPO_SUCCESS` (0) before proceeding; the sample
reports the returned error code on failure.

For caller-owned mutable scratch, `cupo_convolution_plan_prepare` takes the same
input arguments, followed by `uint64_t* workspace_bytes` and the output plan.
Bind at least that many
contiguous device bytes, with a 256-byte aligned base, using
`cupo_convolution_plan_set_workspace(plan, workspace, bytes)` before execution.
Only the required prefix is used as scratch; extra capacity remains unused and
may hold disjoint caller buffers. The scratch prefix cannot overlap any grid.
Automatic and manual plans use the same kernels and math. Execution performs
no allocation or synchronization. Keep workspace alive through queued work and
graph replay; drain work and destroy graphs referencing the old binding before
rebinding. Destruction never frees caller-owned scratch. Non-LTO planning may
still own immutable FFT coefficients; the standard LTO build needs none.

Spectral layout descriptors and caller padding are not part of the API. The
public header and C symbols have changed; rebuild consumers and update
bindings against the matching header and library.
