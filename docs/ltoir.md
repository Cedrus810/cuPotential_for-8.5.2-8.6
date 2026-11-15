<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Scalar Math — LTO-IR Device Functions

Pre-compiled CUDA LTO-IR implementing prolate 1/r^P splittings for P=1
(Coulomb) and P=6 (dispersion), plus spreading, interpolation and spreading-window
Fourier functions, all in fp32 and fp64. All 36 public functions are bundled in
one `cupo_math.cu.o` per math variant, shipped in the cupo tarball beside the runtime.

The cupo tarball's `libcupo.so` includes GPU grid convolution through a host
C API by default (`CUPO_BUILD_COMPACT_CONVOLUTION=ON`, CUDA 13.1+).
Turning it off omits the convolution API while retaining the runtime and scalar
objects.

Paths below are relative to the root of the unpacked cupo tarball or source tree.

The OpenMM source tarball and public GitHub tree also include the scalar math
headers in `include/cupo/math/` and the complete example in
[`samples/ltoir-usage/`](../samples/ltoir-usage/README.md). Both tarballs carry the
matching `cupo_math.cu.o` objects and runtime library. A GitHub clone does not:
the example's `run` downloads them and `build/runtime_build.json` from the
release first (`./fetch_cupo`).

Polynomial and piecewise Fourier coefficients are baked in at compile time;
there are no coefficient arrays in the public API.

## Baked Bandwidths

Both defaults are C2 Prolate splittings. In the mathematical model, their
short-range potential and its first two radial derivatives vanish at the
cutoff. The fast floating-point complement evaluation can leave small
cancellation residuals immediately inside the cutoff; it returns exact zero
at and beyond the represented cutoff. The matching Fourier
kernels have nonzero oscillatory tails: `c` is a design bandwidth, not a support
boundary or a molecular force-error guarantee.

| Exponent | Design bandwidth c | Runtime scale |
|----------|--------------------|---------------|
| P=1      | 10.75              | real cutoff   |
| P=6      | 12.764031957789484  | real cutoff   |

Public scalar signatures are unchanged. Rebuild direct, reciprocal, exclusion,
and self-energy consumers together when updating from the historical model.
The P6 Fourier derivative at zero now returns its finite analytic value.

## Contents

The parts of the cupo tarball this guide is about:

```
include/
  cupo/math/splitting.h         — Splitting __host__ __device__ declarations (P=1, P=6)
  cupo/math/spreading.h         — Spreading-window __host__ __device__ declarations
  cupo/splitting_plan.h         — Runtime splitting-plan lifecycle and queries
  cupo/kspace.h                 — Reciprocal-space C API and host-only grid planner
  cupo/package_version.h        — Shared version macros generated from VERSION
  cupo/convolution.h             — Compact convolution C API
  cupo/status.h                  — Shared result codes
  cupo/visibility.h              — Runtime symbol visibility macros
build/
  libcupo.so                    — Unified runtime, including FP32 compact convolution:
                                  a link to libcupo.so.<ABI>, the library's SONAME,
                                  itself a link to the file, libcupo.so.<VERSION>
  runtime_build.json            — Toolchain metadata and compiled compact SM targets
  cmake/CupoRuntime/            — find_package(CupoRuntime CONFIG) metadata
  ltoir/
    standard/                   — Compiled WITHOUT --use_fast_math
      cupo_math.cu.o           —   All splitting and spreading functions
    fast_math/                  — Compiled WITH --use_fast_math
      cupo_math.cu.o           —   Same functions as standard
share/cmake/CupoConvolution/     — find_package(CupoConvolution CONFIG) metadata
samples/
  common.sh                    — Sample layout and GPU architecture helpers
  ltoir-usage/
    usage.cu                   — Full Coulomb/dispersion particle-mesh example
    run                        — Build and run the selected math variant
    README.md                  — Sample instructions
    gen_reference_bin.py        — Optional reference generator (OpenMM + JAX)
    ewald.py                   — Ewald functions used by the generator
data/water_10k/
  solvated.pdb.gz              — NVIDIA-generated 3,375-water NPT snapshot (300 K, 1 bar)
  water_10k_ref.bin            — Independent energy, force and virial reference
  README.md                   — Dataset provenance and regeneration
docs/ltoir-usage.md             — Particle-mesh assembly and numerical conventions
```

The convolution API in `libcupo.so` and its CMake package are available when the
tarball was built with compact convolution, which is the default.
`build/runtime_build.json` always contains a `compact_architectures` integer
array listing the compiled SM targets; `[]` means the compact backend is disabled.

Each `cupo_math.cu.o` contains both NVVM IR (LTO-IR; selected by the device
linker with `-dlto`) and a host-side object section providing all 36
`__host__ __device__` functions to host code. Link exactly one variant.
Grid planning is implemented only in `libcupo.so`; it has no LTO-IR object.

## Moving from separate scalar objects

Replace the `prolate_p1_device.cu.o`, `prolate_p6_device.cu.o` and
`c2o_device.cu.o` inputs in each link command with the matching variant of
`cupo_math.cu.o`. This also applies when a consumer previously needed only one
of those objects. The scalar headers, function names, signatures and numerical
behavior are unchanged; no runtime ABI change is needed for this consolidation.
The old scalar object files are no longer shipped. Do not link them alongside
`cupo_math.cu.o`, because they define the same symbols.

## Moving from cupo.h

The headers and the paths of the LTO-IR delivery have changed. Code written
against the `cupo-ltoir` tarball stops compiling until its includes are changed.

The LTO-IR objects used to come in their own `cupo-ltoir` tarball, with all
their declarations in one header, `include/cupo.h`. Now they come in the
cupo tarball, and `cupo.h` is split into three headers by area. No header is left
at the name `cupo.h`, and there is no single header that includes all three.

Replace

```cpp
#include "cupo.h"
```

with the headers of the functions you use:

```cpp
#include <cupo/math/splitting.h>  // cupo_p1_*, cupo_p6_*
#include <cupo/math/spreading.h>  // cupo_spread_1d_*, cupo_interpolate_1d_*, cupo_discrete_fourier_kernel_1d_*
#include <cupo/kspace.h>          // cupo_plan_grid; includes splitting-plan declarations
```

The include directory is still the tarball's `include/`.

The scalar math signatures are unchanged. Runtime ABI 32 consolidates grid
planning into `cupo_plan_grid(plan, box, policy, mesh)` in `libcupo.so`.
Create a splitting plan through `<cupo/splitting_plan.h>` to supply the method
and cutoff. Replace `cupo_experimental_plan_grid` with this call using
`CUPO_GRID_DENSE`, and `cupo_experimental_plan_grid_dx` using
`CUPO_GRID_COMPACT`. The legacy wrappers and
`cupo_splitting_plan_mesh_dimensions` have been removed, and the former
cutoff-taking `cupo_plan_grid` signature is no longer supported.
Check the returned status: failure leaves the output unchanged instead of
clearing it, and `cupo_last_error()` describes the error. Remove
`plan_grid.cu.o` from link commands and link `libcupo.so` instead. Rebuild
consumers against the new headers and library.

`cupo_grid_policy` and `cupo_plan_grid` are declared in `<cupo/kspace.h>`, which
includes `<cupo/splitting_plan.h>` for the plan lifecycle functions. Replace any
include of the removed `<cupo/grid_plan.h>` with `<cupo/kspace.h>`. This header
move changes no function names or signatures and requires no further runtime
ABI change.

The declarations formerly in `cupo.h` map to these three headers. Every scalar
function in the table resolves to `cupo_math.cu.o`; `cupo_plan_grid` resolves to
`libcupo.so`, and version macros need no linked object:

| Declarations (each `_f32` and `_f64` unless noted) | Header |
|---|---|
| `cupo_p1_short_range_value`, `cupo_p1_short_range_grad` | `cupo/math/splitting.h` |
| `cupo_p1_long_range_value`, `cupo_p1_long_range_grad` | `cupo/math/splitting.h` |
| `cupo_p1_fourier_kernel_value`, `cupo_p1_fourier_kernel_grad` | `cupo/math/splitting.h` |
| `cupo_p1_self_value`, `cupo_p1_fourier_kernel_dc` | `cupo/math/splitting.h` |
| `cupo_p6_short_range_value`, `cupo_p6_short_range_grad` | `cupo/math/splitting.h` |
| `cupo_p6_long_range_value`, `cupo_p6_long_range_grad` | `cupo/math/splitting.h` |
| `cupo_p6_fourier_kernel_value`, `cupo_p6_fourier_kernel_grad` | `cupo/math/splitting.h` |
| `cupo_p6_self_value` | `cupo/math/splitting.h` |
| `cupo_spread_1d`, `cupo_interpolate_1d` | `cupo/math/spreading.h` |
| `cupo_discrete_fourier_kernel_1d` | `cupo/math/spreading.h` |
| `cupo_plan_grid` | `cupo/kspace.h` |
| the `CUPO_VERSION_*` macros, from `#include "cupo/package_version.h"` | `cupo/math/splitting.h` includes it, as `cupo.h` did |

The other headers kept their names: `cupo/convolution.h`, `cupo/status.h`,
`cupo/visibility.h` and `cupo/package_version.h`.

The paths in the tarball changed as follows:

| `cupo-ltoir` tarball | cupo tarball |
|---|---|
| `cupo-ltoir-<ver>+....tar.gz` | `cupo-<ver>+....tar.gz` |
| `include/cupo.h` | `include/cupo/math/splitting.h`, `include/cupo/math/spreading.h`, `include/cupo/kspace.h` |
| `ltoir/standard/*.cu.o` | `build/ltoir/standard/cupo_math.cu.o` |
| `ltoir/fast_math/*.cu.o` | `build/ltoir/fast_math/cupo_math.cu.o` |
| `lib/libcupo_compact.so` | `build/libcupo.so` |
| `lib/cmake/CupoConvolution/` | `share/cmake/CupoConvolution/` |
| `sample/` | `samples/ltoir-usage/` (full particle-mesh example) |

`include/cupo/convolution.h`, `include/cupo/status.h`,
`include/cupo/visibility.h` and `include/cupo/package_version.h` are where they
were. The compiled compact SM targets are recorded in the `compact_architectures`
array in `build/runtime_build.json`. For CMake, point
`CMAKE_PREFIX_PATH` at the root of the unpacked tarball, as before;
`find_package(CupoConvolution)` and the `Cupo::Convolution` target keep their
names. The target now links `Cupo::Runtime`, which resolves to `libcupo.so`;
direct compiler invocations should replace `-lcupo_compact` with `-lcupo`.

## Public API

The splitting entrypoints are `__host__ __device__` scalar functions — callable
from both kernels and host C++ via the same `.cu.o` object files.
`cutoff` is the only runtime knob; every other geometric scalar
(`inv_cutoff`, `arg_scale`, `pref`) is derived inside the function.

```cpp
#include <cupo/math/splitting.h>

// Real-space per-pair (signature: r², cutoff)
__host__ __device__ float  cupo_p1_short_range_value_f32(float r2, float cutoff);
__host__ __device__ double cupo_p1_short_range_value_f64(double r2, double cutoff);
// ... _short_range_grad, _long_range_value, _long_range_grad (fp32+fp64 each)

// Fourier per-k (signature: k², cutoff)
__host__ __device__ float  cupo_p1_fourier_kernel_value_f32(float k2, float cutoff);
// ... _fourier_kernel_grad — dK/dk² for the reciprocal-space virial (fp32+fp64)

// Scalar self-correction (signature: cutoff)
__host__ __device__ float  cupo_p1_self_value_f32(float cutoff);
__host__ __device__ double cupo_p1_self_value_f64(double cutoff);

// Added in 0.3.0 — Fourier DC bin, P=1 only (signature: cutoff), for the
// net-charge ("plasma") correction
__host__ __device__ float  cupo_p1_fourier_kernel_dc_f32(float cutoff);
__host__ __device__ double cupo_p1_fourier_kernel_dc_f64(double cutoff);
```

P=6 has the same surface with `cupo_p6_` prefix, less `fourier_kernel_dc`:
the P=6 kernel is finite at k = 0, so its DC bin comes straight from
`cupo_p6_fourier_kernel_value(0, rc)`.

`*_grad` returns the analytic derivative w.r.t. the squared distance
`dS/d(r²)` (signed) — not `dS/dr`.  Form the per-component force at the
call site by multiplying by 2 and the displacement:
`F_i = -2·prefactor·qᵢqⱼ·grad(r²)·(rᵢ - rⱼ)`.  This avoids the per-pair
square root and division by r (`(dS/dr)/r ≡ 2·dS/d(r²)`).

`*_self_value` returns the additive self-correction scalar (no per-pair
device-side analogue); the values come from a baked compile-time literal
divided by the appropriate cutoff power.

`cupo_p1_fourier_kernel_dc` returns the **DC bin of the P=1 Fourier
kernel** — the k = 0 background ("plasma") constant, and this splitting's
drop-in replacement for Ewald's `-pi/alpha^2`.

It is a separate entry point because `cupo_p1_fourier_kernel_value`
**diverges** at k = 0: the bare Coulomb pole `4*pi/k^2` is exactly what the
neutralising background cancels, and this is the finite part left over.
(P=6 needs no counterpart — its DC bin is finite, so
`cupo_p6_fourier_kernel_value(0, rc)` returns it directly.)

It drops into the reciprocal sum on the same line as every other bin, with
`|rho(0)|^2 = (sum_i q_i)^2`; the charge, the volume, the half and the
coupling stay with the caller:

```
Ewald:  E = -coulomb * pi * (sum_i q_i)^2 / (2 * alpha^2 * V)
here:   E =  coulomb * 0.5 * cupo_p1_fourier_kernel_dc(rc) * (sum_i q_i)^2 / V
```

`sum_i q_i` is the SIGNED charge sum, so a neutral cell zeroes the whole
term.  The constant is negative (as `-pi/alpha^2` is) and has units of
length², so `volume` must be in the cube of the unit `cutoff` is given in.
Energy only — a uniform background exerts no force, so there is no `_grad`.

With angular frequency `q=2*pi*k`, it is `lim_{q->0} (G(q) - 4*pi/q^2)`.
Its constant belongs to this C2 model. Substituting Ewald's constant changes the
energy of charged systems while leaving their forces unchanged.

Write `L_p(r)=ell_p(r/rc)/rc^p` for the finite long-range complement and
`S_p=r^-p-L_p` inside the cutoff. Outside, `S_p=0` and `L_p=r^-p`.
The Fourier argument is `k²` in squared **cycles per length**, and `z=2*pi*rc*sqrt(k²)`.

Both real-space powers use the same fast complement implementation as the
in-tree evaluators, with scalar adapters for the public `(r2, cutoff)`
signatures and squared-distance gradient convention. Native pair callers
can reuse prepared cutoff geometry and inverse-distance arithmetic. The
coefficient rows and their analytic derivative coupling are shared.

Short-range accuracy is measured against the bare value `r^-p` and bare
gradient magnitude `(p/2)*r^(-p-2)`. A tiny interior short-range value or
gradient can change sign through cancellation; relative accuracy and
cubic/quadratic landing of that tiny remainder are not guaranteed. Long-range
values and gradients retain relative source-accuracy checks. The mathematical
C2 model, Fourier complement and self terms are unchanged. Physical cutoffs
of 5–9 Å correspond to `cutoff=0.5`–`0.9` when distances are in nm.

| Function | P=1 | P=6 |
|----------|-----|-----|
| `short_range_value` | `S_1` | `S_6` |
| `short_range_grad` | `dS_1/d(r²)` | `dS_6/d(r²)` |
| `long_range_value` | `L_1` | `L_6` |
| `long_range_grad` | `dL_1/d(r²)` | `dL_6/d(r²)` |
| `fourier_kernel_value` | full transform of `L_1`, with zero DC convention | full transform of `L_6`, retaining finite DC |
| `fourier_kernel_grad` | `dK/d(k²)`, zero sentinel at DC | `dK/d(k²)`, finite analytic derivative at DC |
| `self_value` | `-L_1(0)/2` | `+L_6(0)/2` for attractive dispersion |
| `fourier_kernel_dc` | finite part `net_charge_unit * rc²` | use `fourier_kernel_value(0,rc)` |

### Splitting Identities

- **P=1**: `SR_value + LR_value = 1/r`,  `(SR_grad + LR_grad)·(2r) = d(1/r)/dr = -1/r²`
- **P=6**: `SR_value + LR_value = 1/r⁶`, `(SR_grad + LR_grad)·(2r) = d(1/r⁶)/dr = -6/r⁷`

### Mesh spreading & grid planning

For the FFT / particle-mesh reciprocal path (an alternative to the exact
structure-factor sum), three more `__host__ __device__` entrypoints implement the
C²-order-3 (4-point) spreading window. The runtime provides a host-only grid planner:

```cpp
#include <cupo/math/spreading.h> // spreading, interpolation, discrete Fourier kernel
#include <cupo/kspace.h>         // host-only grid planner and splitting-plan declarations

// 4-point spreading weights, and their ∂/∂t (force interpolation), at offset t∈[0,1)
__host__ __device__ void cupo_spread_1d_f64(double t, double w[4]);
__host__ __device__ void cupo_interpolate_1d_f64(double t, double w[4]);

// Per-axis discrete deconvolution amplitude |φ̂(k)| of the sampled window (length-G axis)
__host__ __device__ double cupo_discrete_fourier_kernel_1d_f64(int k, int G);

// Host-only runtime query: particle-mesh sizes (Gx,Gy,Gz); box edges MUST be in nm.
// The splitting plan supplies the method, cutoff and method-specific parameters.
cupo_status cupo_plan_grid(cupo_splitting_plan_handle plan,
                           const double box_edges[3], cupo_grid_policy policy,
                           int grid[3]);
```

`cupo_spread_1d` / `cupo_interpolate_1d` / `cupo_discrete_fourier_kernel_1d` also
have `_f32` variants. The mesh pipeline spreads charges with `cupo_spread_1d`,
forward-FFTs, multiplies by the discrete deconvolution kernel
`pref·K(k²)/|φ̂|²`, inverse-FFTs, and gathers forces with `cupo_interpolate_1d`.
`cupo_plan_grid` picks the grid up-front.

The query uses the splitting plan's stored parameters and mesh rule. Dense
sizing supports both Ewald and Prolate plans, including dispersion P6. Ewald
uses its actual stored alpha and rounds up to the next 7-smooth size; Prolate
uses measured cuFFT sizes where available and otherwise the next 7-smooth size.
`CUPO_GRID_COMPACT` supports Coulomb Prolate P1 plans and rounds up to the
smallest compiled compact length, from the 11-smooth sizes through 512.
Both policies enforce a six-point minimum. Plans represent cutoff as float.
Some Prolate results therefore differ from the former standalone 13-smooth
rule: at 0.9 nm cutoff a 6.698671334727977 nm cubic box uses 42³ for dense
sizing and 40³ for compact sizing.

`cupo/kspace.h` is includable by ordinary C and C++ compilers with the CUDA
Toolkit headers on the include path. Link `libcupo.so` or the CMake target
`Cupo::Runtime`; the planner is independent of the scalar objects' standard or
fast-math variant. Plan creation and grid sizing run on the CPU without
accessing a GPU. The shared library's CUDA dependencies must still be available
to the loader. Sizing does not probe backend availability, select an execution
backend, or change reciprocal support. Applications needing NPT expansion
headroom pass an expanded box explicitly.

Grid sizing requires positive finite box lengths and a plan created with a
positive finite cutoff **in nanometers**. Prolate sizing is not invariant to
length-unit changes: the cutoff selects a calibration
at fixed physical lengths. Convert both the box and cutoff from Å by multiplying
by 0.1; for example, a 69 × 58 × 58 Å box and 6 Å cutoff become
`box_diag={6.9, 5.8, 5.8}` and `cutoff=0.6`.

The Prolate policies use `u=1.7` at 0.6 nm and `u=1.4` at 0.9 nm, with interpolation
between and clamping outside. They are calibrated to `5e-4` relative-L2 error
in combined physical Coulomb plus attractive geometric-C6 forces using the
shared C²-order-3 window. This is not an independent per-channel guarantee;
accuracy outside 0.6–0.9 nm has not been qualified. The calibration includes
dense full-spectrum evaluation and compact evaluation with an explicit radial
band truncation. The grid policy itself applies no mask or backend setting.

The query returns `CUPO_INVALID_VALUE` for invalid input and `CUPO_INVALID_SIZE`
when the plan does not support the selected policy or no dimension fits,
leaving the output unchanged. It never clamps to an undersized mesh. Read
`cupo_last_error()` for error details.

Mesh rounding, retained Fourier modes, and execution backend are independent:

| Execution backend | Full spectrum | Truncated spectrum | Mesh requirements |
|---|---|---|---|
| Host cuFFT API | Supported | Supported through a masked influence table; FFT storage and transforms remain dense | Either grid policy's dimensions |
| Compact cuFFTDx | Supported | Supported with packed retained modes | Compiled dimensions and a supported device; use `CUPO_GRID_COMPACT` for sizing |

`CUPO_GRID_DENSE` and `CUPO_GRID_COMPACT` name size ladders, not spectrum layouts.
A dense-policy mesh can also run on the compact backend if all its dimensions
are supported. A compression ratio cannot choose the ladder: full-spectrum
compact execution still needs compiled sizes, and truncated cuFFT execution
still transforms the full grid. Choose retained support independently; the
runtime's `cupo_kspace_configure_convolution` selects the backend and spectrum.
Its `AUTO` backend can fall back to cuFFT without changing the mesh or mask.

For the full C2 Fourier complement, set reciprocal support to
`(Gx/2, Gy/2, Gz/2)` using integer division. Numerical truncation is an explicit
application choice independent of the grid policy.

```cpp
#include <cupo/kspace.h>

double box[3] = {6.0, 8.0, 10.0}; // nm; expand here if the application needs headroom
int grid[3];
cupo_splitting_plan_handle plan = nullptr;
cupo_status status = cupo_splitting_plan_create_prolate(0.9, 1.0, &plan);
if (status != CUPO_SUCCESS) { /* report cupo_last_error() */ }
else {
    status = cupo_plan_grid(plan, box, CUPO_GRID_COMPACT, grid);
    if (status != CUPO_SUCCESS) { /* report cupo_last_error() or try dense sizing */ }
    cupo_splitting_plan_destroy(plan);
}
```

## Compact convolution

Include `<cupo/convolution.h>` from host C or C++ and link
`libcupo.so`. The convolution API implements one real 3D grid per call, with
FP32 FFT arithmetic and intermediates. Real input/output formats can be FP32,
FP16, or signed fixed-point INT16/INT32; influence values and exported complex
spectra are FP32. This is not a batched molecular-dynamics pipeline.

The caller supplies the spread grid, influence grid and CUDA stream. Existing
spreading and force-interpolation code remains with the caller. The library
does not compute a prolate influence function, physical prefactors, energy
reductions or force interpolation. Include normalization, deconvolution and
physical scaling in the supplied influence values as appropriate.

Create a plan outside graph capture with explicit dimensions, support radii and
formats:

```cpp
cupo_convolution_plan plan = nullptr;
cupo_status result = cupo_convolution_plan_create(
    128, 128, 128, 27, 27, 27,
    {CUPO_GRID_FP32, 0}, {CUPO_GRID_FP32, 0}, &plan);
```

This path owns its mutable scratch. For caller-managed scratch, call
`cupo_convolution_plan_prepare` with the geometry and formats followed by
`uint64_t* workspace_bytes` and `cupo_convolution_plan* out`, then bind a contiguous device region with
`cupo_convolution_plan_set_workspace(plan, workspace, bytes)`. The base must be
256-byte aligned and the region must hold at least the reported byte count.
Only that required prefix is scratch; extra capacity remains unused and may
hold disjoint caller buffers. The scratch prefix cannot overlap any grid.
`cupo_convolution_plan_workspace_bytes` also reports the required mutable scratch size.
Preparation leaves scratch unbound; execution fails until it is bound.
The standard LTO build needs no extra FFT coefficients. Non-LTO fallback
planning may own immutable FFT coefficient allocations, separate from mutable
scratch, so preparation does not promise zero GPU allocations of every kind.

Both ownership paths run the same kernels and math. Call
`cupo_convolution_execute` to enqueue work without allocation or synchronization.
Request real output, the compact unweighted forward spectrum, or both.
The real result follows the unnormalized cuFFT convention; include
`1/(Nx*Ny*Nz)` in H for a normalized convolution. Every fallible call returns
`cupo_status`; compare it with `CUPO_SUCCESS` (0) and use the returned error code
for diagnostics. Serialize uses of one plan and retain it and all
buffers, including caller workspace, until queued work and graph replays have
finished. Before rebinding, drain work and destroy graphs referencing the old
workspace binding. Bind outside capture, and retire graphs before destroying
the plan. Destruction never frees caller-owned workspace.

The public header, C symbols and CMake package names have changed. Rebuild
consumers and update bindings against the matching header and library.

Each axis supports the 11-smooth lengths in `[2,512]`. Support radii obey
`0 <= s <= floor(N/2)`; the retained shape is
`[min(Nx,2*sx+1), min(Ny,2*sy+1), sz+1]`. Influence H and optional forward S
must use exactly this shape, contiguous with Z fastest and X/Y in FFT order.
The API has no caller-selected spectral strides, padding or permutations; full
R2C H fits only at full support. Pack dense H into the retained shape before
convolution. Real grids are also contiguous with Z fastest. The header defines
integer scaling and aliasing rules. Unsupported devices and dimensions are
rejected. `CUPO_COMPACT_ARCHS` follows `CMAKE_CUDA_ARCHITECTURES` unless explicitly
overridden. Supported build targets are SM75, SM80, SM86, SM89, SM90, SM100,
SM103 and SM120. The `compact_architectures` integer array in
`build/runtime_build.json` records the actual selection; `[]` means the backend
is disabled. Call `cupo_convolution_check_mesh` to check a mesh on the current
CUDA device. Scalar object architectures are selected independently and do not
add compact targets.

Compact builds require CUDA 13.1+ and MathDx 26.06. Correctness and performance
have been measured on SM120; the other build targets have not been GPU-qualified.
The producer links
device implementations into the shared library using offline cuFFT LTO
extraction. Consumers link the unified `libcupo.so` and need its CUDA/system
runtime dependencies, including cuFFT, NVRTC, nvJitLink and the NVIDIA driver.
The C API needs no MathDx headers, Python or OpenMM. Convolution is compiled
with normal FP32 operations and can accompany either scalar LTO-IR variant;
it is not part of the consumer's scalar device-link step.

For CMake consumers, point `CMAKE_PREFIX_PATH` at the unpacked tarball's root. The
package is in `share/cmake/CupoConvolution/` and uses the runtime package in
`build/cmake/CupoRuntime/` to link `build/libcupo.so`:

```bash
cmake -S . -B out -DCMAKE_PREFIX_PATH=/path/to/cupo-<ver>+...
```

```cmake
find_package(CupoConvolution CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE Cupo::Convolution)
```

The imported `Cupo::Convolution` target supplies the headers and links
`Cupo::Runtime`, using the unified runtime and its CUDA dependencies. Package
version macros in `cupo/package_version.h` share the root `VERSION` with the
scalar headers.

## Scalar LTO-IR Build Requirements

- CUDA Toolkit >= 12.0 (device linker must support LTO-IR)
- Compile your `.cu` files with `-dc` (relocatable device code)

## Standard vs Fast-Math Variants

Two variants of `cupo_math.cu.o` are shipped, with identical public functions:

- **`build/ltoir/standard/`** — compiled without `--use_fast_math`
  (`-ftz=0 -prec_div=1 -prec_sqrt=1`).  Use when your kernels are
  also compiled without fast-math.
- **`build/ltoir/fast_math/`** — compiled with `--use_fast_math`
  (`-ftz=1 -prec_div=0 -prec_sqrt=0`).  Use when your kernels use
  `--use_fast_math`.

**Pick the variant that matches your build.**  Mixing `-ftz` settings
between compilation units causes nvlink errors at device-link time.
Performance difference is <5% (only `sqrt` and division affected;
Horner FMA chains are identical either way).

## Run the shipped sample

The archive includes the complete Coulomb/dispersion particle-mesh example,
with energies, forces and virials compared against an independent reference.
From the root of a compact-enabled cupo tarball, with `nvcc` and `python3` on
`PATH` and a CUDA device whose architecture is included in the math object:

```bash
./samples/ltoir-usage/run              # standard variant, water_10k
./samples/ltoir-usage/run fast_math
```

Use a CUDA Toolkit compatible with the object's producer version, recorded in
`.cuda-version`; the same producer version is recommended, or a newer compatible
toolkit. The script detects the selected GPU, compiles `usage.cu`, and links it
with the selected `cupo_math.cu.o` and `libcupo.so`. It runs the exact reciprocal
and host particle-mesh calculations, then GPU compact convolution on a device
listed in `build/runtime_build.json` under `compact_architectures`. Scalar calls,
spreading, interpolation and energy/virial reductions run on the host. The
compact path uses FP32 GPU transforms and influence multiplication.

The launcher accepts `[standard|fast_math]`, defaulting to `standard`, and uses
the packaged `water_10k` reference. The launcher uses Python 3's standard library
to read build metadata; it needs no NumPy, OpenMM, JAX or external input files.
The system has 3,375 waters (10,125 atoms); its PDB and reference ship in
`data/water_10k/` under Apache-2.0. The optional generator
and its Ewald helper regenerate this reference by default, or accept
user-supplied net-neutral, orthorhombic periodic PDB structures with NumPy,
OpenMM and JAX. `REF_BIN` selects another reference;
a missing reference is an error.
Set `BUILD_DIR` to write build output outside the archive. Compilation, CUDA
and numerical-check failures return a nonzero exit status. The full example
requires compact convolution in `libcupo.so`, including for its host modes.
See [the sample instructions](../samples/ltoir-usage/README.md) and
[particle-mesh guide](ltoir-usage.md) for details.

## Integration

1. Include the headers of the functions you use in your `.cu` kernel files:
   `<cupo/math/splitting.h>` and `<cupo/math/spreading.h>`,
   with the tarball's `include/` on the include path
2. Device-link your kernel objects with the selected `cupo_math.cu.o`
   (`-dlto`); the same object also provides host implementations
   linked normally. Scalar-only consumers do not need `libcupo.so`.
3. For grid planning, include `<cupo/kspace.h>` with the CUDA Toolkit headers
   available, create a splitting plan, and link `libcupo.so` (or the CMake target
   `Cupo::Runtime`). Planning is a host runtime call and is independent of the
   scalar object's device link.

### Usage Example

```cpp
#include <cupo/math/splitting.h>

// Kernel: every device function takes a scalar cutoff.
float sr   = cupo_p1_short_range_value_f32(r2, cutoff);
float fk   = cupo_p1_fourier_kernel_value_f32(k2, cutoff);
float self = cupo_p1_self_value_f32(cutoff);   // works from host too
```

### CMake

When building cupo from source, use its object-library target:

```cmake
target_link_libraries(my_kernel PRIVATE
    cupo_device   # standard cupo_math.cu.o; cupo_device_fm selects fast_math
)
```

### Raw nvlink (command-line)

For `fast_math`, add `--use_fast_math` to both nvcc invocations.

```bash
# Choose variant: standard/ or fast_math/
VARIANT=standard

# 1. Compile consumer kernel with relocatable device code
nvcc -dc -O3 \
    -gencode arch=compute_89,code=lto_89 \
    my_kernel.cu -o my_kernel.o

# 2. Device-link consumer + the selected LTO-IR object
nvcc -dlto -O3 \
    -gencode arch=compute_89,code=sm_89 \
    my_kernel.o \
    build/ltoir/$VARIANT/cupo_math.cu.o \
    -o my_program
```

### nvJitLink (runtime linking)

```cpp
#include <nvJitLink.h>

nvJitLinkHandle handle;
nvJitLinkCreate(&handle, num_opts, opts);  // e.g. {"-arch=sm_89", "-lto"}

// Add the consumer's raw LTO-IR.
nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR,
    my_kernel_ltoir, my_kernel_size, "my_kernel");

// Add the cupo object containing LTO-IR (choose standard or fast_math).
nvJitLinkAddFile(handle, NVJITLINK_INPUT_OBJECT,
    "build/ltoir/standard/cupo_math.cu.o");

// Link and extract final cubin
nvJitLinkComplete(handle);
size_t cubin_size;
nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
// ... load cubin via cuModuleLoadData()
```

Both paths produce equivalent results — the LTO optimizer inlines the
device functions at link time regardless of the linking method.

## Particle-Mesh Assembly

The scalar functions are the pieces of a particle-mesh implementation; this is how
their returns fit together for both Coulomb and dispersion.

### Particle-mesh assembly recipe

For both P=1 and P=6 the total decomposes as

    E_total = E_short + E_recip + E_self − E_excl_LR

where each piece comes from one corner of the public API:

| Term         | Loop shape              | API calls                                           |
|--------------|-------------------------|-----------------------------------------------------|
| `E_short`    | O(N²) pair loop, in cutoff, excluding 1-2/1-3 | `cupo_p{1,6}_short_range_value/_grad_f64`           |
| `E_recip`    | O(N · K³) structure-factor sum | `cupo_p{1,6}_fourier_kernel_value_f64`              |
| `E_self`     | O(N) Σ qᵢ² scalar       | `cupo_p{1,6}_self_value_f64`                        |
| `E_excl_LR`  | O(M) bonded-pair loop   | `cupo_p{1,6}_long_range_value/_grad_f64`            |
| `E_plasma`   | O(N) Σ qᵢ scalar, 0 if neutral | `cupo_p1_fourier_kernel_dc_f64` (constant; caller assembles) |

The reciprocal sum can be computed via an exact-reciprocal recipe (eg. in the style of the classic
Ewald Sum) or via particle mesh methods (eg. in the style of smooth particle-mesh Ewald). 

For each integer-triple **m** ∈ [−M, M]³ the loop computes

    k_θ = m / L                                  (cycles/length)
    ker = cupo_p1_fourier_kernel_value_f64(k², rc)
    S(k) = Σ_j qⱼ · exp(2πi k·rⱼ)                 (C + iD)
    E_recip += (k_e / 2V) · ker · |S|²

This direct structure-factor sum is evaluated on a finite reciprocal shell.
The C2 kernel is not zero beyond that shell, so its convergence must be checked
by increasing the bound. Real-world consumers should use the FFT pipeline; the
direct form makes the math visible without spreading-kernel deconvolution.

### Dispersion sign and combining conventions

Note that dispersion is assumed to use only geometric combining rules for both
real space and reciprocal space. The repulsive term is omitted.

However, OpenMM's LJPME implementation includes repulsive terms, uses
arithmetic combining rules in real space, geometric combining rules in
reciprocal space, and a correction term to account for the mismatch. 

```cpp
// Per-atom LJ "charge" amplitudes with geometric combining:
//   B_i = 4 ε_i σ_i⁶,  √B_i = 2 √ε_i · σ_i³   (paired as √B_i · √B_j)
//   A_i = 4 ε_i σ_i¹², √A_i = 2 √ε_i · σ_i⁶
double sqrtA_i = 2.0 * std::sqrt(epsilon_i) * std::pow(sigma_i, 6);
double sqrtB_i = 2.0 * std::sqrt(epsilon_i) * std::pow(sigma_i, 3);
```

The prolate P=6 kernels split the bare **+r⁻⁶** interaction; the caller
applies the attractive dispersion sign explicitly. Individual split values,
including the oscillatory Fourier tail, need not be positive:

```cpp
// Per-pair (i, j) short-range:
//   U_pair    = -B_ij · cupo_p6_short_range_value(r²)
//   force/r   = +2·B_ij · cupo_p6_short_range_grad(r²)   (grad is dS/d(r²))
double sqrtB_ij = sqrtB_i * sqrtB_j;
double U   = -sqrtB_ij * cupo_p6_short_range_value_f64(r2, rc);
double dF  = +2.0 * sqrtB_ij * cupo_p6_short_range_grad_f64(r2, rc);
```

The `E_excl_LR` term is `−Σ √B_i √B_j · long_range_value(r²)` over the
exclusion list.  For water with TIP3P-FB (ε_H = 0) this vanishes, but
the loop is still written generally so the code transfers to systems
with non-zero LJ on every atom.

### The k = 0 bin subtlety

The Coulomb (P=1) Fourier kernel returns 0 at `k² = 0` by charge-
neutrality convention.  The dispersion (P=6) kernel returns a
**finite, positive** value at the DC bin, and `S_disp(0) = Σ √B_i ≠ 0`,
so that bin contributes to reciprocal energy. Its spatially constant mode
does not contribute to particle forces.

## Scalar LTO-IR Architecture Compatibility

LTO-IR objects are architecture-specific.  The shipped CUDA-13 `.cu.o` files
contain NVVM IR sections for `compute_{75,80,86,89,90,100,103,120}`:

- **sm_75**: Turing (T4, RTX 20-series)
- **sm_80 / sm_86**: Ampere (A100; RTX 30-series, A40)
- **sm_89**: Ada Lovelace (RTX 4090, L40)
- **sm_90**: Hopper (H100, H200)
- **sm_100 / sm_103**: supported datacenter Blackwell targets
- **sm_120**: RTX Blackwell (RTX 50-series)

These are the runtime's architectures. To build the objects for others, set
`CUPO_LTOIR_ARCHS` (a semicolon-separated SM list) when building cupo; each
becomes a `-gencode arch=compute_XX,code=lto_XX` flag. These are scalar-object
targets, not additional compact-library targets.
