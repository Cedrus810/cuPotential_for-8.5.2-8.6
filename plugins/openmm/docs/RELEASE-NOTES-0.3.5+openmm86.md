# nv-openmm-plugin 0.3.5+openmm86 (OpenMM 8.6, lab build)

This is the cuPotential OpenMM plugin from NVIDIA cuPotential **v0.3.5**, rebuilt for **OpenMM 8.6** with one correction. It is a lab build, **not an NVIDIA release**. NVIDIA's 0.3.5, including `nv-openmm-plugin` on PyPI, targets OpenMM 8.4.

## What changed relative to NVIDIA 0.3.5

- **OpenMM 8.6 support.** The plugin's source builds against OpenMM 8.6 unchanged. At build time, a CMake probe reads the OpenMM version, and the result sets three things: the wheel's requirement `openmm[cuda13]>=8.6,<8.7`, the conda recipe's pin, and a check at import time. That check refuses to load the plugin under any other OpenMM series.
- **Energy minimization fix for OpenMM 8.5+.**
  - **The problem:** from OpenMM 8.5, minimization runs on the GPU (`CommonMinimizeKernel`). For a System with constraints, it adds restraint springs into the fixed-point force buffer. Once the spring constant has grown far enough, those forces overflow. OpenMM then rebuilds the whole System on the CPU/Reference platform, where the plugin's forces have no kernels. Minimization fails with `Specified a Platform for a Context which does not support all required kernels`.
  - **The fix:** the plugin now provides OpenMM's `Minimize` kernel on CUDA.
    - A System *without* plugin forces uses OpenMM's own GPU minimizer, unchanged.
    - A System *with* plugin forces (`nv.NonbondedForce`, `nv.MonteCarloBarostat`) is minimized the way OpenMM 8.4 did it. L-BFGS runs on the host, and the constraint springs are added on the host in double precision. This is 8.4's algorithm; these Systems do not get the 8.5+ GPU minimizer, so they minimize more slowly than with it.
    - If a force component of the System itself reaches about 2e9 kJ/mol/nm, for example from overlapping atoms, minimization now raises an error that explains this. It no longer falls back to the CPU.
- **Version** `0.3.5+openmm86`: a PEP 440 local label, so this build cannot be mistaken for NVIDIA's 0.3.5.

The API and usage are identical to NVIDIA 0.3.5. See the [cuPotential documentation](https://docs.nvidia.com/cuda/cupotential) and `plugins/openmm/samples/`.

## Requirements

| | |
|---|---|
| OS | Linux x86_64 (the wheel's tag is `linux_x86_64`, not manylinux) |
| Python | **3.14 only**. The wheel is `cp314-cp314` because it contains a prebuilt SWIG extension |
| OpenMM | 8.6.x with the CUDA 13 platform (`openmm[cuda13]>=8.6,<8.7`) |
| CUDA libraries | `nvidia-cuda-runtime`, `nvidia-cuda-nvrtc` and `nvidia-nvjitlink` `>=13.4,<13.5`, and `nvidia-cufft` `>=12.4,<12.5`. pip installs them. The CUDA minor version is fixed because cupo's embedded LTO IR is tied to the NVRTC version |
| NVIDIA driver | One that supports CUDA 13.4. OpenMM compiles its kernels with NVRTC, and an older driver rejects the PTX (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`). Tested with 615.71; 580.x is untested |
| GPU | sm_75 / 80 / 86 / 89 / 90 / 100 / 103 / 120 (Turing to Blackwell, as built into the cupo runtime). Tested on RTX 5090 only |

Do not install this build in the same environment as NVIDIA's `nv-openmm-plugin`, because they share the package name. If NVIDIA's version is already installed there, run `pip uninstall nv-openmm-plugin` first.

## Install

```bash
python3.14 -m venv omm86 && source omm86/bin/activate     # or a conda env with python=3.14
pip install --upgrade pip
pip install nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl
python -c "import openmm, nv_openmm_plugin; print(openmm.version.full_version, [f for f in openmm.Platform.getPluginLoadFailures() if 'nv_openmm_plugin' in f])"
```

pip resolves OpenMM 8.6.x and the CUDA 13.4 runtime from PyPI. You do not need a CUDA toolkit or `LD_LIBRARY_PATH`. The check should print an 8.6.x version and `[]`.

Installing into an existing conda-forge OpenMM 8.6 environment with `pip install --no-deps` should also work, provided that environment already has the CUDA 13.4 libraries listed above. This route is **untested**.

## Verified

The environment was an RTX 5090, driver 615.71, OpenMM 8.6.1, CUDA 13.4.92 and Python 3.14.7.

| What was run | Result |
|---|---|
| This wheel, installed into a fresh venv (no `LD_LIBRARY_PATH`/`PYTHONPATH`) | Import OK, 0 plugin load failures, all CUDA libraries loaded from the venv's `nvidia/cu13/lib` |
| `samples/solvated_systems`, from the wheel | Pass |
| `CUPO_TEST_INSTALLED=1` `test_unit_conversion` / `test_npt`, from the wheel | 14 / 14 passed |
| Source tree, plugin test suite (non-benchmark) | All pass, except `test_leak_detection`, which needs `cupy` and was not run |
| Source tree, `samples/solvated_systems` and `samples/alchemical` | Pass |
| Source tree, benchmarks with GPU clocks **not** locked | Per-step speedup over stock OpenMM 8.6.1 on the same GPU, across all benchmark systems: 2.3x–3.8x |

## Known limitations

- Only the configuration above has been tested. Other GPUs, older drivers and the conda `--no-deps` route are untested.
- Systems with plugin forces minimize with host-side L-BFGS, the OpenMM 8.4 algorithm, rather than the faster 8.5+ GPU minimizer.
- There is no conda package for this build. It is distributed as a wheel only.

## File

`nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl`: 599,248,028 bytes, SHA-256 `c972481c489472d0449cfaa656ab2e303b7f16306246853c0607935dc48f4662`

## Licensing

Same as NVIDIA cuPotential 0.3.5; see `LICENSE` and `NOTICE`. The new minimizer follows OpenMM 8.4.0's `LocalEnergyMinimizer.cpp` (MIT) and includes OpenMM's copy of the libLBFGS header (MIT). Both are listed in `NOTICE`.
