**English** | [日本語](README.ja.md) | [中文](README.zh.md)

# cuPotential OpenMM plugin — OpenMM 8.6 lab build (`0.3.5+openmm86`)

This repository is a lab fork of [NVIDIA cuPotential](https://github.com/NVIDIA/cuPotential) **v0.3.5**. Its OpenMM plugin is rebuilt for **OpenMM 8.6**, with one correction to energy minimization. It is **not an NVIDIA release**. NVIDIA's 0.3.5, including `nv-openmm-plugin` on PyPI, targets OpenMM 8.4. NVIDIA's original README is kept in [README.NVIDIA.md](README.NVIDIA.md), and the [cuPotential documentation](https://docs.nvidia.com/cuda/cupotential) covers the API, theory and samples, which are unchanged here.

## What is different from NVIDIA 0.3.5

- **OpenMM 8.6.** The plugin's source builds against OpenMM 8.6 without changes. At build time, a CMake probe reads the OpenMM version and sets three things from it: the wheel's requirement `openmm[cuda13]>=8.6,<8.7`, the conda recipe's pin, and a check at import time. That check refuses to load the plugin under any other OpenMM series.
- **Energy minimization on OpenMM 8.5+.**
  - **What goes wrong:** from 8.5, OpenMM minimizes on the GPU. For a System with constraints, the restraint springs eventually overflow the fixed-point force buffer, and OpenMM falls back to a CPU copy of the whole System, where the plugin's forces have no kernels. Minimization then fails with `Specified a Platform for a Context which does not support all required kernels`.
  - **What this build does:** a System that contains plugin forces (`nv.NonbondedForce`, `nv.MonteCarloBarostat`) is minimized the OpenMM 8.4 way, with L-BFGS on the host and springs added in double precision. This is 8.4's algorithm, not the 8.5+ GPU minimizer, so it is slower than the GPU minimizer.
  - **Systems without plugin forces:** they use OpenMM's own GPU minimizer, unchanged.
  - **Overlapping atoms:** if a force component of the System itself reaches about 2e9 kJ/mol/nm, for example from overlapping atoms, you get an error that explains this instead of a CPU fallback.
- **Version `0.3.5+openmm86`.** This is a PEP 440 local label, so the package cannot be mistaken for NVIDIA's 0.3.5.

The API and usage are the same as NVIDIA 0.3.5.

## Requirements

| | |
|---|---|
| OS | Linux x86_64 |
| Python | **3.14 only**. The wheel is `cp314` because it ships a prebuilt SWIG extension |
| OpenMM | 8.6.x with the CUDA 13 platform. pip installs it from PyPI |
| CUDA libraries | CUDA runtime, NVRTC and nvJitLink `>=13.4,<13.5`, and cuFFT `>=12.4,<12.5`. pip installs them. No CUDA toolkit is needed |
| NVIDIA driver | One that supports CUDA 13.4. Tested with 615.71; 580.x is untested |
| GPU | sm_75–sm_120 (Turing to Blackwell). Tested on RTX 5090 only |

Do not install this build in the same environment as NVIDIA's `nv-openmm-plugin`, because the two share a package name.

## Install

Download `nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl` from this repository's **Releases** page (tag `v0.3.5+openmm86`). Then:

```bash
python3.14 -m venv omm86 && source omm86/bin/activate     # or: a conda env with python=3.14
pip install --upgrade pip
pip install nv_openmm_plugin-0.3.5+openmm86-cp314-cp314-linux_x86_64.whl
python -c "import openmm, nv_openmm_plugin; print(openmm.version.full_version, [f for f in openmm.Platform.getPluginLoadFailures() if 'nv_openmm_plugin' in f])"
```

The last command should print an 8.6.x version and `[]`. You do not need `LD_LIBRARY_PATH`.

Installing into an existing conda-forge OpenMM 8.6 environment with `pip install --no-deps` is untested.

## Verified

The environment was an RTX 5090, driver 615.71, OpenMM 8.6.1, CUDA 13.4.92 and Python 3.14.7.

| What was run | Result |
|---|---|
| The wheel, installed into a fresh venv | Import OK, and all CUDA libraries are loaded from the venv |
| `samples/solvated_systems`, from the wheel | Pass |
| Installed-plugin tests `test_unit_conversion` / `test_npt`, from the wheel | Pass |
| Plugin test suite, from the source tree | All pass, except `test_leak_detection`, which needs `cupy` and was not run |
| Benchmarks, from the source tree, GPU clocks not locked | 2.3x–3.8x per step over stock OpenMM 8.6.1 on the same GPU |

Release notes: [plugins/openmm/docs/RELEASE-NOTES-0.3.5+openmm86.md](plugins/openmm/docs/RELEASE-NOTES-0.3.5+openmm86.md).

## Build from source (maintainers)

You need these:

- a conda environment with `openmm=8.6`, `python=3.14`, `swig`, `cmake>=3.24` and `cuda-version=13.4`;
- the CUDA 13.4 toolkit (`nvcc`);
- gcc.

Docker is not needed.

```bash
E=<that conda env>
export PATH=<cuda-13.4>/bin:$E/bin:$PATH OPENMM=$E
CUPO_REPOSITORY=NVIDIA/cuPotential python3 fetch_cupo      # NVIDIA's v0.3.5 cupo runtime (CUDA 13.4)
plugins/openmm/run_build
GITHUB_RELEASE_TAG=v0.3.5 python -m pip wheel --no-build-isolation --no-deps -w dist plugins/openmm
```

The root `VERSION`/`RELEASE_TAG` stay `0.3.5`/`v0.3.5`, because `fetch_cupo` uses them to download the cupo runtime. The package version is in `plugins/openmm/VERSION`.

## Licensing

The licensing is the same as NVIDIA cuPotential 0.3.5; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The new minimizer follows OpenMM 8.4.0's `LocalEnergyMinimizer.cpp` (MIT) and includes OpenMM's copy of the libLBFGS header (MIT). Both are listed in `NOTICE`.
