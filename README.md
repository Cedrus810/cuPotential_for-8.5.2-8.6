<p align="center">
  <img src="plugins/openmm/docs/cupotential-openmm.png" alt="Bar chart: the speedup of OpenMM 8.4.0 with cuPotential over stock OpenMM, NPT in mixed precision, for six systems of 24k to 409k atoms on seven NVIDIA GPUs (RTX PRO 6000 Blackwell Server, L40S, A10, T4, B200, H100 PCIe and A100 PCIe). The largest speedup on each GPU is between 2.64x and 3.99x." width="720">
</p>

# NVIDIA cuPotential

cuPotential is a CUDA-X library for efficient computation of molecular interactions
found in standard forcefields for molecular dynamics. Its OpenMM plugin provides
nonbonded forces, an integrator, and a barostat backed by the cuPotential runtime.

**[cuPotential documentation](https://docs.nvidia.com/cuda/cupotential)** contains
installation instructions, examples, API guides, theory, and benchmarks.

- [Installation](https://docs.nvidia.com/cuda/cupotential/install)
- [OpenMM NonbondedForce: standard and custom interactions](https://docs.nvidia.com/cuda/cupotential/nonbonded-force)
- [C API (Experimental)](https://docs.nvidia.com/cuda/cupotential/c-api/getting-started)
- [Releases](https://github.com/NVIDIA/cuPotential/releases)

cuPotential is in early-access mode. Its C API is experimental: expect breaking
changes, including to its structure, in upcoming releases.

## Licensing

See [LICENSE](LICENSE) for the component licensing overview and links to the
full license texts.
