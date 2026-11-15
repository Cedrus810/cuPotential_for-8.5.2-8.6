<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Where the sample's systems come from

- `bace/solvated.pdb.gz` is the BACE protein of the JACS benchmark set as
  distributed by [OpenFreeEnergy/openfe-benchmarks](https://github.com/OpenFreeEnergy/openfe-benchmarks)
  (MIT, Copyright (c) 2022 Irfan Alibay), with missing atoms and hydrogens added,
  solvated by NVIDIA in TIP3P-FB water with 0.15 M NaCl and equilibrated with
  Amber14 and OpenMM 8.4; 69,983 atoms.

The sample's other two systems, apoa1 and dhfr, are not here and are not
distributed with the sample. They are OpenMM's benchmark systems
`examples/benchmarks/apoa1.pdb` and `examples/benchmarks/5dfr_solv-cube_equil.pdb`
([openmm/openmm](https://github.com/openmm/openmm)). The sample reads them from
your OpenMM installation (conda-forge's `openmm` package installs them under
`share/openmm/examples/benchmarks/`), or, when it has none, downloads them from
OpenMM's repository at release 8.4.0 when it runs, into
`~/.cache/nv-openmm-plugin/` (the plugin's `nv_openmm_plugin._benchmark_systems`).

## openfe-benchmarks licence

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
