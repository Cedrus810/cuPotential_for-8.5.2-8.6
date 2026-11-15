# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Does the fused integrator kernel fit its register cap without spilling?

fusedLangevinMiddleStep runs at a register cap set by its launch bounds (96 per
thread on sm_89 and sm_90 for the kernel with SHAKE units). When the float-float
instantiation needs more, ptxas spills to local memory, and the step slows down
by a few percent: on the RTX 6000 Ada, H20 and L20 that was the whole gap to the
earlier integrator. The spills come back silently with any change that keeps
more values live, so this test reads the built library's resource usage and
fails when a fused kernel for sm_89 or sm_90 has a stack frame.

The check runs on the host (cuobjdump), not on the GPU, so it covers every
architecture the library was built for, whatever card the test runs on. The
native-double instantiations are left out: they run only on GPUs with fast FP64,
where the cost of a spill is a separate question.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

_ARCHS = ("sm_89", "sm_90")
_KERNEL = "fusedLangevinMiddleStep"


def _library():
    import nv_openmm_plugin as nv
    return nv._cuda_platform_dir(nv._PLUGIN_DIR) / "libOpenMMNvidiaCUDA.so"


def _fused_kernels():
    """(arch, mangled name, stack bytes, local bytes) of each fused kernel image."""
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is not on PATH; cannot read the kernels' resource usage")
    proc = subprocess.run([cuobjdump, "--dump-resource-usage", str(_library())],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"cuobjdump failed:\n{proc.stderr}"
    kernels, arch, elf, name = [], None, False, None
    for line in proc.stdout.splitlines():
        if line.startswith("Fatbin "):
            elf = line.startswith("Fatbin elf code")
        elif m := re.match(r"arch = (sm_\d+)", line):
            arch = m[1]
        elif m := re.match(r"\s*Function (\S+):", line):
            name = m[1]
        elif elf and name and _KERNEL in name and (m := re.search(r"STACK:(\d+).*LOCAL:(\d+)", line)):
            kernels.append((arch, name, int(m[1]), int(m[2])))
            name = None
    return kernels


def test_fused_kernel_does_not_spill():
    # float (If) and float-float (N4matx6fltflt) instantiations; double (Id) is left out.
    checked = [k for k in _fused_kernels()
               if k[0] in _ARCHS and (f"{_KERNEL}If" in k[1] or f"{_KERNEL}IN4matx6fltflt" in k[1])]
    if not checked:
        pytest.skip(f"the library has no {' or '.join(_ARCHS)} image of {_KERNEL}")
    spilled = [k for k in checked if k[2] or k[3]]
    assert not spilled, "fused kernels with a stack frame (register spills):\n" + "\n".join(
        f"  {arch} {name}: stack {stack} B, local {local} B" for arch, name, stack, local in spilled)
