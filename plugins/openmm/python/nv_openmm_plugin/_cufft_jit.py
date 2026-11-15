# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Making cuFFT's JIT toolchain resident before the first reciprocal-space plan.

cuFFT loads its JIT toolchain when it compiles the LTO callback cuPo registers.
libcupo now directly links nvJitLink; NVRTC is also needed by libcupo's expression
compiler and by cuFFT. Explicit preloading controls which toolchain libraries
become resident before other GPU packages are loaded.

A conda cuFFT may not find the copies installed under site-packages/nvidia through
its own RUNPATH. Loading matching SONAMEs here lets its later dlopen reuse them,
without changing the process search path. Missing dependencies can fail at library
load or during cuFFT plan creation, depending on which component first needs them.

Version compatibility matters as well as presence. cuFFT can require nvJitLink
entry points from a particular CUDA minor, so an older resident library may fail
even when its SONAME matches. A machine can hold a toolkit copy on the default
loader path and a newer one in a pip wheel. Later loads can reuse the first
resident library. This helper orders candidates; it does not verify that the
selected nvJitLink and NVRTC versions satisfy cuFFT's requirements.

Matched by stem, not by name. The file name carries the CUDA major version, and the
directory pip puts it in has already moved once (nvidia/nvjitlink/lib in the CUDA 12
wheels, nvidia/cu13/lib in the 13 ones). Pinning either here means this file has to
be edited for a CUDA bump that changes nothing else.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

_CUFFT_JIT_STEMS = ("libnvJitLink", "libnvrtc")


def _version_key(path: Path):
    # Everything after ".so.", as integers. A field that is not a number sorts
    # below every number, so an oddly-named file loses to a plain one rather
    # than raising.
    tail = path.name.partition(".so.")[2]
    return [int(f) if f.isdigit() else -1 for f in tail.split(".")]


def preload_cufft_jit_toolchain(searches) -> None:
    """Load the newest libnvJitLink/libnvrtc found, one per stem.

    `searches` is (root, pattern) pairs in priority order, where pattern carries a
    `{stem}` placeholder. Priority is between roots; WITHIN a root the newest
    version wins, keyed on the version FIELDS of the file name rather than on the
    name as a string ("libnvrtc.so.9" sorts above "libnvrtc.so.13" as text). Keyed
    on the name and not the path, because a path key would rank
    nvidia/nvjitlink/lib (a CUDA 12 wheel) above nvidia/cu13/lib purely on the
    directory spelling, and load the major cuFFT is not asking for.

    Best-effort by design. If nothing can be loaded here, normal library loading
    or cuFFT plan creation reports the dependency error. Roots are provided by
    the packaging adapter, including its environment lib/ and pip NVIDIA paths;
    preloading also controls which version becomes resident first.
    """
    for stem in _CUFFT_JIT_STEMS:
        candidates = []
        for root, pattern in searches:
            found = sorted(Path(root).glob(pattern.format(stem=stem)),
                           key=_version_key, reverse=True)
            candidates.extend(found)
        for candidate in candidates:
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue
