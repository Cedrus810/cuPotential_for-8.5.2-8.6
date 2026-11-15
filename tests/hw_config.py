# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The clocks the benchmarks lock the GPU at, per GPU (tests/gpu_clocks.py).

Keyed by the GPU name NVML reports (nvidia-smi --query-gpu=name), the name the
benchmarks can see from inside the container. Values are (graphics MHz, memory
MHz), as passed to nvidia-smi -lgc and -lmc. Each should be within 2% of a clock
the card offers (nvidia-smi --query-supported-clocks=memory,graphics); when it is
not, the benchmarks run at what the card offers, with a warning in the golden and
on the plots, and print the line to put here. Cards of one SKU do not all offer
the same clocks: two L40S cards can differ in the memory clocks they offer.

The memory values are the clocks the cards ran at in the committed goldens.

A lock is a ceiling, not a floor. A card at its power or temperature limit runs
below it under load; what it actually ran at is recorded with each measurement.

A GPU missing here is locked at 95% of its maximum clocks, and the session prints
the line to add here.

Extra entries are read from the file that cupo-repo.ini at the repository root
names, when present ([tests] extra_hw_config), and added to these.
"""

CLOCKS = {
    "NVIDIA GeForce RTX 5090": (2392, 14001),
    "NVIDIA RTX 6000 Ada Generation": (2300, 9500),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": (2308, 12481),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": (2392, 13365),
    "NVIDIA RTX A6000": (1995, 7600),
    "NVIDIA L40": (2365, 9001),
    "NVIDIA L40S": (2394, 9001),
    "NVIDIA A100-PCIE-40GB": (1339, 1215),
    "NVIDIA B200": (1866, 3996),
    "NVIDIA GB300": (1966, 3996),
    "NVIDIA H100 PCIe": (1667, 1593),
    "Tesla T4": (1510, 5000),
    "NVIDIA A2": (1681, 6251),
    "NVIDIA A10": (1610, 6251),
    "NVIDIA L2": (1938, 6251),
    "NVIDIA L4": (1938, 6251),
    "NVIDIA L20": (2394, 9001),
    "NVIDIA H20": (1881, 2619),
    "NVIDIA H200": (1881, 3201),
    "NVIDIA RTX 5000 Ada Generation": (2949, 8550),
}


def _extra_clocks(repo_dir=None):
    """The CLOCKS of the file [tests] extra_hw_config in cupo-repo.ini at the
    repository root names, relative to it; {} when there is no such file."""
    import configparser
    import importlib.util
    from pathlib import Path
    repo_dir = Path(repo_dir or Path(__file__).resolve().parents[1])
    config = configparser.ConfigParser()
    config.read(repo_dir / "cupo-repo.ini")
    name = config.get("tests", "extra_hw_config", fallback="").strip()
    if not name or not (repo_dir / name).is_file():
        return {}
    spec = importlib.util.spec_from_file_location("hw_config_extra", repo_dir / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.CLOCKS)


CLOCKS.update(_extra_clocks())
