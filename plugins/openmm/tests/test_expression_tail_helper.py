# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU/Reference regression for the private expression-tail helper.

Run without CUDA or GPU exclusivity checks with pytest --noconftest on this file.
Only the compiler and installed OpenMM development files used by the plugin build
are required.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import tree_only

# It compiles the tree's own source of the helper, not the plugin under test.
pytestmark = tree_only


@pytest.fixture(scope="module")
def expression_tail_driver(tmp_path_factory):
    plugin = Path(__file__).resolve().parents[1]
    source = plugin / "cpp/platforms/cuda/src/nonbonded"
    prefixes = [Path(os.environ.get("OPENMM_HOME", "/opt/openmm")), Path(sys.prefix)]
    openmm = next((p for p in prefixes if (p / "include/lepton/Parser.h").exists()
                   and (p / "lib/libOpenMM.so").exists()), None)
    assert openmm, "Install the OpenMM development files required by the plugin build"
    library = openmm / "lib"
    executable = tmp_path_factory.mktemp("expression-tail") / "driver"
    result = subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
        "-DNVOPENMM_TEST_EXPRESSION_TAIL",
        "-I" + str(openmm / "include"), "-I" + str(source),
        str(source / "ExpressionTailCorrection.cpp"),
        str(Path(__file__).with_name("ExpressionTailCorrectionDriver.cpp")),
        "-L" + str(library), "-Wl,-rpath," + str(library), "-lOpenMM",
        "-o", str(executable)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize("case", ["stock", "cache", "integrals", "force-only"])
def test_expression_tail_regressions(expression_tail_driver, case):
    result = subprocess.run([str(expression_tail_driver), case], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(result.stdout.splitlines()) == (6 if case == "stock" else 1)
    assert all(line.startswith("passed ") for line in result.stdout.splitlines())
