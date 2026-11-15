# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pytest configuration for openmm plugin tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent.parent

# The repository root, for the datasets (data/datasets.py, data/tyk2_alchemical.py),
# and the core tests' directory, for the few helpers the plugin's tests share with
# them: the observed-error table (observed.py), the benchmarks' locked clocks
# (gpu_clocks.py, hw_config.py) and the memory tracker (mem_tracker.py). Those are
# all this suite takes from tests/; it does not load the core tests' conftest, which
# sets up JAX and the reference implementation this suite does not use. The plugin's
# source tree carries these files and not the rest of tests/.
for _path in (_REPO, _REPO / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# And the built plugin package, so `import nv_openmm_plugin` works in a test module
# without the caller having set PYTHONPATH. Here rather than in the test modules
# because conftest is imported before any of them: several modules used to do this
# themselves just before importing the plugin, which left every module that did not
# depending on being collected after one that did. Alphabetical order was holding
# that up, and a test module whose name sorted first found no plugin and failed the
# whole collection.
#
# With CUPO_TEST_INSTALLED=1 the tests test the plugin pip or conda installed
# instead: the path is left alone, and nv_openmm_plugin is imported from wherever
# it was installed, with the libraries installed with it.
_PLUGIN_PYTHON = str(Path(__file__).resolve().parent.parent / "python")
TEST_INSTALLED = os.environ.get("CUPO_TEST_INSTALLED", "") not in ("", "0")
if not TEST_INSTALLED and _PLUGIN_PYTHON not in sys.path:
    sys.path.insert(0, _PLUGIN_PYTHON)

# What a child Python process puts first on sys.path to import the plugin under
# test: the tree's python/ directory, or, with CUPO_TEST_INSTALLED=1, nothing, as the
# installed plugin is on the path already.
PLUGIN_PATH = [] if TEST_INSTALLED else [_PLUGIN_PYTHON]

def toolkit_required(environ=None):
    """Whether a CUDA toolkit tool a test needs, such as cuda-gdb, must be there:
    in the dev container testing the tree's build (CUPO_DEV_CONTAINER=1, without
    CUPO_TEST_INSTALLED), whose image has the toolkit, its absence means the image
    is broken. Elsewhere, as for an installed package in an environment without
    the toolkit, the test skips."""
    environ = os.environ if environ is None else environ
    installed = environ.get("CUPO_TEST_INSTALLED", "") not in ("", "0")
    return environ.get("CUPO_DEV_CONTAINER") == "1" and not installed


# For a test that needs what only the tree has, such as its build directory.
tree_only = pytest.mark.skipif(
    TEST_INSTALLED, reason="needs the tree's own build; skipped with CUPO_TEST_INSTALLED=1")


def tested_plugin():
    """(which, the module's path, its library directory, its version) of the plugin
    the session tests: "installed" with CUPO_TEST_INSTALLED=1, else "tree build".
    Raises pytest.UsageError when the variable is set and no installed plugin
    imports, or the one that imports is the tree's."""
    which = "installed" if TEST_INSTALLED else "tree build"
    try:
        import nv_openmm_plugin as nv
    except ImportError as e:
        if TEST_INSTALLED:
            raise pytest.UsageError(
                "CUPO_TEST_INSTALLED=1, but no installed nv_openmm_plugin imports (%s): "
                "install the plugin with pip or conda first, or unset the variable to "
                "test the tree's build" % e)
        raise
    module = Path(nv.__file__).resolve()
    if TEST_INSTALLED and Path(_PLUGIN_PYTHON) in module.parents:
        raise pytest.UsageError(
            "CUPO_TEST_INSTALLED=1, but nv_openmm_plugin imports from the tree's "
            "build, %s: install the plugin with pip or conda, and do not put %s on "
            "PYTHONPATH" % (module, _PLUGIN_PYTHON))
    try:
        from importlib import metadata
        version = metadata.version("nv-openmm-plugin")
    except Exception:
        version = (Path(__file__).resolve().parent.parent / "VERSION").read_text().strip()
    return which, module, Path(getattr(nv, "_PLUGIN_DIR", module.parent)), version


def pytest_report_header(config):
    which, module, library, version = tested_plugin()
    return ["nv_openmm_plugin under test: %s, version %s" % (which, version),
            "  module:  %s" % module, "  library: %s" % library]

from observed import record_observed, write_observed_docs  # noqa: E402,F401
# The benchmarks' locked clocks and GPU telemetry (tests/gpu_clocks.py).
import gpu_clocks  # noqa: E402



@pytest.fixture(scope="session")
def gpu_hw():
    """The session's BenchClocks, with the GPU locked, as tests/gpu_clocks.py's own
    fixture of that name; clocks reset and telemetry stopped at the end.

    Where the clocks cannot be locked -- that takes passwordless sudo for nvidia-smi
    and, in a container, --cap-add=SYS_ADMIN -- the benchmarks fail, saying how to
    fix it (gpu_clocks.lock_failure_message). With CUPO_ALLOW_UNLOCKED_CLOCKS=1 they
    run at whatever clocks the GPU has instead, marked "clocks not locked", and
    compare with nothing (test_bench_plugin.py's _compared_entry)."""
    clocks = gpu_clocks.BenchClocks()
    try:
        try:
            clocks.lock()
        except RuntimeError as e:
            if not gpu_clocks.allow_unlocked():
                pytest.fail(gpu_clocks.lock_failure_message(e), pytrace=False)
            clocks.run_unlocked()
        yield clocks
    finally:
        gpu_clocks.unlock()
        clocks.close()


def pytest_terminal_summary(terminalreporter):
    gpu_clocks.report_suggestions(terminalreporter)


@pytest.fixture(autouse=True, scope="session")
def _assert_gpu_exclusive():
    """Fail fast if another user's process is on our GPU."""
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-compute-apps=pid,used_gpu_memory,process_name",
         "--format=csv,noheader"],
        text=True, timeout=5,
    ).strip()
    if out:
        pytest.fail(
            f"GPU is not exclusive — other processes detected:\n{out}\n"
            "Kill zombie processes or run on another GPU."
        )


@pytest.fixture(scope="session")
def plugin_dir():
    """Ensure the NvidiaForce plugin is loaded; return its directory.

    There is no --plugin-dir any more: nv_openmm_plugin loads the libraries that
    ship with it, so the tests exercise the build that sits alongside the Python
    package (plugins/openmm/build/cpp) and cannot be pointed at a second
    one. Kept as a fixture so tests have an explicit "plugin is up" dependency.
    """
    # Importing loads the libraries and registers the CUDA kernels. This fixture
    # gives tests an explicit "the plugin stack is available" dependency to declare.
    # The module is on the path already: the tree's (above), or the installed one.
    import nv_openmm_plugin
    return Path(getattr(nv_openmm_plugin, "_PLUGIN_DIR",
                        Path(nv_openmm_plugin.__file__).resolve().parent))


def replace_nonbonded(system, npt_box_safety_factor=1.0, method=None):
    """Swap a System's single NonbondedForce for an NvNonbondedForce, in place.

    A TEST helper, not part of the plugin's API -- deliberately. The package
    exposes the force class and nothing else, because converting is just
    constructing one, and a System-level helper has to guess three things the
    caller knows better: which NonbondedForce (a System may hold several),
    whether to remove the original, and where the replacement goes.

    Test systems all have exactly one and always want it replaced, so those
    guesses are safe HERE and nowhere else. Asserts the "exactly one" part rather
    than assuming it.

    `method` is an NvOpenMM NonbondedMethod (PME or Prolate); None keeps the
    converting constructor's own default, which is PROLATES -- the splitting the
    plugin ships. So saying nothing runs what a user gets, and the tests that
    compare against STOCK have to ask for PME by name, since a different
    decomposition disagrees with Ewald by its own truncation error rather than by
    round-off. This helper chooses the splitting at construction; changing it
    later through setLatticeSumInteraction requires reinitializing the Context.
    """
    import openmm as mm
    import nv_openmm_plugin as nv

    found = [i for i in range(system.getNumForces())
             if isinstance(system.getForce(i), mm.NonbondedForce)]
    assert len(found) == 1, f"expected exactly one NonbondedForce, found {len(found)}"
    index = found[0]
    if method is None:
        force = nv.NonbondedForce(system.getForce(index), npt_box_safety_factor)
    else:
        force = nv.NonbondedForce(system.getForce(index), npt_box_safety_factor,
                                  method)
    system.addForce(force)
    system.removeForce(index)
    return system


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "benchmark: performance benchmark (deselect: -m 'not benchmark')")


def pytest_sessionfinish(session, exitstatus):
    write_observed_docs()
