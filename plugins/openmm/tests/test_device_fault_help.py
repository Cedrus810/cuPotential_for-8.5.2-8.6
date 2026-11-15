# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A device fault must tell the caller how to find where it happened.

A bare __trap() -- how every contract check in these kernels ends -- poisons the
CUDA context, and the error then surfaces at whatever CUDA call runs next. That
is routinely the wrong place: provoking the cell list's finiteness check with a
NaN coordinate produces "cuda_charge2grid_cell_o4_4t8a launch: unspecified launch
failure", naming a k-space kernel three launches downstream of the one that
actually died. The name is not just missing, it is misleading.

cupo_check_cuda answers that by appending the recipe for recovering the real
site from outside the kernel, which costs the kernel nothing. Two tests here:
one that the recipe is issued, and one that follows it -- provokes the fault
under a GPU core dump, reads the dump back with cuda-gdb, and checks it names the
trapping kernel and thread, and the line too when cupo was built with line
information. The second is what stops the advice from rotting into something
that no longer works.

Every test runs in a child process, and has to: a trap kills the CUDA context
for the whole process, so provoking one in-process would fail every test after
it.

A real check is provoked rather than a test-only hook added for the purpose.
cuda_matrix_migrate rejects a position the wrap cannot bring home, and a NaN
coordinate is exactly that -- deterministic, needs no tuning, and exercises the
path a real violation takes.
"""


import os
import re
import resource
import shutil
import subprocess
import sys
import textwrap

import pytest

from conftest import PLUGIN_PATH, toolkit_required

# The one line of the guidance that has to survive rewording: the environment
# variable a reader must set. Pinning a fragment rather than the whole block
# keeps this test failing on a broken mechanism rather than on an edited
# sentence.
_DUMP_HINT = "CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1"



def _child(body, timeout=300, env=None):
    """Run `body` in a fresh interpreter and return the CompletedProcess.

    Prints are the channel back: the child is expected to die, so anything it
    wants to report has to be flushed before the abort that follows a trap.
    """
    script = textwrap.dedent(f"""
        import sys
        sys.path[:0] = {PLUGIN_PATH!r}
        import numpy as np
        import openmm as mm, openmm.unit as unit, openmm.app as app
        import nv_openmm_plugin as nv

        NM = unit.nanometer

        def build(nan_atom=None, box_nm=3.0):
            # 3 nm at a 0.9 nm cutoff is the smallest box the cell list accepts
            # (it wants >= 3 cells per axis); small keeps the child quick.
            ff = app.ForceField("tip3p.xml")
            mod = app.Modeller(app.Topology(), [])
            mod.addSolvent(ff, model="tip3p", boxSize=mm.Vec3(box_nm, box_nm, box_nm) * NM)
            system = ff.createSystem(mod.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=0.9 * NM,
                                     constraints=app.HBonds, rigidWater=True)
            found = [i for i in range(system.getNumForces())
                     if isinstance(system.getForce(i), mm.NonbondedForce)]
            assert len(found) == 1
            system.addForce(nv.NonbondedForce(system.getForce(found[0]), 1.0))
            system.removeForce(found[0])
            pos = np.array(mod.positions.value_in_unit(NM))
            if nan_atom is not None:
                pos[nan_atom] = [float("nan")] * 3
            integ = mm.VerletIntegrator(0.001 * unit.picoseconds)
            ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                             {{"Precision": "mixed"}})
            ctx.setPositions(pos * NM)
            return ctx, integ

{textwrap.indent(textwrap.dedent(body), " " * 8)}
        sys.stdout.flush()
        sys.stderr.flush()
    """)
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    # The child is expected to abort, and an abort writes a host core file of a
    # few GB into the working directory. Nothing here reads it -- the GPU dump
    # the recipe asks for is a separate file -- so the child may write none.
    return subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=timeout,
                          env=child_env, preexec_fn=_no_core_file)


def _no_core_file():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))




@pytest.fixture(scope="module")
def caught():
    """A fault provoked and caught, the way a real script would catch it.

    The child asks no questions -- it prints the exception and nothing else --
    so whatever text shows up got there from cupo_check_cuda on its own.
    """
    proc = _child("""
        ctx, integ = build(nan_atom=0)
        try:
            ctx.getState(getEnergy=True)
            print("NO_TRAP")
        except Exception as exc:
            print("EXC_BEGIN", flush=True)
            print(str(exc), flush=True)
            print("EXC_END", flush=True)
    """)
    assert "NO_TRAP" not in proc.stdout, (
        "a NaN coordinate no longer trips the finiteness check, so this test is "
        f"no longer testing anything:\n{proc.stdout}\n{proc.stderr}")
    body = proc.stdout.split("EXC_BEGIN", 1)
    assert len(body) == 2, f"child printed no exception:\n{proc.stdout}\n{proc.stderr}"
    return body[1].split("EXC_END", 1)[0]


def test_device_fault_says_how_to_find_the_site(caught):
    """The recipe reaches the caller, on the exception it actually catches."""
    assert _DUMP_HINT in caught, (
        "a device fault no longer tells the caller how to locate it:\n" + caught)
    for needed in ("CUDA_COREDUMP_FILE", "cuda-gdb", "compute-sanitizer"):
        assert needed in caught, f"guidance does not mention {needed}:\n{caught}"


def test_device_fault_warns_the_named_site_may_be_wrong(caught):
    """The most misleading part of the bare error, called out explicitly.

    The kernel named in the message is wherever the poisoned context was first
    observed. On this very case that is a k-space kernel, while the trap was in
    the cell list -- so a reader who trusts the name looks in the wrong file.
    """
    assert "_detected_" in caught, (
        "the message does not warn that the named site is where the error was "
        f"detected rather than where it happened:\n{caught}")


def test_guidance_is_additive_and_appears_once(caught):
    """The recipe is appended to the old message, not substituted for it.

    The first line still has to be the concise "<site>: <cuda message>" that
    every other caller greps for, and the block has to appear exactly once --
    two copies would mean it is being attached at more than one level, which is
    how the pre-revert version ended up duplicating its own report.

    The complementary case -- an ordinary cuda error staying short -- is not
    reachable from here, and that is a property of the code rather than an
    omission: cupo's host-side failures (a box below the grid minimum, a
    cudaMalloc that cannot be satisfied, an natom past the sort-index limit)
    print their own FATAL line and abort, so they never travel through
    cupo_check_cuda. The guard for that case is cupo_is_device_fault's switch,
    which only admits the codes that mean the context died.
    """
    first = caught.strip().splitlines()[0]
    assert "unspecified launch failure" in first, (
        f"the concise first line was replaced rather than added to:\n{caught}")
    assert _DUMP_HINT not in first, (
        f"the guidance is on the first line, where it crowds out the error:\n{first}")
    assert caught.count(_DUMP_HINT) == 1, (
        f"the guidance appears {caught.count(_DUMP_HINT)} times, so it is being "
        f"attached at more than one level:\n{caught}")


def test_the_whole_message_fits_the_shared_error_buffer(caught):
    """The shared runtime C API keeps an error in a 4096-byte buffer
    (cupo_last_error) and cuts what does not fit, from the end -- which is
    where the advice is. The message caught here, with its long first line, has
    to fit with the terminating NUL."""
    assert len(caught.strip().encode()) < 4096, (
        f"the message is {len(caught.strip().encode())} bytes; "
        "cupo_last_error would cut its end")


def test_following_the_recipe_finds_the_trapping_line(tmp_path):
    """Execute the advice, end to end.

    This is the test that keeps the message honest: it sets the environment the
    message tells the reader to set, provokes the same fault, then reads the dump
    back with the command the message prints. If a future CUDA release changes
    the variables, the dump format or cuda-gdb's output, this fails and the text
    gets corrected instead of quietly becoming wrong.

    In the dev container, testing the tree's build, a missing cuda-gdb fails:
    cuda-gdb ships with the CUDA toolkit these kernels are built with, which that
    image has, so its absence means the image is broken, or the advice names a
    tool the toolkit no longer has. Elsewhere, as for an installed plugin in an
    environment without the toolkit, the test skips, saying so
    (conftest.toolkit_required).
    """
    cuda_gdb = shutil.which("cuda-gdb")
    if cuda_gdb is None and not toolkit_required():
        pytest.skip("cuda-gdb is not installed: the CUDA toolkit is needed to follow "
                    "the device-fault advice")
    assert cuda_gdb is not None, (
        "cuda-gdb is not on PATH in the dev container, whose image has the CUDA "
        "toolkit: the image is broken, or cuda-gdb has moved or been dropped from "
        "the toolkit, and then the printed advice needs to name whatever replaced it.")
    dump = tmp_path / "cupo_%h_%p.nvcudmp"
    proc = _child("""
        ctx, integ = build(nan_atom=11)
        try:
            ctx.getState(getEnergy=True)
        except Exception:
            pass
    """, timeout=900, env={"CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
                           "CUDA_COREDUMP_FILE": str(dump)})
    dumps = sorted(tmp_path.glob("*.nvcudmp"))
    assert dumps, (
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION produced no dump, so the first half "
        f"of the advice is wrong:\n{proc.stdout}\n{proc.stderr[-2000:]}")

    out = subprocess.run(
        [cuda_gdb, "-batch", "-ex", f"target cudacore {dumps[0]}",
         "-ex", "info cuda kernels", "-ex", "info cuda threads"],
        capture_output=True, text=True, timeout=900).stdout

    assert "cuda_matrix_migrate" in out, (
        "the dump does not name the kernel the trap is in, so the second half of "
        f"the advice is wrong:\n{out[:2000]}")
    # CI's builds have no line information (./run_build turns CUPO_LINEINFO off
    # there), and then cuda-gdb names no file. A local build, with CUPO_LINEINFO
    # on by default, names the file, and then it has to show the line of the
    # failing check as well.
    if "CellList.cu" in out:
        assert "__trap()" in out, (
            f"the dump does not show the source line of the failing check:\n{out[:2000]}")
    # The thread index is what the message offers in place of local variables,
    # which -O3 does not keep. Atom 11 is the one the NaN was planted on.
    assert re.search(r"\(11,0,0\)", out), (
        "the trapping thread is not reported, so the message's claim that the "
        f"thread index identifies the element is wrong:\n{out[:2000]}")


def test_a_fault_the_plugin_notices_first_gets_the_advice_once():
    """The plugin's own CUDA checks print the advice too, once.

    Under CUDA graph replay the trap is noticed first by the plugin's own
    synchronising calls (NV_CUDA_CHECK in CudaKernels.cu), not by cupo's, so
    cupo_check_cuda never sees it. Those checks print one line per failed call,
    and after a device fault several fail. The advice has to come once, after
    the first, and with the plugin's line naming its two variables.
    """
    proc = _child("""
        ctx, integ = build()
        # Past the graph-capture threshold (CudaKernels.cu: graph_warmup_ >= 50),
        # so that the faulting step replays a captured graph.
        for _ in range(60):
            ctx.getState(getEnergy=True)
        pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(NM)
        pos[11] = [float("nan")] * 3
        ctx.setPositions(pos * NM)
        try:
            ctx.getState(getEnergy=True)
            print("NO_TRAP")
        except Exception:
            pass
    """)
    assert "NO_TRAP" not in proc.stdout, (
        f"a NaN coordinate no longer trips the finiteness check:\n{proc.stderr}")
    err = proc.stderr
    assert "[nv-plugin cupo] CUDA error" in err, (
        "the plugin's own CUDA checks did not notice the fault, so this test no "
        f"longer tests them:\n{err[-3000:]}")
    assert err.count(_DUMP_HINT) == 1, (
        f"the advice appears {err.count(_DUMP_HINT)} times; it should come once, "
        f"after the first failed check:\n{err[-3000:]}")
    assert err.index("[nv-plugin cupo] CUDA error") < err.index(_DUMP_HINT)
    for variable in ("CUPO_OPENMM_PLUGIN_RNG_SEED=42",
                     "CUPO_OPENMM_STOCK_FORCE_DETERMINISTIC_FORCES=1"):
        assert variable in err, f"the plugin's line does not name {variable}:\n{err[-3000:]}"
