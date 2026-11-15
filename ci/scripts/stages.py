#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every stage there is: build and test cupo and its plugins, and the tarballs they
ship in.

./run_test is this file under the name people type -- a symlink, so the two names are
one program. CI runs it under the stages.py name, because a job whose whole work is
to build release tarballs reads badly as a job running run_test.
"""

# What --help prints under the options. The parsing, the choosing and the running are
# all in ci/lib/stages.py; what is this script's own is the examples, because they name
# this script's own stages.
usage = """\
  ./run_test                        the stages that run by default
  ./run_test cupo_unit              that stage and what it depends on
  ./run_test cupo_unit -k test_p1   that stage, with a pytest filter
  ./run_test -k test_p1             the default stages, with a pytest filter
  ./run_test customer_env_cupo      a stage that does not run by default
  ./run_test --list-stages          the stages there are
"""

import glob
import os
import pathlib
import shutil
import sys

repo_dir = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_dir / "ci"))

# Some modules of ci/lib are imported by the stages that use them, and not here: a
# tree that carries this runner may carry only the modules imported here.
from lib import console, proc, timing  # noqa: E402
from lib.stages import (artifacts_dir, commit_sha_short,  # noqa: E402
                        compiling_stage, filtered, later_shard, log_path, main,
                        no_artifacts, packaging_stage, run_pytest, stage,
                        tarball_stage, unpack)


def _cupo_env():
    """PYTHONPATH for the built extension and the reference implementation."""
    return proc.path_prepend(proc.env_with(), "PYTHONPATH",
                             repo_dir / "build", repo_dir / "src" / "reference")


def test_installed():
    """Whether the plugin's tests test the plugin pip or conda installed rather than
    the tree's build: CUPO_TEST_INSTALLED set to anything but "" or "0"."""
    return os.environ.get("CUPO_TEST_INSTALLED", "") not in ("", "0")


def _openmm_env():
    """Environment for anything that loads the OpenMM plugin.

    libcupo.so is looked for in the plugin's own build/ first -- that is where a
    self-contained tarball puts it -- and in cupo's build/ otherwise, which is the
    dev workflow. Exported into our own environment, not just handed to one command,
    because the stages after this one want it too. With CUPO_TEST_INSTALLED=1 the
    environment is left as it is: the installed plugin loads the cupo and the
    OpenMM it was installed with, wherever pip or conda put them.
    """
    if test_installed():
        return
    build_dir = repo_dir / "plugins" / "openmm" / "build"
    cupo_lib = build_dir if (build_dir / "libcupo.so").is_file() else repo_dir / "build"
    proc.path_prepend(os.environ, "LD_LIBRARY_PATH", cupo_lib, "/opt/openmm/lib")
    os.environ["OPENMM_PLUGIN_DIR"] = "/opt/openmm/lib/plugins"


def _testing_installed():
    """openmm_build's skip_on: with CUPO_TEST_INSTALLED=1 the tests test the
    installed plugin, and the tree's is not built."""
    return "testing the installed plugin (CUPO_TEST_INSTALLED=1)" if test_installed() else ""


def dev_image():
    """The dev image the host-side stages hand to docker: CI_IMAGE, which the CI
    exports, or this tree's own, pulled or built."""
    image = os.environ.get("CI_IMAGE")
    if image:
        return image
    from lib import docker_images
    return docker_images.ensure(docker_images.DEV)


CUSTOMER_ENV = "ci/customer-env/test_customer_env"


def _customer_env(route):
    """Build an image from a CUDA base, install the artifact the customer's way, and
    run it."""
    proc.run([str(repo_dir / CUSTOMER_ENV),
              route, str(artifacts_dir())])


# ── Stages: receiving another job's tarballs ──────────────────────────────────
#
# A CI job builds nothing it can get from an upstream job. It downloads that job's
# tarballs into its own working directory and unpacks them over the clone, and the
# stages here are what does that -- one per producer, so the stage that reads a
# tarball is what names it, and `git grep cupo-nv-openmm-plugin` lands on the stage
# rather than on a table in another file.
#
# where="host" every time. ARTIFACTS_DIR is the job's working directory, which is
# outside the clone, and run_docker mounts only the clone -- so the tarballs do not
# exist inside the dev container, and this cannot be folded into the container stages
# that consume it.
#
# Nothing depends on these. Unpacking and building are two ways to get the same files,
# never both in one run, and depends= has no way to spell "either" -- a build stage
# listed alongside one of these would read as needing both. The jobs that receive
# tarballs name the unpack stage they want in STAGES, which follows no dependencies,
# and the build stages take themselves out of the way when CI_TEST_STAGE says the
# binaries already arrived (./run_build, plugins/openmm/run_build).
#
# skip_on=no_artifacts: unpacking is what a job does with what it was handed, and a
# working tree that built its own has nothing to unpack. See ci/lib/stages.py.


# The repository's settings file, at its root; a tree without it has none of the
# settings, and the stages that need them say so.
REPO_SETTINGS = repo_dir / "cupo-repo.ini"


def repo_setting(key):
    """[ci] `key` in the repository's settings file (REPO_SETTINGS); "" without it."""
    import configparser
    config = configparser.ConfigParser()
    config.read(REPO_SETTINGS)
    return config.get("ci", key, fallback="")


def test_tarballs(key):
    """The names, as globs, of the CI's own test tarballs of one kind: [ci] `key` in
    the repository's settings, one per line; [] without them."""
    return repo_setting(key).split()


@stage(where="host", skip_on=no_artifacts)
def unpack_cupo():
    """The cupo runtime and its pybind module, which everything downstream links or
    imports: libcupo.so, the LTO-IR objects and the headers into build/ and include/,
    cupo.cpython-*.so into build/. And the CI's test tarballs of cupo, when the
    repository's settings name them (test_tarballs)."""
    unpack("cupo-[0-9]*.tar.gz", ".")
    unpack("cupo-pybind-*.tar.gz", ".")
    for pattern in test_tarballs("cupo_test_tarballs"):
        unpack(pattern, ".")

    # .env_hash alongside them, so that `clean --soft` does not decide the environment
    # changed and wipe the build directories these tarballs just filled.
    env_hash = artifacts_dir() / ".env_hash"
    if env_hash.is_file():
        shutil.copy2(str(env_hash), str(repo_dir / ".env_hash"))


@stage(where="host", skip_on=no_artifacts)
def unpack_openmm():
    """The plugin binaries and tests the openmm stages run: build/cpp/*.so, the SWIG
    module, and tests/, from the CI's own tarball of the plugin's build and tests,
    which the repository's settings name (test_tarballs). The source, wheel and conda tarballs
    are the shipped article, which the customer_env_* stages install in containers
    of their own.
    """
    for pattern in test_tarballs("plugin_test_tarballs"):
        unpack(pattern, "plugins/openmm")


# ── Stages: build and test ────────────────────────────────────────────────────
#
# A CI test job may be split into shards. cupo_unit and openmm_unit are what is
# split: each shard runs its share of their tests (shard=True). The other stages of
# those jobs run whole, in shard 1 only (skip_on=later_shard): the benchmarks,
# which are never split, the LTO-IR programs and the samples. The unpack_* stages
# above run in every shard, since every shard's tree needs the tarballs. Outside a
# split job none of this applies.

# cupo's sources. The plugin's tree on GitHub has cupo's headers and its built
# libraries and none of these, so there cupo_build skips and openmm_build builds
# against the libraries (plugins/openmm/run_build).
CUPO_SOURCES = ("run_build", "src")


@compiling_stage(needs_files=CUPO_SOURCES)
def cupo_build():
    proc.run([str(repo_dir / "run_build")])


@stage(autorun=True, depends=cupo_build, needs_files=CUPO_SOURCES + ("ci/tests",))
def cupo_unit():
    # The CI's own tests too: they are pure logic -- no GPU, no scheduler -- so they
    # need no stage of their own, and they sit outside tests/, where that suite's
    # GPU-requiring conftest would apply to them.
    return run_pytest([repo_dir / "tests", repo_dir / "ci" / "tests"],
                      "not benchmark", _cupo_env(), shard=True)


# The benchmarks lock the GPU clocks themselves (tests/gpu_clocks.py), and give the
# GPU back when they are done.
@stage(autorun=True, depends=cupo_build, skip_on=later_shard, needs_files=CUPO_SOURCES)
def cupo_bench():
    return run_pytest([repo_dir / "tests"], "benchmark", _cupo_env())


# The LTO-IR checks that are programs: the runtime grid planner, the shipped
# objects' SASS, the device harness, the LTO-IR sample and a CMake consumer of the
# convolution API. The LTO-IR pytest modules run in cupo_unit, with the rest of
# tests/. No pytest here, so a pytest filter skips the stage, as it does the
# samples'.
@stage(autorun=True, depends=cupo_build, skip_on=[filtered, later_shard],
       needs_files=("tests/ltoir/run",))
def cupo_ltoir():
    return proc.tee([str(repo_dir / "tests" / "ltoir" / "run")], log_path())


@compiling_stage(depends=cupo_build, needs_files=("plugins/openmm/run_build",),
                 skip_on=_testing_installed)
def openmm_build():
    proc.run([str(repo_dir / "plugins" / "openmm" / "run_build")])


@stage(autorun=True, depends=openmm_build, needs_files=("plugins/openmm/tests",))
def openmm_unit():
    _openmm_env()
    return run_pytest([repo_dir / "plugins" / "openmm" / "tests"], "not benchmark",
                      shard=True)



def _openmm_samples():
    """Where the samples live. One place, so the sample stages cannot drift."""
    return repo_dir / "plugins" / "openmm" / "samples"


@stage(autorun=True, depends=openmm_build, skip_on=[filtered, later_shard])
def openmm_sample():
    _openmm_env()
    proc.run([str(_openmm_samples() / "solvated_systems" / "run")])


@stage(autorun=True, depends=openmm_build, skip_on=[filtered, later_shard])
def openmm_sample_alchemical():
    """Combine TYK2 RBFE native and custom nonbonded forces in the plugin.

    Check the alchemical parameter updates against stock OpenMM."""
    _openmm_env()
    proc.run([str(_openmm_samples() / "alchemical" / "run")])


# The plugin benchmarks, one stage per benchmark group (golden_runs.GROUPS and
# _SYSTEM_GROUPS in plugins/openmm/tests/test_bench_plugin.py), so that CI can run the
# groups as jobs of their own, in parallel. Each group writes its goldens to its own
# subdirectory of GOLDEN_OUTPUT_DIR. Run together, as ./run_test does by default, the
# two cover every system once.
def _openmm_bench_group(group):
    _openmm_env()
    return run_pytest([repo_dir / "plugins" / "openmm" / "tests"], "benchmark",
                      proc.env_with(CUPO_BENCH_GROUP=group))


# The benchmarks need the core tests' clock helpers (tests/gpu_clocks.py), which the
# plugin's source tree carries.
BENCH_FILES = ("plugins/openmm/tests", "tests/gpu_clocks.py")


@stage(autorun=True, depends=openmm_build, skip_on=later_shard, needs_files=BENCH_FILES)
def openmm_bench():
    return _openmm_bench_group("main")


@stage(autorun=True, depends=openmm_build, skip_on=later_shard, needs_files=BENCH_FILES)
def openmm_bench_triclinic():
    """The cell-shape set: bace in a cube, a rhombic dodecahedron and a truncated
    octahedron of one width."""
    return _openmm_bench_group("triclinic")


# ── Stages: the release tarballs ──────────────────────────────────────────────
#
# What the CI's build jobs run, on a machine with no GPU, since only nvcc is
# needed to compile: the cupo tarball (libraries, LTO-IR objects, headers), the
# pybind tarball, and the CI's own test tarballs, which the repository's settings
# name (test_tarballs).
#
# The plugin's public files -- the source tarball, and the wheel and the conda
# package built from it -- are customer_env_package's, below.
#
# None of them is autorun, which is what keeps a bare ./run_test from spending twenty
# minutes packaging what it was asked to test. CI names them in STAGES, one job
# each, and the openmm job receives the cupo tarball as an artifact rather
# than building it -- which is why STAGES does not follow the depends edge below. A
# person naming a stage on the command line does, and means it.

def is_this_build(path):
    """Whether a tarball in the tree came out of the commit being built.

    Every tarball name ends in the short commit hash -- cupo-<ver>+<cuda>.<arch>.g<sha>
    -- and that is the whole of what tells one build's output from another's, since
    the version and the architecture list are the same across commits more often than
    not.
    """
    return path.endswith(".g%s.tar.gz" % commit_sha_short())


@tarball_stage(needs_files=CUPO_SOURCES + ("make_release_tarball",))
def cupo_tarball():
    proc.run([str(repo_dir / "make_release_tarball")])


@tarball_stage(depends=cupo_tarball,
               needs_files=("plugins/openmm/make_release_tarball",))
def openmm_tarball():
    """The CI's tarball of the plugin's build and tests, from a build of the plugin
    in this tree: what the plugin's test jobs and the benchmarks run on. Its kind,
    the argument make_release_tarball takes, is [ci] plugin_test_tarball_kind in
    the repository's settings."""
    kind = repo_setting("plugin_test_tarball_kind")
    if not kind:
        raise proc.Fatal("no [ci] plugin_test_tarball_kind in %s" % REPO_SETTINGS.name)
    proc.run([str(repo_dir / "plugins" / "openmm" / "make_release_tarball"), kind])


# after, not depends: this stage copies whatever tarballs the run produced and asks
# for none of them, so `./run_test customer_env_cupo` builds the cupo tarball and not
# the plugin's. What the edges say is what decides the order (ci/lib/stages.py), and
# what this one says is "not before a tarball stage that is running anyway".
@stage(after=[cupo_tarball, openmm_tarball],
       needs_files=("ci/scripts/list_artifacts",))
def collect_artifacts():
    # repo_dir/artifacts, and deliberately not artifacts_dir(): this stage
    # empties the directory before filling it, and in CI artifacts_dir() is where the
    # upstream jobs' tarballs arrived.
    artifacts = repo_dir / "artifacts"
    proc.run(["rm", "-rf", str(artifacts)])
    artifacts.mkdir(parents=True)

    # Each tarball stage may or may not have run, so each of these patterns may match
    # nothing. That is not an error here: which tarballs exist is decided by which
    # stages ran, and this stage collects whatever those stages left behind.
    sources, stale = [], []
    for pattern in ("cupo-*.tar.gz",
                    "plugins/openmm/cupo-nv-openmm-plugin-*.tar.gz"):
        for path in sorted(glob.glob(str(repo_dir / pattern))):
            (sources if is_this_build(path) else stale).append(path)

    # .env_hash travels with the tarballs so a downstream build does not decide the
    # environment changed and wipe the build directories the artifacts provided. It
    # carries no version, so it is taken whatever the tarballs are.
    sources += sorted(glob.glob(str(repo_dir / ".env_hash")))

    # A tarball from an earlier commit is still sitting in the tree: none of the
    # make_release_tarball scripts removes the one it replaces, and in CI nothing ever
    # noticed, because each build job starts from a fresh clone. Copying it here would
    # put two runtime tarballs in one artifacts directory, and the packaging stages
    # refuse that rather than guess which one they were meant to install -- so it is
    # named and left where it is.
    for path in stale:
        console.color(console.SKIP, "--- not collecting %s: not from this build ---"
                      % os.path.basename(path))

    if sources:
        proc.run(["cp"] + sources + [str(artifacts)])

    print()
    proc.run(["ls", "-lh", str(artifacts)])
    print()

    tarballs = sorted(glob.glob(str(artifacts / "*.tar.gz")))
    if tarballs:
        proc.run([str(repo_dir / "ci" / "scripts" / "list_artifacts")] + tarballs)


# ── Stages: the tarballs we ship ──────────────────────────────────────────────
#
# What comes out of a release tarball, rather than the tree it was built from. Every
# one of these takes a tarball out of the artifacts directory, so every one depends on
# the stage that produces the tarball it takes, and on collect_artifacts, which is
# what puts a built tarball in that directory.
#
# The CI does not follow those edges and is not meant to -- a test job receives the
# artifacts from the build jobs rather than building them -- but a person running
# `./run_test customer_env_cupo` on a clean tree means "and whatever it needs", and
# what it needs is a cupo tarball.

# The plugin's public files, which the customer_env_* stages then install the
# customers' ways: the source tarball, and the wheel and the conda package (with the
# conda channel) built from it, as make_release_tarball makes them. Two stages,
# since the build runs in the dev container and the files go to the artifacts
# directory, which is outside it. The routes below take them from there, and the
# CI keeps them as the job's artifacts.
def _public_tarballs():
    from lib import release
    return (release.SRC_GLOB, release.WHEEL_GLOB, release.CONDA_GLOB)


PLUGIN_RELEASE_TARBALL = "plugins/openmm/make_release_tarball"


def release_files_from():
    """The pipeline whose release files this job was handed in place of building
    its own, or "": RELEASE_FILES_FROM_PIPELINE, which ci/jobs/test sets on the GPU
    node when the pipeline has BUILD_PIPELINE_ID, after putting those files in the
    artifacts directory."""
    return os.environ.get("RELEASE_FILES_FROM_PIPELINE", "")


def _release_files_reused():
    """customer_env_package's skip_on."""
    pipeline = release_files_from()
    return ("the release files are pipeline %s's (BUILD_PIPELINE_ID)" % pipeline
            if pipeline else "")


@tarball_stage(depends=cupo_tarball, needs_files=(PLUGIN_RELEASE_TARBALL,),
               skip_on=_release_files_reused)
def customer_env_package():
    """make_release_tarball public: the source tarball, then the plugin built from
    it, then the wheel and the conda package. Without CI_TEST_STAGE, which a CI test
    job sets so that run_build skips the build its tarballs brought: this build is
    the one whose files are released."""
    env = {k: v for k, v in os.environ.items() if k != "CI_TEST_STAGE"}
    proc.run([str(repo_dir / "plugins" / "openmm" / "make_release_tarball"), "public"],
             env=env)


@stage(where="host", depends=customer_env_package, needs_files=(PLUGIN_RELEASE_TARBALL,))
def customer_env_release_files():
    """Put the files customer_env_package made, this build's only, in the artifacts
    directory. Older ones of the same names are taken out first: the routes refuse
    two of one kind.

    With another pipeline's release files (release_files_from) nothing was made,
    and they are in the artifacts directory already: checked to be there, one of
    each kind, and left as they are."""
    out = artifacts_dir()
    out.mkdir(parents=True, exist_ok=True)
    pipeline = release_files_from()
    if pipeline:
        reused = []
        for pattern in _public_tarballs():
            found = sorted(out.glob(pattern))
            if len(found) != 1:
                raise proc.Fatal("expected one %s of pipeline %s in %s, found %d"
                                 % (pattern, pipeline, out, len(found)))
            reused += found
        print("=== the release files of pipeline %s, not rebuilt ===" % pipeline)
        proc.run(["ls", "-lh"] + [str(p) for p in reused])
        return
    made = []
    for pattern in _public_tarballs():
        for old in out.glob(pattern):
            old.unlink()
        ours = [p for p in sorted((repo_dir / "plugins" / "openmm").glob(pattern))
                if is_this_build(str(p))]
        if len(ours) != 1:
            raise proc.Fatal("expected one %s of this build in plugins/openmm, found %d"
                             % (pattern, len(ours)))
        made += ours
    proc.run(["cp"] + [str(p) for p in made] + [str(out)])
    proc.run(["ls", "-lh"] + [str(out / p.name) for p in made])


# The in-tree samples as a user of the source tarball or the GitHub repository runs
# them: plugins/openmm/samples/run_docker, in the dev image, with the plugin this
# tree built, or unpacked from the CI's tarball of the plugin's build and tests
# (unpack_openmm). Both samples.
_PLUGIN_FILES = ("build/cpp/libOpenMMNvidia.so",
                 "build/cpp/platforms/cuda/libOpenMMNvidiaCUDA.so",
                 "cpp/swig/_nvplugin.so", "cpp/swig/nvplugin.py",
                 "python/nv_openmm_plugin/__init__.py")
_SAMPLES = ("solvated_systems", "alchemical")
# Run by run_docker in samples/: what the samples' common.sh imports, given the
# plugin directory, and where from.
_WHICH_PLUGIN = ('source ./common.sh && ensure_plugin "$(cd .. && pwd)" && '
                 'python3 -c "import nv_openmm_plugin; print(nv_openmm_plugin.__file__)"')


def _docker_on_host():
    if not shutil.which("docker"):
        raise proc.Fatal("no docker on this host, and openmm_samples_docker runs the "
                         "samples with plugins/openmm/samples/run_docker")


# The dev image comes from dev_image(), whose module this tree may not carry;
# without it the samples run with samples/run_docker itself.
@stage(where="host", depends=openmm_build, pre=_docker_on_host,
       skip_on=[filtered, later_shard], needs_files=("ci/lib/docker_images.py",))
def openmm_samples_docker():
    """Run both samples with plugins/openmm/samples/run_docker and the plugin in this
    tree: build/ and cpp/swig/ from the build, or in CI from the CI's tarball of the
    plugin's build and tests.

    Fails, saying what is missing, when the tree has no built plugin, and when the
    plugin the samples import is not the tree's: the dev image has none installed,
    and that is checked first, so a sample cannot pass on another one."""
    plugin = repo_dir / "plugins" / "openmm"
    missing = [rel for rel in _PLUGIN_FILES if not (plugin / rel).is_file()]
    if missing:
        raise proc.Fatal("no built plugin in %s: %s missing. Build it (openmm_build), "
                         "or unpack the CI's tarball of the plugin (unpack_openmm)"
                         % (plugin, ", ".join(missing)))
    # run_docker in samples/ mounts plugins/openmm and nothing above it, so cupo's
    # libcupo.so in the root's build/ is out of the container's sight. The CUDA
    # platform library looks for it in plugins/openmm/build/ too (its RUNPATH), as a
    # source build has it, and a hard link puts it there without a copy.
    libcupo = plugin / "build" / "libcupo.so"
    if not libcupo.is_file():
        cupo = repo_dir / "build" / "libcupo.so"
        if not cupo.is_file():
            raise proc.Fatal("no libcupo.so in %s or %s: build cupo, or unpack the cupo "
                             "tarball (unpack_cupo)" % (libcupo.parent, cupo.parent))
        # The library with its links, libcupo.so -> libcupo.so.<ABI> ->
        # libcupo.so.<VERSION>: the loader asks for the second, the library's
        # SONAME. Not the .debug file beside them.
        for src in sorted(cupo.parent.glob("libcupo.so*")):
            if src.name.endswith(".debug"):
                continue
            dst = libcupo.parent / src.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            if src.is_symlink():
                os.symlink(os.readlink(src), dst)
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    run_docker = str(_openmm_samples() / "run_docker")
    env = proc.env_with(PREBUILT_IMAGE=dev_image())
    if proc.rc([run_docker, "python3", "-c", "import nv_openmm_plugin"], env=env) == 0:
        raise proc.Fatal("the dev image has an nv_openmm_plugin of its own, so the "
                         "samples could pass without this tree's")
    imported = proc.capture([run_docker, "bash", "-c", _WHICH_PLUGIN], env=env)
    expected = str(plugin / "python" / "nv_openmm_plugin" / "__init__.py")
    if imported.splitlines()[-1].strip() != expected:
        raise proc.Fatal("the samples import %s, not this tree's %s"
                         % (imported.splitlines()[-1].strip(), expected))
    print("=== the samples import %s ===" % expected)
    for sample in _SAMPLES:
        proc.run([run_docker, "./%s/run" % sample], env=env)


@packaging_stage(depends=[cupo_tarball, collect_artifacts], needs_files=(CUSTOMER_ENV,))
def customer_env_cupo():
    _customer_env("cupo")


# The LTO-IR objects and convolution API of the cupo tarball, used the way the
# cupo tarball's LTO-IR guide says, with the CUDA toolkit and Python 3, since
# its payload and image are not customer_env_cupo's.
@packaging_stage(depends=[cupo_tarball, collect_artifacts], needs_files=(CUSTOMER_ENV,))
def customer_env_cupo_ltoir():
    _customer_env("cupo-ltoir")


@packaging_stage(depends=customer_env_release_files, needs_files=(CUSTOMER_ENV,))
def customer_env_plugin_conda():
    _customer_env("plugin-conda")


@packaging_stage(depends=customer_env_release_files, needs_files=(CUSTOMER_ENV,))
def customer_env_plugin_wheel_conda():
    _customer_env("plugin-wheel-conda")


@packaging_stage(depends=customer_env_release_files, needs_files=(CUSTOMER_ENV,))
def customer_env_plugin_wheel_pip():
    _customer_env("plugin-wheel-pip")


# The conda-forge recipes (packaging/conda-forge/), for the cupo tarball and the
# plugin's source tarball: linted, built as conda-forge builds them, installed and
# tested (ci/customer-env/run-conda-forge).
@packaging_stage(depends=[cupo_tarball, customer_env_release_files],
                 needs_files=(CUSTOMER_ENV, "packaging/conda-forge/render_recipes.py"))
def customer_env_conda_forge():
    _customer_env("conda-forge")


@packaging_stage(depends=customer_env_release_files, needs_files=(CUSTOMER_ENV,))
def customer_env_readme():
    """The documented install commands, as the GitHub test job runs them,
    run here on the files this pipeline made, with only each download
    replaced by the local file (route readme). In a merge request with the label
    Release, then the documentation's pip install, from PyPI, and its conda
    install, each in an empty environment, with this pipeline's wheel in place of
    the PyPI package."""
    _customer_env("readme")



# ── Stages: the released repository, from GitHub ─────────────────────────────
#
# What a user gets from GitHub after a release. github_clone and github_check run
# on the host, and the user paths below, github_*_via_*, on a GPU machine, which
# gets the clone through the artifacts directory. They download from GitHub
# themselves, as a user does, so the GPU machine gets the token too, for them.
#
# `./run_test github_pip_via_pypi`, or any of them, runs github_clone first, and
# needs only the token:
# GITHUB_TOKEN, or the variable GITHUB_TOKEN_VARIABLE names, and GITHUB_REPOSITORY
# for another repository than NVIDIA/cuPotential. `./run_test github_check` builds
# this tree's tarballs as well, to compare them with the release, so it passes only
# on the commit that was released.

GITHUB_CHECK_REL = "ci/jobs/github/check"
GITHUB_CHECK = repo_dir / GITHUB_CHECK_REL


# after collect_artifacts, which empties the artifacts directory this stage fills.
@stage(where="host", after=collect_artifacts, needs_files=(GITHUB_CHECK_REL,))
def github_clone():
    """Clone the GitHub repository at the release's tag, v<VERSION> for the
    published repository, and download its release's files, with the token, into the artifacts directory:
    github-clone.tar.gz and github-release/."""
    proc.run([str(GITHUB_CHECK), "fetch"])


@stage(where="host", depends=[github_clone, customer_env_release_files, collect_artifacts],
       needs_files=(GITHUB_CHECK_REL,))
def github_check():
    """Check what github_clone fetched against this tree's build: the clone's tree
    is the source tarball's less build/, the release's files are this build's, and
    the review passes on the clone."""
    proc.run([str(GITHUB_CHECK), "compare"])


def clone_fetch_env(environ=None):
    """How the clone's run_build downloads cupo's libraries, as env_with() keywords.
    It takes the repository from the clone's origin remote (fetch_cupo),
    and nothing here names one. For the published repository, no token and no
    tag: it downloads from the release v<VERSION>'s URLs, as a user's does. For
    another one (GITHUB_REPOSITORY), GITHUB_TOKEN is GITHUB_INSTALL_TOKEN, and the
    release is read through the API, as the README says to do for a repository
    that needs a token. Which release is fetch_cupo's own choice, from the clone's
    RELEASE_TAG, as a user's is: CUPO_REPOSITORY, CUPO_RELEASE_TAG and
    CUPO_RELEASE_URL are left out either way."""
    from lib import readme_install
    environ = os.environ if environ is None else environ
    repository = environ.get("GITHUB_REPOSITORY") or readme_install.PUBLISHED_REPOSITORY
    out = dict(GITHUB_TOKEN=None, GH_TOKEN=None, CUPO_REPOSITORY=None, CUPO_RELEASE_URL=None,
               CUPO_RELEASE_TAG=None)
    if not readme_install.is_published(repository):
        token = environ.get("GITHUB_INSTALL_TOKEN")
        if not token:
            raise proc.Fatal("%s needs a token: GITHUB_INSTALL_TOKEN is needed for the clone's "
                             "run_build to download cupo's libraries" % repository)
        out["GITHUB_TOKEN"] = token
    return out


# ── The user paths, one stage each ────────────────────────────────────────────
#
# Each stage below is one way a user gets the plugin from the release, named
# github_<what>_via_<where from>: what the user ends up with (a build, the LTO-IR
# sample, a pip or a conda install), and what they got it through (the clone, the
# source tarball, the release's files, PyPI). Each needs only github_clone, and
# each works in a directory of its own, which it empties first: its own unpacked
# clone or tarball, and, for the installs, test_customer_env's own staging
# directory with its own venv or conda environment. So what one leaves behind,
# after a failure, cannot make another pass or fail. The GitHub test jobs run them
# with --keep-going (ci/github.gitlab-ci.yml), so each path gets its own verdict.
#
# Every command block runs in a user's environment, with none of this run's
# variables (readme_install.user_env()): a block's ./run_test is a run of the
# clone's own, a customer's, and not one of this run's stages. The dev image stands
# in for the one run_docker would build from the clone's Dockerfile, which is the
# file the dev image is built from. The commands are the snapshot's, generated from
# the Fern pages, or an older release's README's (readme_install.command_document()).


def _fresh(name):
    """repo_dir/name, emptied: a stage's own directory. In the tree, as run_docker
    mounts the tree it is in."""
    path = repo_dir / name
    shutil.rmtree(path, ignore_errors=True)
    return path


def _unpack_clone(name):
    """The clone github_clone fetched, unpacked afresh at repo_dir/name."""
    import tempfile
    dest = _fresh(name)
    unpack = pathlib.Path(tempfile.mkdtemp(prefix=".github-unpack-", dir=repo_dir))
    proc.run(["tar", "-xzf", str(artifacts_dir() / "github-clone.tar.gz"), "-C", str(unpack)])
    (unpack / "github-clone").rename(dest)
    unpack.rmdir()
    return dest


def _documented(tree, read):
    """read(command document's text, LTO-IR README's text) for the tree, a stage's
    own clone: the blocks the stage runs, or proc.Fatal naming the document."""
    from lib import readme_install
    document = readme_install.command_document(tree)
    try:
        return read(document.read_text(), (tree / readme_install.LTOIR_README).read_text())
    except (OSError, readme_install.NoInstallCommand) as e:
        raise proc.Fatal("%s: %s" % (document, e))


@packaging_stage(depends=github_clone, needs_files=(GITHUB_CHECK_REL,))
def github_build_via_clone():
    """Build the plugin from source in the clone, run the samples and the tests,
    with the documented blocks as written, from the clone's root:
    `./run_docker plugins/openmm/run_build`, the block of `./run_docker` lines,
    which runs every sample, and the block of `./run_test` lines. The clone has
    no build/, so its run_build fetches cupo's libraries from the release of the
    repository its origin remote names: for the published repository from
    github.com with no token, as a user's does, and for another through the API
    with GITHUB_TOKEN, as documented (clone_fetch_env()). In .github-clone, which
    the job keeps, logs and all."""
    from lib import readme_install
    clone = _unpack_clone(".github-clone")
    blocks, tests = _documented(clone, lambda doc, _ltoir: (
        [readme_install.build_commands(doc), readme_install.sample_commands(doc)],
        readme_install.test_commands(doc)))
    env = readme_install.user_env(PREBUILT_IMAGE=dev_image(), **clone_fetch_env())
    for block in blocks:
        print("=== the documented commands, from the clone's root ===\n%s" % block)
        proc.run(["bash", "-exc", block], cwd=clone, env=env)
    print("=== the documented tests, from the clone's root ===\n%s" % tests)
    return proc.rc(["bash", "-exc", tests], cwd=str(clone), env=env)


@packaging_stage(depends=github_clone, needs_files=(GITHUB_CHECK_REL,))
def github_build_via_source_tarball():
    """The same in the release's source tarball, which github_clone downloaded:
    the documented block that unpacks it and goes into it, run where the tarball
    is, then the build, the samples and the LTO-IR sample's README commands
    (readme_install.source_tarball_script()), then the tests in the tree it
    unpacked. The tarball has cupo's libraries in build/. In .github-source."""
    from lib import readme_install, release
    clone = _unpack_clone(".github-source-docs")
    from_tarball, tests = _documented(clone, lambda doc, ltoir: (
        readme_install.source_tarball_script(doc, ltoir), readme_install.test_commands(doc)))
    shutil.rmtree(clone, ignore_errors=True)
    source = _fresh(".github-source")
    source.mkdir()
    shutil.copy(release.one(artifacts_dir() / "github-release", release.SRC_GLOB), source)
    image = dev_image()
    print("=== the documented commands, where the source tarball is ===\n%s" % from_tarball)
    proc.run(["bash", "-exc", from_tarball], cwd=source,
             env=readme_install.user_env(PREBUILT_IMAGE=image))
    unpacked = [p for p in source.iterdir() if p.is_dir()]
    if len(unpacked) != 1:
        raise proc.Fatal("the documented block unpacked %d directories in %s, not one"
                         % (len(unpacked), source))
    print("=== the documented tests, in the unpacked source tarball ===\n%s" % tests)
    return proc.rc(["bash", "-exc", tests], cwd=str(unpacked[0]),
                   env=readme_install.user_env(PREBUILT_IMAGE=image, **clone_fetch_env()))


@packaging_stage(depends=github_clone, needs_files=(GITHUB_CHECK_REL,))
def github_ltoir_via_clone_and_cupo_tarball():
    """The LTO-IR sample, which needs no plugin build, as its README,
    samples/ltoir-usage/README.md, gives it: its `./run_docker
    samples/ltoir-usage/run` lines from the root of a fresh clone, whose
    samples/ltoir-usage/run downloads build/ itself, in .github-clone-ltoir; then
    its block for the cupo runtime tarball, `./samples/ltoir-usage/run` lines, from
    the root of the release's cupo tarball, unpacked in .github-cupo, in the
    customer CUDA Toolkit and Python 3 image, without OpenMM. Both directories are
    removed at the end."""
    from lib import docker_images, readme_install, release
    fresh = _unpack_clone(".github-clone-ltoir")
    ltoir, ltoir_in_cupo = _documented(fresh, lambda _doc, ltoir: (
        readme_install.ltoir_sample_commands(ltoir), readme_install.ltoir_tarball_commands(ltoir)))
    env = readme_install.user_env(PREBUILT_IMAGE=dev_image(), **clone_fetch_env())
    print("=== %s's commands, from a fresh clone's root ===\n%s"
          % (readme_install.LTOIR_README, ltoir))
    proc.run(["bash", "-exc", ltoir], cwd=fresh, env=env)
    shutil.rmtree(fresh, ignore_errors=True)
    cupo = _fresh(".github-cupo")
    cupo.mkdir()
    proc.run(["tar", "-xzf", str(release.one(artifacts_dir() / "github-release",
                                             release.CUPO_GLOB)), "-C", str(cupo)])
    [top] = [p for p in cupo.iterdir() if p.is_dir()]
    print("=== %s's commands, in the unpacked cupo tarball, with CUDA and Python 3 ===\n%s"
          % (readme_install.LTOIR_README, ltoir_in_cupo))
    proc.run(["docker", "run", "--rm", "--gpus", "device=%s" % os.environ.get("NV_GPU", "all"),
              "--ipc=host", "-u", "%d:%d" % (os.getuid(), os.getgid()), "-e", "HOME=/tmp",
              "-v", "%s:%s" % (top, top), "-w", str(top),
              docker_images.ensure("customer-env-cupo-ltoir"),
              "bash", "-exc", ltoir_in_cupo], env=readme_install.user_env())
    shutil.rmtree(cupo, ignore_errors=True)


def _github_install(path):
    """One of test_customer_env's github-readme paths (GITHUB_PATHS there), in its
    own staging directory, with its own unpacked clone, which it removes at the
    end. For the published repository the blocks run as written. For another
    repository, whose release files cannot be downloaded from their URLs without a
    token, the files are fetched through the API first (GITHUB_INSTALL_TOKEN), and
    only the download step uses the local file."""
    proc.run([str(repo_dir / CUSTOMER_ENV), "github-readme", str(artifacts_dir()), path])


@packaging_stage(depends=github_clone, needs_files=(CUSTOMER_ENV,))
def github_pip_via_releases():
    """The documented pip install of the release's wheel, from its URL on the
    GitHub release, in the tree's Docker environment: `./run_docker bash`, `pip
    install <release URL>` and the sample, then, in the same session, the
    documented test of the installed plugin."""
    _github_install("pip-docker")


@packaging_stage(depends=github_clone, needs_files=(CUSTOMER_ENV,))
def github_conda_via_releases():
    """The documented conda install from the release's conda channel, in the tree's
    Docker environment: the channel's download, then, in one `./run_docker bash`
    session, `micromamba create ...`, the sample and the documented test of the
    installed plugin."""
    _github_install("conda-docker")


@packaging_stage(depends=github_clone, needs_files=(CUSTOMER_ENV,))
def github_pip_via_pypi():
    """The documented install from PyPI, `pip install nv-openmm-plugin`, into an
    empty venv (ci/customer-env/run-empty-env), then the sample and the test of
    the installed plugin. For the published repository against PyPI, as written,
    and pip has to install the release's version; for another, the release's wheel
    in place of the package."""
    _github_install("pip-empty")


@packaging_stage(depends=github_clone, needs_files=(CUSTOMER_ENV,))
def github_conda_empty_via_releases():
    """The documented conda install from the release's conda channel, in an image
    holding micromamba alone (ci/customer-env/run-empty-env): this release's
    version from that channel, then the sample and the test of the installed
    plugin."""
    _github_install("conda-empty")


if __name__ == "__main__":
    main(usage)
