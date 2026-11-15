# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenMM's own benchmark systems, apoa1 and dhfr, found where OpenMM put them.

This repository does not distribute these two systems. They are OpenMM's files
``examples/benchmarks/apoa1.pdb`` and ``examples/benchmarks/5dfr_solv-cube_equil.pdb``
(https://github.com/openmm/openmm), and conda-forge's ``openmm`` package installs
both under ``<prefix>/share/openmm/examples/benchmarks/``. ``openmm_benchmark_file``
returns the path of one of them:

1. The installed OpenMM's copy, when there is one. The directories tried are
   ``share/openmm/examples/benchmarks`` under the prefix of the ``openmm`` Python
   package, ``sys.prefix``, ``$CONDA_PREFIX`` and ``/opt/openmm``, in that order.
   ``$CUPO_OPENMM_BENCHMARKS_DIR``, when set, is the only directory tried.
2. In CI (``$CI`` set), nothing else: a CI machine is expected to have OpenMM
   installed with its examples, so a missing file is an error there, not a
   download.
3. Elsewhere, a copy downloaded earlier into the cache directory, or else a
   download from OpenMM's GitHub repository at the release tag
   ``OPENMM_RELEASE``, into that cache directory.

Every file, installed or downloaded, is used only when its SHA-256 is the one
recorded here, so every run reads the same atoms whatever OpenMM version supplied
them. The PyPI ``openmm`` wheels do not carry ``examples/``; with one of those,
the files are downloaded.

One file serves the plugin's ``solvated_systems`` sample and the tests. The sample
imports it from the installed package, ``nv_openmm_plugin._benchmark_systems``,
and keeps downloads in the user's cache directory (default_cache_dir()). The
tests, cupo's and the plugin's, reach it through data/datasets.py, which loads
this file by its path in the tree and keeps downloads in the tree's .tmp/.

So this module imports nothing of the plugin, nor anything outside the standard
library: data/datasets.py loads it without the package, where the plugin may be
neither built nor installed. Imported as part of the package, the package's
__init__ runs first and loads the plugin's libraries, as any
``import nv_openmm_plugin`` does.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# The OpenMM release the files are downloaded from: the series the plugin is built
# against. Tag 8.4.0 is commit 47684368dbbe4185d068be77d32a962059cfc37c; conda-forge's
# openmm 8.4.0 installs these same two files.
OPENMM_RELEASE = "8.4.0"
URL = ("https://raw.githubusercontent.com/openmm/openmm/" + OPENMM_RELEASE
       + "/examples/benchmarks/{file}")

# System name -> (file name in OpenMM's examples/benchmarks/, its SHA-256).
SYSTEMS = {
    "apoa1": ("apoa1.pdb",
              "900cdb7a2d9a90f4cbd1de43e9562566cb281a92e8696ce5a87f95a4bb2cb88e"),
    "dhfr": ("5dfr_solv-cube_equil.pdb",
             "fc9759761d9b33c58b09a3d0bf1dbdb76f23dbe0e89a8bd92f43ebcf0f2f2fdf"),
}

DIR_ENV = "CUPO_OPENMM_BENCHMARKS_DIR"
_SUBDIR = Path("share") / "openmm" / "examples" / "benchmarks"


def installed_dirs() -> list[Path]:
    """The directories an installed OpenMM keeps its benchmark files in, in the
    order they are tried. Only $CUPO_OPENMM_BENCHMARKS_DIR when that is set."""
    override = os.environ.get(DIR_ENV)
    if override:
        return [Path(override)]
    prefixes = []
    spec = importlib.util.find_spec("openmm")
    if spec is not None and spec.origin:
        # <prefix>/lib/python3.X/site-packages/openmm/__init__.py
        package = Path(spec.origin).resolve().parent
        if len(package.parents) > 3:
            prefixes.append(package.parents[3])
    prefixes.append(Path(sys.prefix))
    if os.environ.get("CONDA_PREFIX"):
        prefixes.append(Path(os.environ["CONDA_PREFIX"]))
    prefixes.append(Path("/opt/openmm"))
    dirs = []
    for prefix in prefixes:
        if prefix / _SUBDIR not in dirs:
            dirs.append(prefix / _SUBDIR)
    return dirs


def default_cache_dir() -> Path:
    """Where downloads go when the caller names no directory: the user's cache
    directory ($XDG_CACHE_HOME, else ~/.cache)."""
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "nv-openmm-plugin" / "openmm-benchmarks"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def openmm_benchmark_file(name: str, cache_dir=None) -> Path:
    """The path of OpenMM's benchmark file for system ``name`` ("apoa1" or "dhfr").

    ``cache_dir`` is where a download is kept, in a subdirectory named after
    ``OPENMM_RELEASE``; default_cache_dir() when not given. See the module's
    docstring for the order the places are tried in.

    Raises FileNotFoundError when the file is not installed and cannot be
    downloaded (in CI, when it is not installed), and ValueError when a download
    does not have the recorded SHA-256.
    """
    if name not in SYSTEMS:
        raise KeyError("%r is not one of OpenMM's benchmark systems (%s)"
                       % (name, ", ".join(SYSTEMS)))
    filename, sha256 = SYSTEMS[name]
    tried = []
    for d in installed_dirs():
        path = d / filename
        if not path.is_file():
            tried.append("%s (not there)" % path)
        elif _sha256(path) != sha256:
            tried.append("%s (a different file: its SHA-256 is not %s)" % (path, sha256))
        else:
            return path

    how_to_fix = ("Install OpenMM from conda-forge, which installs the file under "
                  "<prefix>/%s, or set %s to a directory holding %s."
                  % (_SUBDIR, DIR_ENV, filename))
    if os.environ.get("CI"):
        raise FileNotFoundError(
            "OpenMM's benchmark file %s for %r was not found, and CI is set, so it "
            "is not downloaded. Tried:\n  %s\n%s"
            % (filename, name, "\n  ".join(tried), how_to_fix))

    cache = Path(cache_dir or default_cache_dir()) / OPENMM_RELEASE
    cached = cache / filename
    if cached.is_file() and _sha256(cached) == sha256:
        return cached

    url = URL.format(file=filename)
    print("Downloading %s from %s into %s" % (filename, url, cache), file=sys.stderr)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    except (urllib.error.URLError, OSError) as e:
        raise FileNotFoundError(
            "OpenMM's benchmark file %s for %r is not installed, and downloading it "
            "from %s failed: %s. Tried:\n  %s\n%s"
            % (filename, name, url, e, "\n  ".join(tried), how_to_fix)) from e
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise ValueError("%s: the download's SHA-256 is %s, not the recorded %s"
                         % (url, got, sha256))
    # Written under a temporary name and renamed into place, so that a run started
    # at the same time never reads a file half written.
    cache.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=cache, prefix=filename + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)  # mkstemp makes it 0600
        os.replace(tmp, cached)
    except BaseException:
        os.unlink(tmp)
        raise
    return cached
