#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SITE="$PREFIX/lib/python${PY_VER}/site-packages"
PKG="$SITE/nv_openmm_plugin"
mkdir -p "$PKG/lib"

cp "$SRC_DIR/python/nv_openmm_plugin/"*.py "$PKG/"
cp "$SRC_DIR/python/nv_openmm_plugin/_nvplugin.so" "$PKG/"
cp "$SRC_DIR/python/nv_openmm_plugin/lib/"*.so "$PKG/lib/"
# The cuPo runtime, under its SONAME: libcupo.so.<ABI>.
cp "$SRC_DIR/python/nv_openmm_plugin/lib/"libcupo.so.* "$PKG/lib/"
cp -a "$SRC_DIR/python/nv_openmm_plugin/notices" "$PKG/"
cp "$SRC_DIR/python/nv_openmm_plugin/runtime_build.json" "$PKG/"
mkdir -p "$PKG/notices"
cp "$SRC_DIR/python/nv_openmm_plugin/notices/LEPTON_NOTICE.txt" "$PKG/notices/"

# Every relative import in the package must resolve inside it. Checked statically
# because importing for real needs CUDA libraries conda-build has no reason to have.
python - "$PKG" <<"PY_END"
import ast, pathlib, sys

pkg = pathlib.Path(sys.argv[1])
have = {p.stem for p in pkg.glob("*.py")}
have |= {p.name.split(".")[0] for p in pkg.glob("*.so")}
missing = []
for src in sorted(pkg.glob("*.py")):
    for node in ast.walk(ast.parse(src.read_text(), filename=str(src))):
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            if node.module.split(".")[0] not in have:
                missing.append("%s imports .%s" % (src.name, node.module))
if missing:
    sys.exit("recipe: the package is incomplete -- " + "; ".join(missing)
             + ". Copied: " + ", ".join(sorted(have)))
print("recipe: %d modules, every relative import resolves" % len(have))
PY_END
