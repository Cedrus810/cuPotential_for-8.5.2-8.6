# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared by every sample's run script: make nv_openmm_plugin importable, or say
# why it cannot be.
#
# Sourced, not executed. A sample's run script is then two lines:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
#     exec python3 sample.py "$@"
#
# The package is looked for beside the samples directory, which is the tarball
# root in a release and plugins/openmm in the source tree. Three shapes ship, so
# three are handled: a wheel, a conda package, or an already-built python/ tree.

ensure_plugin() {
    local root="$1"          # where a package would sit: the samples' parent
    local import_err
    if import_err=$(python3 -c "import nv_openmm_plugin" 2>&1); then
        return 0
    fi
    if [ -e "$root"/*.whl ]; then
        pip install --quiet --user --break-system-packages "$root"/*.whl
    elif [ -e "$root"/*.conda ]; then
        micromamba install -y -p /opt/openmm --offline "$root"/*.conda
    elif [ -d "$root/python/nv_openmm_plugin" ]; then
        export PYTHONPATH="$root/python${PYTHONPATH:+:$PYTHONPATH}"
    else
        echo "ERROR: nv_openmm_plugin will not import, and there is no package to" >&2
        echo "       install in $root" >&2
        echo "--- what the import said ---" >&2
        echo "$import_err" >&2
        return 1
    fi
}
