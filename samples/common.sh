# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared by every sample's run script: where cupo's headers, LTO-IR objects and
# libraries are, and which GPU the sample is built for.
#
# Sourced, not executed:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
#     cupo_layout "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
#
# The root is the samples directory's parent: the unpacked cupo tarball in a
# release, the repository after ./run_build in the source tree, or the OpenMM
# source tarball or a clone of its GitHub tree. A clone has no build/: it is on
# the GitHub release, and cupo_layout downloads it with the root's fetch_cupo, as
# plugins/openmm/run_build does. The layouts are the same except for the generated
# include/cupo/package_version.h (the tree's is in build/include/).

# Print the compact targets recorded beside libcupo.so, separated by semicolons.
# An empty array means compact convolution was disabled. Missing or malformed
# metadata must fail rather than silently skipping a backend that should work.
cupo_compact_architectures() {
    python3 - "$1/build/runtime_build.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    architectures = json.loads(path.read_text())["compact_architectures"]
    if not isinstance(architectures, list) or any(
            type(arch) is not int or arch <= 0 for arch in architectures):
        raise ValueError("compact_architectures must be an array of positive SM integers")
except (OSError, ValueError, KeyError, TypeError) as error:
    sys.exit(f"Invalid compact architecture metadata in {path}: {error}. "
             "Rebuild or download the matching cuPO runtime.")
print(";".join(map(str, architectures)))
PY
}

# cupo_layout ROOT sets:
#   CUPO_INCLUDE_FLAGS     -I flags for include/cupo/
#   CUPO_LTOIR_DIR         the directory holding standard/ and fast_math/
#   CUPO_LIB_DIR           the directory holding libcupo.so
#   CUPO_COMPACT_ARCHS     semicolon-separated compact targets, or "" if disabled
cupo_layout() {
    local root="$1"
    CUPO_INCLUDE_FLAGS=()
    if [ -d "$root/build/include" ]; then
        CUPO_INCLUDE_FLAGS+=(-I"$root/build/include")
    fi
    CUPO_INCLUDE_FLAGS+=(-I"$root/include")
    CUPO_LTOIR_DIR="$root/build/ltoir"
    CUPO_LIB_DIR="$root/build"
    # Neither built nor buildable here, so a clone of the GitHub tree: build/ comes
    # from the release (fetch_cupo says which release, how and from where).
    if [ ! -d "$CUPO_LTOIR_DIR/standard" ] && [ -f "$root/fetch_cupo" ] \
            && [ ! -f "$root/run_build" ]; then
        python3 "$root/fetch_cupo" || return 1
    fi
    if [ ! -d "$CUPO_LTOIR_DIR/standard" ]; then
        echo "No LTO-IR objects in $CUPO_LTOIR_DIR: run ./run_build in the source" >&2
        echo "tree, or run the sample from an unpacked release tarball." >&2
        return 1
    fi
    CUPO_COMPACT_ARCHS=$(cupo_compact_architectures "$root") || return 1
}

# Print the selected CUDA device's SM, respecting CUDA_VISIBLE_DEVICES.
# Uses the toolkit the samples need anyway, so no Python is needed.
detect_architecture() {
    local probe_dir
    probe_dir=$(mktemp -d "${TMPDIR:-/tmp}/cupo-device-arch.XXXXXX")
    cat > "$probe_dir/device_architecture.cpp" <<'CPP'
#include <cuda_runtime_api.h>
#include <cstdio>

int main() {
    int device = 0, major = 0, minor = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status == cudaSuccess)
        status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (status == cudaSuccess)
        status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Cannot query selected CUDA device: %s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("%d\n", 10 * major + minor);
    return 0;
}
CPP
    local status=0
    nvcc -std=c++17 "$probe_dir/device_architecture.cpp" -o "$probe_dir/device_architecture" >&2 \
        && "$probe_dir/device_architecture" || status=$?
    rm -rf "$probe_dir"
    return "$status"
}
