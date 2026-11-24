# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate runtime/compiler compatibility and derive wheel/conda requirements."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--runtime-dir', type=Path, required=True)
parser.add_argument('--plugin-root', type=Path, required=True)
args = parser.parse_args()
metadata = json.loads((args.runtime_dir/'runtime_build.json').read_text())
library = args.runtime_dir/'libcupo.so'
if hashlib.sha256(library.read_bytes()).hexdigest() != metadata['library_sha256']:
    raise SystemExit('cuPo runtime metadata does not match libcupo.so; rebuild the runtime')
plugin_metadata = json.loads((args.plugin_root/'build/plugin_build.json').read_text())
if plugin_metadata['cuda_min'] != metadata['cuda_min']:
    raise SystemExit('Build the OpenMM plugin and cuPo runtime with the same CUDA major/minor')
# The OpenMM series bounds the CMake probe recorded. Every consumer -- the
# wheel requirements below, the Python loader's import-time gate, and
# recipe/meta.yaml's conda pin -- reads the same pair, so one probe run keeps
# all three in step.
for key in ('openmm_min', 'openmm_max'):
    if key not in plugin_metadata:
        raise SystemExit(f'plugin_build.json lacks {key}; the OpenMM-series probe '
                         'in plugins/openmm/CMakeLists.txt writes it -- rebuild')
    metadata[key] = plugin_metadata[key]
package = args.plugin_root/'python/nv_openmm_plugin'
# The package holds the runtime once, under its SONAME (run_build).
shipped = sorted((package/'lib').glob('libcupo.so*'))
if len(shipped) != 1 or shipped[0].read_bytes() != library.read_bytes():
    raise SystemExit('The Python package contains a different cuPo runtime')
plugin_library = args.plugin_root/'build/cpp/platforms/cuda/libOpenMMNvidiaCUDA.so'
shipped_plugin = package/'lib/libOpenMMNvidiaCUDA.so'
if shipped_plugin.read_bytes() != plugin_library.read_bytes():
    raise SystemExit('The Python package contains a different CUDA plugin')
metadata['plugin_library_sha256'] = hashlib.sha256(plugin_library.read_bytes()).hexdigest()
(package/'runtime_build.json').write_text(json.dumps(metadata, indent=2)+'\n')
requirements = [f'{name}>={metadata[minimum]},<{metadata[maximum]}' for name, minimum, maximum in [
    ('nvidia-cuda-runtime', 'cuda_min', 'cuda_max'),
    ('nvidia-cufft', 'cufft_min', 'cufft_max'),
    ('nvidia-nvjitlink', 'cuda_min', 'cuda_max'),
    ('nvidia-cuda-nvrtc', 'cuda_min', 'cuda_max')]]
# OpenMM, capped to the series these binaries were built against, as recipe/meta.yaml
# caps it for conda. The bounds come straight from the build's OpenMM-series probe
# (plugin_build.json), not a hand-edited literal. The CUDA extra is named so that pip
# moves OpenMM and its CUDA platform (OpenMM-CUDA-<major>, which the extra pins to the
# same version) together, never one without the other. A conda OpenMM inside the range
# satisfies this as it is: pip sees it through its egg-info, which declares no
# extras, so pip warns that the extra is missing and installs nothing over it.
cuda_major = metadata['cuda_min'].split('.')[0]
requirements.append(f"openmm[cuda{cuda_major}]>={metadata['openmm_min']},<{metadata['openmm_max']}")
(args.plugin_root/'runtime-requirements.txt').write_text('\n'.join(requirements)+'\n')
