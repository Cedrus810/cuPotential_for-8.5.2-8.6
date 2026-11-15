# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The prebuilt SWIG extension makes this a platform/CPython-specific wheel.

Build the wheel with the same Python interpreter used to build the extension.
The extension is deliberately built by CMake, not rebuilt by setuptools.
"""
import os
import posixpath
import re
from pathlib import Path

# The repository the plugin is released to, and where its README sits in the tree
# released there.
PUBLISHED_REPOSITORY = 'NVIDIA/cuPotential'
README_DIR = 'plugins/openmm'

# An inline Markdown link or image: [text](target) or ![alt](target).
_LINK = re.compile(r'(!?)\[([^\]]*)\]\(([^)\s]+)\)')
# The src="target" of an HTML image, <img ... src="target" ...>.
_IMG_SRC = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")')


def absolute_links(text, repository, tag):
    """`text`, the plugin's README, with each relative link made absolute: a link
    to the file at `tag` on GitHub's page for the repository `repository`, and an
    image, Markdown or an HTML <img>, to the file's raw bytes, which is what an
    image needs to show. A target relative to the README is taken from
    README_DIR; one that starts with / from the root of the tree. Anchors
    (#section) and URLs with a scheme, such as https: or mailto:, are left as
    they are."""
    def url(target, kind):
        """The absolute URL for `target`, or None to leave it as it is."""
        if target.startswith('#') or re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target):
            return None
        path, hash_, fragment = target.partition('#')
        if path.startswith('/'):
            path = posixpath.normpath(path.lstrip('/'))
        else:
            path = posixpath.normpath(posixpath.join(README_DIR, path))
        if path == '..' or path.startswith('../'):
            raise ValueError('README link %s leaves the tree' % target)
        return 'https://github.com/%s/%s/%s/%s%s%s' % (repository, kind, tag, path,
                                                       hash_, fragment)

    def link(m):
        image, label, target = m.groups()
        absolute = url(target, 'raw' if image else 'blob')
        return m.group(0) if absolute is None else '%s[%s](%s)' % (image, label, absolute)

    def img(m):
        absolute = url(m.group(2), 'raw')
        return m.group(0) if absolute is None else m.group(1) + absolute + m.group(3)
    return _IMG_SRC.sub(img, _LINK.sub(link, text))


def long_description(root, environ=os.environ):
    """The README as the wheel's METADATA carries it, where a relative link does
    not resolve: a package index's project page, `pip show`. The README in the
    tree and the source tarball keep their relative links.

    The links point at the release of this version, v<VERSION> on
    NVIDIA/cuPotential, which is where a customer gets the wheel.
    GITHUB_REPOSITORY and GITHUB_RELEASE_TAG, when set, name another repository
    or tag."""
    version = (root/'VERSION').read_text().strip()
    repository = environ.get('GITHUB_REPOSITORY') or PUBLISHED_REPOSITORY
    tag = environ.get('GITHUB_RELEASE_TAG') or 'v%s' % version
    return absolute_links((root/'README.md').read_text(), repository, tag)


# setuptools runs this file as __main__; the tests import it for the functions above.
if __name__ == '__main__':
    from setuptools import Distribution, setup

    # Source checkouts need run_build first. Do not quietly emit an incomplete wheel
    # when the prebuilt payload has not been staged.
    root = Path(__file__).resolve().parent
    for relative in (
        'runtime-requirements.txt',
        'python/nv_openmm_plugin/runtime_build.json',
        'python/nv_openmm_plugin/notices/OpenMM-Bonded-MIT.txt',
        'python/nv_openmm_plugin/notices/OpenMM-Bonded-UPSTREAM.txt',
        'python/nv_openmm_plugin/_nvplugin.so',
        'python/nv_openmm_plugin/lib/libOpenMMNvidia.so',
        'python/nv_openmm_plugin/lib/libOpenMMNvidiaCUDA.so',
        'python/nv_openmm_plugin/notices/LEPTON_NOTICE.txt',
        'python/nv_openmm_plugin/notices/NOTICE',
        'python/nv_openmm_plugin/notices/THIRD-PARTY.txt',
        'python/nv_openmm_plugin/notices/LICENSE',
        'python/nv_openmm_plugin/notices/LICENSE-apache.txt',
        'python/nv_openmm_plugin/notices/LICENSE-cupo.txt',
    ):
        if not (root/relative).is_file():
            raise RuntimeError(f'Missing release payload {relative}; run run_build first')
    # The cuPo runtime, one file under its SONAME: libcupo.so.<ABI>.
    runtime = sorted(p.name for p in (root/'python/nv_openmm_plugin/lib').glob('libcupo.so*'))
    if len(runtime) != 1:
        raise RuntimeError('Expected one cuPo runtime in python/nv_openmm_plugin/lib, '
                           f'found {runtime}; run run_build first')

    class BinaryDistribution(Distribution):
        def has_ext_modules(self):
            return True

    setup(distclass=BinaryDistribution, long_description=long_description(root),
          long_description_content_type='text/markdown')
