# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build through ./run_docker, or pass the CUDA version from .cuda-version:
#   docker build --build-arg CUDA_VERSION=$(cat .cuda-version) .
ARG CUDA_VERSION
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
# Again after FROM, for the RUN lines below: an ARG before FROM is seen by FROM only.
ARG CUDA_VERSION

RUN apt-get update -qq
RUN apt-get install -y -qq --no-install-recommends \
        make python3 python3-dev python3-pip python3-numpy \
        python3-pytest
RUN apt-get install -y -qq --no-install-recommends curl bzip2 sudo

RUN pip3 install --quiet --break-system-packages cmake
# sccache from its GitHub release, not pip: PyPI stops at 0.16.0, which cannot
# cache nvcc's LTO-IR compiles (code=lto_XX). 0.16 and 0.17 take the wrong
# argument as cicc's input and fatbinary then fails with "Could not open input
# file '...ltoir'"; 0.18.0 fixed it. Pinned with the sha256 of each tarball.
RUN SCCACHE_VERSION=0.18.0 && \
    case "$(uname -m)" in \
        x86_64) SCCACHE_SHA256=45f1447fbe231e3037bde351ef70677dd212216c8d62ae7ca409fecc4d6acc89 ;; \
        aarch64) SCCACHE_SHA256=2b3284d5da3b46a47dc4229e75bb7b88ac4aa99c8d754fb7d2f84997e5a4354a ;; \
        *) echo "no sccache release for $(uname -m)" >&2; exit 1 ;; \
    esac && \
    SCCACHE_NAME="sccache-v${SCCACHE_VERSION}-$(uname -m)-unknown-linux-musl" && \
    curl -fsSL -o /tmp/sccache.tar.gz \
        "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/${SCCACHE_NAME}.tar.gz" && \
    echo "${SCCACHE_SHA256}  /tmp/sccache.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/sccache.tar.gz -C /usr/local/bin --strip-components=1 "${SCCACHE_NAME}/sccache" && \
    rm /tmp/sccache.tar.gz && \
    sccache --version
# pybind11 from pip, not apt: distro 2.11.x predates numpy 2 runtime support
# (stride-0 arrays); [global] puts the CMake config on the default search path.
RUN pip3 install --quiet --break-system-packages "pybind11[global]>=2.12"
RUN pip3 install --quiet --break-system-packages --ignore-installed numpy cupy-cuda${CUDA_VERSION%%.*}x
RUN pip3 install --quiet --break-system-packages "jax[cuda${CUDA_VERSION%%.*}]"
RUN pip3 install --quiet --break-system-packages matplotlib
# mpmath backs the C2 Prolate scientific construction, table generators and
# independent high-precision reference checks, so it is also a test dependency.
RUN pip3 install --quiet --break-system-packages mpmath
# zstandard: the CI reads the .tar.zst artifacts with it. It used
# to arrive unasked, as a dependency of conda-build's packages in /opt/openmm, and
# an image rebuilt from today's conda-forge no longer has it there.
RUN pip3 install --quiet --break-system-packages zstandard

# MathDx 26.06.1, the complete SDK the compact convolution library is compiled
# from: its cuFFTDx headers, and the LTO helper in
# example/cufftdx/lto_helper, which the headers-only wheel nvidia-mathdx on PyPI
# does not have. MATHDX_ROOT tells CMake where it is. There is one tarball per CUDA
# major, taken from CUDA_VERSION as the cupy and jax lines do; the sha256 is the
# one cupo's build checks. doc/ is 51 MB of HTML that no build reads.
RUN MATHDX_VERSION=26.06.1 && \
    CUDA_MAJOR=${CUDA_VERSION%%.*} && \
    case "$CUDA_MAJOR" in \
        13) MATHDX_SHA256=59a9233db34b75568acbcc5284e6cefe6fad5577ee644f85044971d62eeea353 ;; \
        *) echo "no MathDx ${MATHDX_VERSION} pinned for CUDA ${CUDA_MAJOR}" >&2; exit 1 ;; \
    esac && \
    MATHDX_NAME="nvidia-mathdx-${MATHDX_VERSION}-cuda${CUDA_MAJOR}" && \
    curl -fsSL --retry 5 --retry-all-errors -o /tmp/mathdx.tar.gz \
        "https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx/cuda${CUDA_MAJOR}/${MATHDX_NAME}.tar.gz" && \
    echo "${MATHDX_SHA256}  /tmp/mathdx.tar.gz" | sha256sum -c - && \
    mkdir -p /opt/nvidia && \
    tar -xzf /tmp/mathdx.tar.gz -C /opt/nvidia --strip-components=2 "${MATHDX_NAME}/nvidia/mathdx" && \
    rm -r /tmp/mathdx.tar.gz /opt/nvidia/mathdx/26.06/doc && \
    test -f /opt/nvidia/mathdx/26.06/example/cufftdx/lto_helper/cufftdx_cufft_lto_helper.cpp
ENV MATHDX_ROOT=/opt/nvidia/mathdx/26.06

# OpenMM with its CUDA platform, from conda-forge. Pinned, because the plugin is
# built and supported against one OpenMM series, and OpenMM's CUDA build has to
# match the CUDA in .cuda-version, which the base image uses too. OpenMM compiles
# its kernels with the environment's nvrtc, and a driver older than that nvrtc
# rejects the PTX with CUDA_ERROR_UNSUPPORTED_PTX_VERSION. The base image's compat
# driver covers data center and RTX Pro cards; a GeForce card uses the host's
# driver, which must support the CUDA in .cuda-version.
RUN ARCH=$(uname -m | sed 's/x86_64/64/') && \
    curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-${ARCH}/latest" \
        | tar -xvj -C /usr/local bin/micromamba
# swig from conda-forge (4.3+) — must match the SWIG runtime ABI used to build
# OpenMM's _openmm.so (swig_runtime_data5); apt's swig 4.2 produces runtime_data4.
RUN micromamba create -y -p /opt/openmm -c conda-forge openmm=8.4.0 python=3.12 cuda-version=${CUDA_VERSION%.*} swig conda-build
# Allow non-root users to install additional conda packages into the prefix.
RUN chmod -R a+w /opt/openmm/conda-meta && chmod a+w /opt/openmm/lib/python3.12/site-packages
ENV OPENMM_PLUGIN_DIR=/opt/openmm/lib/plugins
ENV LD_LIBRARY_PATH=/opt/openmm/lib:$LD_LIBRARY_PATH
# Make conda env's site-packages visible to system python (replaces per-package symlinks).
RUN echo "/opt/openmm/lib/python3.12/site-packages" > /usr/local/lib/python3.12/dist-packages/openmm-env.pth
# The system python then imports conda-forge's pip from there, and that pip's
# certifi looks for its CA bundle at <sys.base_prefix>/ssl/cacert.pem: for the system
# python, /usr/ssl/cacert.pem, which does not exist, and every pip command that opens
# a session (pip wheel, when the plugin's wheel is built) fails with
# FileNotFoundError. Point it at the conda environment's bundle.
RUN mkdir -p /usr/ssl && ln -s /opt/openmm/ssl/cacert.pem /usr/ssl/cacert.pem
# pip installs into this image's Python, the way the install docs say, without a
# virtual environment: each ./run_docker container is removed when it ends, so
# Ubuntu's protection of the system Python (PEP 668) protects nothing here.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Allow any user to lock/unlock GPU clocks without password.
# Skip PAM account validation (pam_unix checks /etc/shadow which isn't available
# when running as a mapped host user).
RUN echo "ALL ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi" > /etc/sudoers.d/nvidia-smi && \
    sed -i 's/@include common-account/account sufficient pam_permit.so/' /etc/pam.d/sudo
