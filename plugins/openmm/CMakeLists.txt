# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

# Detect CUDA architectures from the pre-built cupo runtime (libcupo.so).
# Parses embedded PTX ".target sm_XX" directives — works across toolkit versions
# and doesn't require the cupo source tree.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  if(CUPO_RUNTIME_DIR)
    # Source tarball: lib/; full tree: build/
    if(EXISTS "${CUPO_RUNTIME_DIR}/lib/libcupo.so")
      set(_CUPO_LIB "${CUPO_RUNTIME_DIR}/lib/libcupo.so")
    else()
      set(_CUPO_LIB "${CUPO_RUNTIME_DIR}/build/libcupo.so")
    endif()
  else()
    # Auto-detect: cupo/ beside us (src tarball) or ../../build (full tree)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cupo/lib/libcupo.so")
      set(_CUPO_LIB "${CMAKE_CURRENT_LIST_DIR}/cupo/lib/libcupo.so")
    else()
      set(_CUPO_LIB "${CMAKE_CURRENT_LIST_DIR}/../../build/libcupo.so")
    endif()
  endif()
  if(EXISTS "${_CUPO_LIB}")
    execute_process(
      COMMAND strings "${_CUPO_LIB}"
      COMMAND grep ".target sm_"
      COMMAND sort -u
      OUTPUT_VARIABLE _SM_RAW
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    string(REGEX MATCHALL "sm_([0-9]+)" _SM_MATCHES "${_SM_RAW}")
    set(_ARCHS "")
    foreach(_m ${_SM_MATCHES})
      string(REGEX REPLACE "sm_" "" _sm "${_m}")
      list(APPEND _ARCHS "${_sm}")
    endforeach()
    list(REMOVE_DUPLICATES _ARCHS)
    # The pipeline above sorts text, so sm_100 comes before sm_75. Sort the
    # numbers instead: this list is printed and used to name files.
    list(SORT _ARCHS COMPARE NATURAL)
    if(_ARCHS)
      set(CMAKE_CUDA_ARCHITECTURES "${_ARCHS}")
      message(STATUS "Detected CUDA architectures from cupo: ${_ARCHS}")
    else()
      message(FATAL_ERROR
        "Could not detect CUDA architectures from ${_CUPO_LIB}. "
        "Set CMAKE_CUDA_ARCHITECTURES explicitly.")
    endif()
  else()
    message(FATAL_ERROR
      "libcupo.so not found at ${_CUPO_LIB}. Build cupo first, "
      "or set CMAKE_CUDA_ARCHITECTURES explicitly.")
  endif()
endif()

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _VERSION)
string(STRIP "${_VERSION}" _VERSION)
# project(VERSION) takes digits and dots only; a PEP 440 local label such as
# "+openmm86" belongs to the package version (pyproject.toml reads VERSION
# itself), not to CMake's.
string(REGEX REPLACE "\\+.*$" "" _VERSION "${_VERSION}")
project(OpenMMNvidiaPlugin VERSION ${_VERSION} LANGUAGES CXX)

option(OPENMM_NVIDIA_STANDALONE "Build the Nvidia plugin outside the OpenMM source tree" ON)

# Source paths compiled into the plugin, such as __FILE__ in NV_CUDA_CHECK's
# message, relative to the repository two directories up: -ffile-prefix-map maps
# its directory to nothing, so the message names
# "plugins/openmm/cpp/platforms/cuda/src/nonbonded/CudaNonbondedKernels.cu". nvcc
# hands it to the host compiler. In every build, as cupo's own build does.
get_filename_component(_repo_dir "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:-ffile-prefix-map=${_repo_dir}/=>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${_repo_dir}/=>")

if(NOT DEFINED CUPO_RUNTIME_DIR AND NOT DEFINED ENV{CUPO_RUNTIME_DIR})
    # Source tarball layout: cupo/ beside this CMakeLists.txt
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cupo/include/cupo")
        set(CUPO_RUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}/cupo" CACHE PATH
            "Cupo runtime source/install tree")
    else()
        # Full source tree: ../../include/cupo
        get_filename_component(CUPO_SOURCE_ROOT
            "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
        if(EXISTS "${CUPO_SOURCE_ROOT}/include/cupo")
            set(CUPO_RUNTIME_DIR "${CUPO_SOURCE_ROOT}" CACHE PATH
                "Cupo runtime source/install tree")
        endif()
    endif()
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT OPENMM_INCLUDE_DIR)
    find_path(OPENMM_INCLUDE_DIR
        NAMES openmm/OpenMM.h
        PATH_SUFFIXES include)
endif()
if(NOT OPENMM_CORE_LIBRARY)
    find_library(OPENMM_CORE_LIBRARY NAMES OpenMM)
endif()
if(NOT OPENMM_CUDA_LIBRARY)
    find_library(OPENMM_CUDA_LIBRARY NAMES OpenMMCUDA)
endif()

if(NOT OPENMM_INCLUDE_DIR OR NOT OPENMM_CORE_LIBRARY OR NOT OPENMM_CUDA_LIBRARY)
    message(FATAL_ERROR
        "Set OPENMM_INCLUDE_DIR, OPENMM_CORE_LIBRARY, and OPENMM_CUDA_LIBRARY "
        "to build the standalone plugin against stock OpenMM.")
endif()

# OpenMM series probe. OpenMM ships no version header in any series we support
# (8.4-8.6): the version exists only as the runtime string
# Platform::getOpenMMVersion(). The series decides the NV_OPENMM_GE_* macros
# (attached to the plugin targets in cpp/CMakeLists.txt) and the openmm bounds
# recorded in plugin_build.json, which write_package_metadata.py carries into
# the package for the Python loader and both package managers. Discovered by
# compiling and running a one-liner against OPENMM_INCLUDE_DIR and
# OPENMM_CORE_LIBRARY, so it works for conda and pip layouts alike;
# -DOPENMM_SERIES=8.5 skips the probe. The allowed set guards against pairing
# with a series the plugin was never built for -- libOpenMM.so has no versioned
# soname, so a wrong pairing loads silently (recipe/meta.yaml documents an
# install test that resolved 8.6.0 against a >=8.4 pin exactly that way).
set(NV_OPENMM_ALLOWED_SERIES "8.4;8.5;8.6" CACHE STRING
    "OpenMM major.minor series the plugin may be built against")
if(NOT OPENMM_SERIES)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/openmm_series_probe.cpp"
        "#include <iostream>\n"
        "#include <openmm/Platform.h>\n"
        "int main() { std::cout << OpenMM::Platform::getOpenMMVersion() << std::endl; }\n")
    get_filename_component(_OPENMM_LIB_DIR "${OPENMM_CORE_LIBRARY}" DIRECTORY)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -x c++
            "${CMAKE_CURRENT_BINARY_DIR}/openmm_series_probe.cpp"
            # Quote whole arguments: in an unquoted argument such as -I"${X}"
            # CMake keeps the quotes literally, and the compiler gets a path
            # with quote characters in it.
            "-I${OPENMM_INCLUDE_DIR}" "-L${_OPENMM_LIB_DIR}"
            "-Wl,-rpath,${_OPENMM_LIB_DIR}" -lOpenMM
            -o "${CMAKE_CURRENT_BINARY_DIR}/openmm_series_probe"
        RESULT_VARIABLE _PROBE_BUILD
        ERROR_VARIABLE _PROBE_BUILD_ERR)
    if(NOT _PROBE_BUILD EQUAL 0)
        string(REPLACE "\n" "; " _PROBE_ERR "${_PROBE_BUILD_ERR}")
        message(FATAL_ERROR
            "Could not build the OpenMM version probe against "
            "OPENMM_INCLUDE_DIR=${OPENMM_INCLUDE_DIR} and "
            "OPENMM_CORE_LIBRARY=${OPENMM_CORE_LIBRARY}: ${_PROBE_ERR}. "
            "Set -DOPENMM_SERIES=<major.minor> to skip the probe.")
    endif()
    set(_OLD_LLP "$ENV{LD_LIBRARY_PATH}")
    set(ENV{LD_LIBRARY_PATH} "${_OPENMM_LIB_DIR}:${_OLD_LLP}")
    execute_process(
        COMMAND "${CMAKE_CURRENT_BINARY_DIR}/openmm_series_probe"
        OUTPUT_VARIABLE _OPENMM_FULL_VERSION
        RESULT_VARIABLE _PROBE_RUN
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    set(ENV{LD_LIBRARY_PATH} "${_OLD_LLP}")
    if(NOT _PROBE_RUN EQUAL 0 OR NOT _OPENMM_FULL_VERSION)
        message(FATAL_ERROR
            "The OpenMM version probe did not run (built against "
            "OPENMM_CORE_LIBRARY=${OPENMM_CORE_LIBRARY}). "
            "Set -DOPENMM_SERIES=<major.minor> to skip the probe.")
    endif()
    set(OPENMM_SERIES "${_OPENMM_FULL_VERSION}")
endif()
# "8.5.2" and a dev build like "8.5.2.dev-36a30cb" alike reduce to the series
# "8.5"; the upper bound is the next minor, matching the closed per-series caps
# the recipe keeps.
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _OPENMM_SERIES_MATCH "${OPENMM_SERIES}")
if(NOT _OPENMM_SERIES_MATCH)
    message(FATAL_ERROR "Cannot parse an OpenMM series out of '${OPENMM_SERIES}'. "
        "Set -DOPENMM_SERIES=<major.minor> explicitly.")
endif()
set(_OPENMM_MAJOR "${CMAKE_MATCH_1}")
set(_OPENMM_MINOR "${CMAKE_MATCH_2}")
set(OPENMM_SERIES "${_OPENMM_MAJOR}.${_OPENMM_MINOR}")
if(NOT OPENMM_SERIES IN_LIST NV_OPENMM_ALLOWED_SERIES)
    message(FATAL_ERROR
        "OpenMM ${OPENMM_SERIES} (${_OPENMM_FULL_VERSION}) is not a series this "
        "plugin supports (${NV_OPENMM_ALLOWED_SERIES}). Build against a listed "
        "series, or extend NV_OPENMM_ALLOWED_SERIES if you know the pairing.")
endif()
math(EXPR _OPENMM_NEXT_MINOR "${_OPENMM_MINOR} + 1")
set(NV_OPENMM_SERIES_MIN "${OPENMM_SERIES}")
set(NV_OPENMM_SERIES_MAX "${_OPENMM_MAJOR}.${_OPENMM_NEXT_MINOR}")
set(NV_OPENMM_COMPILE_DEFINITIONS)
if(NOT _OPENMM_MINOR LESS 5)
    list(APPEND NV_OPENMM_COMPILE_DEFINITIONS NV_OPENMM_GE_85)
endif()
if(NOT _OPENMM_MINOR LESS 6)
    list(APPEND NV_OPENMM_COMPILE_DEFINITIONS NV_OPENMM_GE_86)
endif()
message(STATUS "OpenMM series: ${OPENMM_SERIES} (${_OPENMM_FULL_VERSION}); "
    "bounds ${NV_OPENMM_SERIES_MIN}..<${NV_OPENMM_SERIES_MAX}; "
    "defines ${NV_OPENMM_COMPILE_DEFINITIONS}")

set(OPENMM_EXTRA_INCLUDE_DIRS "" CACHE STRING
    "Additional OpenMM include directories needed for source-tree builds")
set(OPENMM_SOURCE_DIR "" CACHE PATH
    "Optional OpenMM source tree root for standalone source-tree builds")
set(OPENMM_BUILD_DIR "" CACHE PATH
    "Optional OpenMM build tree root for generated headers")

if(NOT OPENMM_SOURCE_DIR)
    get_filename_component(_OPENMM_SOURCE_CANDIDATE
        "${OPENMM_INCLUDE_DIR}/../.." ABSOLUTE)
    if(EXISTS "${_OPENMM_SOURCE_CANDIDATE}/serialization/include")
        set(OPENMM_SOURCE_DIR "${_OPENMM_SOURCE_CANDIDATE}")
    endif()
endif()

if(OPENMM_SOURCE_DIR)
    list(APPEND OPENMM_EXTRA_INCLUDE_DIRS
        "${OPENMM_SOURCE_DIR}/olla/include"
        "${OPENMM_SOURCE_DIR}/libraries/lepton/include"
        "${OPENMM_SOURCE_DIR}/serialization/include"
        "${OPENMM_SOURCE_DIR}/platforms/common/include"
        "${OPENMM_SOURCE_DIR}/platforms/cuda/include")
    if(NOT OPENMM_BUILD_DIR AND EXISTS "${OPENMM_SOURCE_DIR}/build")
        set(OPENMM_BUILD_DIR "${OPENMM_SOURCE_DIR}/build")
    endif()
endif()

if(OPENMM_BUILD_DIR)
    list(APPEND OPENMM_EXTRA_INCLUDE_DIRS
        "${OPENMM_BUILD_DIR}/platforms/common/src"
        "${OPENMM_BUILD_DIR}/platforms/cuda/src")
endif()
list(REMOVE_DUPLICATES OPENMM_EXTRA_INCLUDE_DIRS)

set(OPENMM_BUILD_CUDA_LIB ON CACHE BOOL "Build CUDA implementation" FORCE)
add_subdirectory(cpp)
