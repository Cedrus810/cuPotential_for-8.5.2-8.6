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
