# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#---------------------------------------------------
# OpenMM NvNonbondedForce CUDA Plugin
#
# Creates OpenMMNvidiaCUDA library.
# Unix: libOpenMMNvidiaCUDA.so
#
# Uses CUDA kernels compiled at build time by nvcc
# via CMake's native CUDA language support (enable_language(CUDA)).
#----------------------------------------------------

# Enable native CUDA compilation so .cu files are compiled by nvcc.
# This call is scoped to the project (not just this subdirectory).
enable_language(CUDA)
string(APPEND CMAKE_CUDA_FLAGS " --threads=0")

SET(OPENMMNVDIACUDA_LIBRARY_NAME OpenMMNvidiaCUDA)
SET(SHARED_TARGET ${OPENMMNVDIACUDA_LIBRARY_NAME})

# Collect source files (.cpp and .cu), one subdirectory per component.
#
#   common/      the kernel factory -- the one piece that knows about all components
#   nonbonded/   unified nonbonded execution and expression input adaptation
#   barostat/    NvMonteCarloBarostat's device work
#   integrator/  NvLangevinMiddleIntegrator, its AOT kernels and constraints
#   bonded/      grouped CUPO bonded adapter and stock OpenMM fallback
#
# Listed explicitly rather than globbed recursively, so a new component has to be
# declared here instead of appearing by accident.
SET(SOURCE_FILES)
SET(SOURCE_INCLUDE_FILES)
SET(NV_CUDA_COMPONENTS common nonbonded barostat integrator bonded)

FOREACH(component ${NV_CUDA_COMPONENTS})
    FILE(GLOB c_src   ${CMAKE_CURRENT_SOURCE_DIR}/src/${component}/*.cpp)
    FILE(GLOB c_cu    ${CMAKE_CURRENT_SOURCE_DIR}/src/${component}/*.cu)
    FILE(GLOB c_incl  ${CMAKE_CURRENT_SOURCE_DIR}/src/${component}/*.hpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/${component}/*.cuh)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${c_src} ${c_cu})
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${c_incl})
ENDFOREACH()

# Include directories. Every header lives beside the code that uses it, in its
# component folder, so src/ is the only one needed: a same-folder include resolves
# on its own, and the one cross-component case (the factory) spells out the
# component it is reaching into.
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# OpenMM CUDA platform headers
IF(OPENMM_NVIDIA_STANDALONE)
    INCLUDE_DIRECTORIES(BEFORE ${OPENMM_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(BEFORE ${OPENMM_EXTRA_INCLUDE_DIRS})
ELSE()
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/include)
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/src)
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/cuda/src)
ENDIF()

# OpenMM common platform headers
IF(NOT OPENMM_NVIDIA_STANDALONE)
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/common/include)
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/common/src)
ENDIF()

# CUDA toolkit
INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})

# NvNonbondedForce API headers
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/../../openmmapi/include)

ADD_LIBRARY(${SHARED_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES})
target_compile_features(${SHARED_TARGET} PRIVATE cxx_std_17)
# Expression compilation and prepared layouts belong to Cupo::Runtime.
# Record the plugin's CUDA toolkit for package compatibility checks.
file(WRITE "${CMAKE_BINARY_DIR}/plugin_build.json"
    "{\"cuda_min\": \"${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}\"}\n")

SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DOPENMM_BUILDING_SHARED_LIBRARY"
    LINK_FLAGS "${EXTRA_LINK_FLAGS} -Wl,--enable-new-dtags"
    # RUNPATH so the linker finds libcupo.so without LD_LIBRARY_PATH:
    #   $ORIGIN              — wheel layout (all .so's co-located in lib/)
    #   $ORIGIN/../../..              — tarball layout (build/cpp/platforms/cuda/ -> build/)
    #   $ORIGIN/../../../../../../build — source tree (plugins/openmm/build/cpp/platforms/cuda/ -> repo/build/)
    #   $ORIGIN/../../../..           — installed wheel: <site-packages>/nv_openmm_plugin/lib
    #                                   -> the environment's lib/, for libcudart.so.13
    #                                   and libOpenMM.so
    #   $ORIGIN/../../nvidia/cu<N>/lib — installed wheel -> the payload dir pip's CUDA
    #                                   <N> wheels share; <N> is the toolkit cmake
    #                                   found, not a literal, so it follows an image
    #                                   bump the way the sonames do
    # The last two exist because pip, unlike conda-build, relocates nothing.
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../..;\$ORIGIN/../../../../../../build;\$ORIGIN/../../../..;\$ORIGIN/../../nvidia/cu${CUDAToolkit_VERSION_MAJOR}/lib"
    INSTALL_RPATH_USE_LINK_PATH FALSE)

# Match OpenMM's JIT path, which compiles every kernel with --use_fast_math
# (CudaContext defaultOptimizationOptions). Without it the AOT integrator kernels
# use IEEE-precise div/sqrt and lose ~2-3% on div/sqrt-heavy SETTLE/SHAKE (single)
# vs the JIT reference. But we override --ftz=false: fast-math bundles --ftz=true
# (flush denormals), which perturbs the df64/fltflt error-free-transform codegen
# and costs mixed precision ~2.5% across ALL fltflt kernels (even div/sqrt-free
# Part1). With ftz=false, single keeps the approximate-div/sqrt + fast-transcendental
# win and mixed is back to precise-level (Part2 even gains from the fast RNG
# transcendentals). CUDA-only (the .cpp sources would reject the flags).
target_compile_options(${SHARED_TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--ftz=false>)

IF(OPENMM_NVIDIA_STANDALONE)
    TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_CORE_LIBRARY})
    TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_CUDA_LIBRARY})
ELSE()
    TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME})
    TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}CUDA)
ENDIF()
TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${SHARED_NVIDIA_TARGET})
TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${CMAKE_DL_LIBS})

# Link CUDA runtime and cuFFT (required by Cupo SPME)
if(CUDAToolkit_FOUND)
    TARGET_LINK_LIBRARIES(${SHARED_TARGET} CUDA::cudart)
    TARGET_LINK_LIBRARIES(${SHARED_TARGET} CUDA::cuda_driver)
    TARGET_LINK_LIBRARIES(${SHARED_TARGET} CUDA::cufft)
endif()

TARGET_LINK_LIBRARIES(${SHARED_TARGET} Cupo::Runtime)

INSTALL(TARGETS ${SHARED_TARGET}
    DESTINATION lib/plugins)
