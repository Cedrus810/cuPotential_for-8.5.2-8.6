# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#---------------------------------------------------
# OpenMM NvNonbondedForce Plugin
#
# Creates OpenMMNvidia plugin library.
# Unix: libOpenMMNvidia.so
#----------------------------------------------------

SET(OPENMM_NVIDIA_PLUGIN_SOURCE_SUBDIRS . openmmapi)

SET(OPENMM_NVIDIA_LIBRARY_NAME OpenMMNvidia)
SET(SHARED_NVIDIA_TARGET ${OPENMM_NVIDIA_LIBRARY_NAME})

# Header directories for the API
SET(API_NVIDIA_INCLUDE_DIRS)
FOREACH(subdir ${OPENMM_NVIDIA_PLUGIN_SOURCE_SUBDIRS})
    SET(API_NVIDIA_INCLUDE_DIRS ${API_NVIDIA_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include/internal)
ENDFOREACH(subdir)

SET(API_NVIDIA_ABS_INCLUDE_FILES)
FOREACH(dir ${API_NVIDIA_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_NVIDIA_ABS_INCLUDE_FILES ${API_NVIDIA_ABS_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# Collect source files
SET(SOURCE_NVIDIA_FILES)
SET(SOURCE_NVIDIA_INCLUDE_FILES)
FOREACH(subdir ${OPENMM_NVIDIA_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_NVIDIA_FILES         ${SOURCE_NVIDIA_FILES}         ${src_files})
    SET(SOURCE_NVIDIA_INCLUDE_FILES ${SOURCE_NVIDIA_INCLUDE_FILES} ${incl_files})
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)
IF(OPENMM_NVIDIA_STANDALONE)
    INCLUDE_DIRECTORIES(BEFORE ${OPENMM_INCLUDE_DIR})
    INCLUDE_DIRECTORIES(BEFORE ${OPENMM_EXTRA_INCLUDE_DIRS})
ENDIF()

ADD_LIBRARY(${SHARED_NVIDIA_TARGET} SHARED
    ${SOURCE_NVIDIA_FILES}
    ${SOURCE_NVIDIA_INCLUDE_FILES}
    ${API_NVIDIA_ABS_INCLUDE_FILES})
target_compile_features(${SHARED_NVIDIA_TARGET} PUBLIC cxx_std_17)
# Series gate, probed by the parent CMakeLists: NV_OPENMM_GE_85/GE_86 let the
# sources carry 8.5/8.6-only call sites behind #if when one ever exists -- the
# audit found none today, so the macros are insurance, not load-bearing.
target_compile_definitions(${SHARED_NVIDIA_TARGET} PRIVATE ${NV_OPENMM_COMPILE_DEFINITIONS})

SET_TARGET_PROPERTIES(${SHARED_NVIDIA_TARGET} PROPERTIES
    LINK_FLAGS "${EXTRA_LINK_FLAGS} -Wl,--enable-new-dtags"
    COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DOPENMM_NVIDIA_BUILDING_SHARED_LIBRARY"
    # RUNPATH: $ORIGIN for wheel layout (co-located in lib/), then the two
    # site-packages-relative climbs that let a pip-installed wheel reach the
    # environment's own lib/ and pip's CUDA payload. pip needs them and conda
    # does not, because pip, unlike conda-build, relocates nothing.
    #
    # Of the two, only the climb to the environment's lib/ does anything here:
    # this target NEEDs libOpenMM.so, which lives there, and no CUDA library at
    # all -- readelf shows libcudart on libOpenMMNvidiaCUDA.so, not on this one.
    # The nvidia/cuN/lib entry is carried so the three targets we ship have one
    # RUNPATH shape between them, and matches nothing today.
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../../..;\$ORIGIN/../../nvidia/cu${CUDAToolkit_VERSION_MAJOR}/lib")

IF(NOT APPLE)
    FIND_LIBRARY(DL_LIBRARY dl)
    IF(DL_LIBRARY)
        TARGET_LINK_LIBRARIES(${SHARED_NVIDIA_TARGET} ${DL_LIBRARY})
    ENDIF(DL_LIBRARY)
ENDIF()

IF(OPENMM_NVIDIA_STANDALONE)
    TARGET_LINK_LIBRARIES(${SHARED_NVIDIA_TARGET} ${OPENMM_CORE_LIBRARY})
ELSE()
    TARGET_LINK_LIBRARIES(${SHARED_NVIDIA_TARGET} ${SHARED_TARGET})
ENDIF()

# Cupo runtime: an installed CMake package (CupoRuntime_DIR), or a directory with
# cupo's include/ and its library in lib/ or build/ (CUPO_RUNTIME_DIR), such as the
# unpacked cupo tarball or the root of this tree.
if(DEFINED ENV{CUPO_RUNTIME_DIR} AND NOT DEFINED CUPO_RUNTIME_DIR)
    SET(CUPO_RUNTIME_DIR $ENV{CUPO_RUNTIME_DIR})
endif()
if(CupoRuntime_DIR)
    find_package(CupoRuntime REQUIRED CONFIG)
elseif(CUPO_RUNTIME_DIR)
    set(_CUPO_RUNTIME_INCLUDE_DIRS "${CUPO_RUNTIME_DIR}/include")
    # The version macros come with the headers, in cupo/package_version.h. A
    # directory that has cmake/CupoPackageVersion.cmake generates them here instead.
    if(EXISTS "${CUPO_RUNTIME_DIR}/cmake/CupoPackageVersion.cmake")
        include("${CUPO_RUNTIME_DIR}/cmake/CupoPackageVersion.cmake")
        cupo_generate_package_version("${CMAKE_CURRENT_BINARY_DIR}/cupo/include")
        list(PREPEND _CUPO_RUNTIME_INCLUDE_DIRS "${CUPO_PACKAGE_VERSION_INCLUDE_DIR}")
    endif()
    INCLUDE_DIRECTORIES(BEFORE ${_CUPO_RUNTIME_INCLUDE_DIRS})
    find_library(CUPO_LIBRARY
        NAMES cupo
        PATHS ${CUPO_RUNTIME_DIR}/lib ${CUPO_RUNTIME_DIR}/build
        NO_DEFAULT_PATH)
    if(NOT CUPO_LIBRARY)
        message(FATAL_ERROR "Could not find libcupo.so under CUPO_RUNTIME_DIR=${CUPO_RUNTIME_DIR}")
    endif()
    add_library(Cupo::Runtime SHARED IMPORTED)
    set_target_properties(Cupo::Runtime PROPERTIES
        IMPORTED_LOCATION ${CUPO_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES "${_CUPO_RUNTIME_INCLUDE_DIRS}")
else()
    find_package(CupoRuntime REQUIRED CONFIG)
endif()

# Share the host-only LJ tail integrals without linking the API library to CUDA.
get_target_property(_CUPO_HOST_INCLUDE_DIRS Cupo::Runtime INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(OpenMMNvidia PRIVATE ${_CUPO_HOST_INCLUDE_DIRS})

# CUDA platform
IF(OPENMM_NVIDIA_STANDALONE)
    SET(OPENMM_BUILD_NVIDIA_CUDA_LIB ON CACHE BOOL "Build NvNonbondedForce CUDA implementation")
ELSEIF(OPENMM_BUILD_CUDA_LIB)
    SET(OPENMM_BUILD_NVIDIA_CUDA_LIB ON CACHE BOOL "Build NvNonbondedForce CUDA implementation")
ELSE()
    SET(OPENMM_BUILD_NVIDIA_CUDA_LIB OFF CACHE BOOL "Build NvNonbondedForce CUDA implementation")
ENDIF()
IF(OPENMM_BUILD_NVIDIA_CUDA_LIB)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF()

INSTALL(TARGETS ${SHARED_NVIDIA_TARGET}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib)

# Public headers install under nvopenmm/, NOT openmm/: the classes are
# NvOpenMM::NonbondedForce et al., and a header named openmm/NonbondedForce.h
# would shadow OpenMM's own on the include path.
FILE(GLOB TOP_HEADERS      openmmapi/include/nvopenmm/*.h)
FILE(GLOB INTERNAL_HEADERS openmmapi/include/nvopenmm/internal/*.h)
INSTALL_FILES(/include/nvopenmm          FILES ${TOP_HEADERS})
INSTALL_FILES(/include/nvopenmm/internal FILES ${INTERNAL_HEADERS})

# --- SWIG Python wrapper for NvLangevinMiddleIntegrator ---
# Uses conda-forge swig (4.3+) for ABI compat with OpenMM's swig_runtime_data5.
find_package(SWIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
include(UseSWIG)

set(SWIG_IFACE "${CMAKE_CURRENT_SOURCE_DIR}/swig/NvPlugin.i")
set_source_files_properties(${SWIG_IFACE} PROPERTIES
    CPLUSPLUS ON
    SWIG_MODULE_NAME nvplugin
    USE_TARGET_INCLUDE_DIRECTORIES ON)

swig_add_library(nvplugin
    TYPE SHARED
    LANGUAGE python
    OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/swig"
    OUTFILE_DIR "${CMAKE_CURRENT_BINARY_DIR}/swig"
    SOURCES ${SWIG_IFACE})

target_include_directories(nvplugin PRIVATE
    ${OPENMM_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include")
target_link_libraries(nvplugin PRIVATE
    ${SHARED_NVIDIA_TARGET}
    ${OPENMM_CORE_LIBRARY}
    Python3::Module)
# The generated wrapper compiles the same nvopenmm headers (the .i %include list
# pulls them in), so it sees the same series macros as the two libraries: a
# future #if NV_OPENMM_GE_85 member in a public header must not give
# _nvplugin.so a different class shape than the libraries were built with.
# SWIG's own parse-time preprocessor is separate from this; the first call site
# SWIG itself must see conditionally gets explicit handling then.
target_compile_definitions(nvplugin PRIVATE ${NV_OPENMM_COMPILE_DEFINITIONS})
set_target_properties(nvplugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/swig"
    OUTPUT_NAME "_nvplugin"
    PREFIX ""
    SUFFIX ".so"
    # RUNPATH so _nvplugin.so finds libOpenMMNvidia.so:
    #   $ORIGIN/lib                  — wheel layout (nv_openmm_plugin/lib/)
    #   $ORIGIN/../../build/cpp  — source tree (cpp/swig/ -> build/cpp/)
    LINK_FLAGS "-Wl,--enable-new-dtags"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN/lib;\$ORIGIN/../../build/cpp")

ENABLE_TESTING()
