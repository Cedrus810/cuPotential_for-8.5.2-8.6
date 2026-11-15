// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <openmm/OpenMMException.h>
#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

namespace NvOpenMM {
namespace clear_migration_detail {

inline std::string kernelName(cudaGraphNode_t node, CUDA_KERNEL_NODE_PARAMS* output = nullptr) {
    cudaGraphNodeType type;
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeKernel)
        return {};
    CUDA_KERNEL_NODE_PARAMS params{};
    const char* name = nullptr;
    if (cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node), &params) != CUDA_SUCCESS ||
        cuFuncGetName(&name, params.func) != CUDA_SUCCESS || !name)
        return {};
    if (output) *output = params;
    return name;
}

// Check the private kernel ABI before indexing kernelParams. Unrecognized
// signatures retain their captured dependencies and launch dimensions.
inline bool parameterSizes(const CUDA_KERNEL_NODE_PARAMS& params, std::initializer_list<size_t> sizes) {
    if (!params.kernelParams) return false;
    size_t index = 0, offset = 0, actual = 0;
    for (size_t size : sizes) {
        if (cuFuncGetParamInfo(params.func, index, &offset, &actual) != CUDA_SUCCESS || actual != size ||
            !params.kernelParams[index])
            return false;
        ++index;
    }
    return cuFuncGetParamInfo(params.func, index, &offset, &actual) == CUDA_ERROR_INVALID_VALUE;
}

inline bool disjointAllocations(const CUDA_KERNEL_NODE_PARAMS& clear,
                                const CUDA_KERNEL_NODE_PARAMS& migrate) {
    if (!parameterSizes(clear, {8, 4, 8, 4}) ||
        !parameterSizes(migrate, {4, 4, 4, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8}))
        return false;
    for (int buffer = 0; buffer < 2; ++buffer) {
        const CUdeviceptr ptr = *static_cast<const CUdeviceptr*>(clear.kernelParams[2*buffer]);
        const int count = *static_cast<const int*>(clear.kernelParams[2*buffer+1]);
        if (count < 0) return false;
        const size_t bytes = size_t(count)*sizeof(int);
        if (!bytes) continue;
        CUdeviceptr base = 0;
        size_t size = 0;
        if (cuMemGetAddressRange(&base, &size, ptr) != CUDA_SUCCESS || ptr < base ||
            ptr-base > size || bytes > size-(ptr-base))
            return false;
        // Every pointer read or written by cuda_matrix_migrate, including its
        // geometry/hash inputs, arrivals and staged wrapped positions.
        for (int arg : {3, 4, 5, 7, 8, 9, 10, 11, 12}) {
            const CUdeviceptr other = *static_cast<const CUdeviceptr*>(migrate.kernelParams[arg]);
            CUdeviceptr otherBase = 0;
            size_t otherSize = 0;
            if (cuMemGetAddressRange(&otherBase, &otherSize, other) != CUDA_SUCCESS ||
                (ptr < otherBase+otherSize && otherBase < ptr+bytes))
                return false;
        }
    }
    return true;
}

inline cudaError_t rewrite(cudaGraph_t graph, int sms, int& overlaps, int& resizes) {
    size_t count = 0;
    cudaError_t status = cudaGraphGetNodes(graph, nullptr, &count);
    if (status != cudaSuccess) return status;
    std::vector<cudaGraphNode_t> nodes(count);
    if ((status = cudaGraphGetNodes(graph, nodes.data(), &count)) != cudaSuccess) return status;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        if ((status = cudaGraphNodeGetType(node, &type)) != cudaSuccess) return status;
        if (type == cudaGraphNodeTypeGraph) {
            cudaGraph_t child = nullptr;
            if ((status = cudaGraphChildGraphNodeGetGraph(node, &child)) != cudaSuccess) return status;
            if ((status = rewrite(child, sms, overlaps, resizes)) != cudaSuccess) return status;
            continue;
        }
        CUDA_KERNEL_NODE_PARAMS migrate{}, clear{};
        if (type != cudaGraphNodeTypeKernel ||
            kernelName(node, &migrate).find("cuda_matrix_migrate") == std::string::npos)
            continue;
        size_t nparent = 0, nchild = 0;
        if ((status = cudaGraphNodeGetDependencies(node, nullptr, nullptr, &nparent)) != cudaSuccess) return status;
        if ((status = cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &nchild)) != cudaSuccess) return status;
        if (nparent != 1 || nchild != 1) continue;
        cudaGraphNode_t clearNode = nullptr, sort = nullptr;
        if ((status = cudaGraphNodeGetDependencies(node, &clearNode, nullptr, &nparent)) != cudaSuccess) return status;
        if ((status = cudaGraphNodeGetDependentNodes(node, &sort, nullptr, &nchild)) != cudaSuccess) return status;
        if (kernelName(clearNode, &clear) != "clearTwoBuffers" ||
            kernelName(sort).find("cuda_matrix_rowsort") == std::string::npos ||
            !disjointAllocations(clear, migrate))
            continue;
        size_t nsort = 0;
        if ((status = cudaGraphNodeGetDependencies(sort, nullptr, nullptr, &nsort)) != cudaSuccess) return status;
        if (nsort != 1) continue;
        size_t beforeCount = 0;
        if ((status = cudaGraphNodeGetDependencies(clearNode, nullptr, nullptr, &beforeCount)) != cudaSuccess) return status;
        std::vector<cudaGraphNode_t> before(beforeCount);
        if ((status = cudaGraphNodeGetDependencies(clearNode, before.data(), nullptr, &beforeCount)) != cudaSuccess) return status;
        for (auto predecessor : before)
            if ((status = cudaGraphAddDependencies(graph, &predecessor, &node, nullptr, 1)) != cudaSuccess) return status;
        // Migration uses private cell-list storage. Sorting still waits for BOTH
        // migration and OpenMM clearing, preserving every force producer's order.
        if ((status = cudaGraphAddDependencies(graph, &clearNode, &sort, nullptr, 1)) != cudaSuccess) return status;
        if ((status = cudaGraphRemoveDependencies(graph, &clearNode, &node, nullptr, 1)) != cudaSuccess) return status;
        ++overlaps;

        // OpenMM's int4 clear has a grid-stride loop and thread-zero scalar tail.
        // Bound its otherwise overprovisioned grid, without removing either clear.
        if (clear.blockDimX && clear.blockDimY == 1 && clear.blockDimZ == 1 &&
            clear.gridDimY == 1 && clear.gridDimZ == 1) {
            const int size1 = *static_cast<const int*>(clear.kernelParams[1]);
            const int size2 = *static_cast<const int*>(clear.kernelParams[3]);
            unsigned blocks = std::max(1u, (unsigned(std::max(size1, size2))/4+clear.blockDimX-1)/clear.blockDimX);
            blocks = std::min(blocks, unsigned(std::max(1, sms/2)));
            if (blocks < clear.gridDimX) {
                clear.gridDimX = blocks;
                if (cuGraphKernelNodeSetParams(reinterpret_cast<CUgraphNode>(clearNode), &clear) != CUDA_SUCCESS)
                    return cudaErrorInvalidValue;
                ++resizes;
            }
        }
    }
    return cudaSuccess;
}
} // namespace clear_migration_detail

// Called only for composed multi-step graphs. Rewriting an independently
// launched single-step graph regressed full-step time; leave those untouched.
inline cudaError_t nvOptimizeBatchedForceClear(cudaGraph_t graph, int& overlaps, int& resizes) {
    const char* setting = std::getenv("CUPO_PLUGIN_CLEAR_PARALLEL");
    const std::string mode = setting ? setting : "batched";
    if (mode == "off") return cudaSuccess;
    if (mode != "batched")
        throw OpenMM::OpenMMException("CUPO_PLUGIN_CLEAR_PARALLEL must be batched or off");
    int dev = 0, sms = 0;
    cudaError_t status = cudaGetDevice(&dev);
    if (status != cudaSuccess) return status;
    if ((status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)) != cudaSuccess) return status;
    return clear_migration_detail::rewrite(graph, sms, overlaps, resizes);
}
} // namespace NvOpenMM
