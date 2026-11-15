// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NVOPENMM_COMMON_CUH
#define NVOPENMM_COMMON_CUH

// -----------------------------------------------------------------------------
// Shared types + vector operators for the AOT Nvidia LangevinMiddle kernels.
//
// The templated kernel bodies (Constraints.cuh) are plain CUDA and use these
// two things directly: mm_long (the fixed-point force accumulator) and the small
// set of float3/double3/float4 operators CUDA does not provide out of the box.
// The per-precision compute type (float / float-float df64) and its nvsqrt/
// nvrsqrt/nvrecip/nvmk* overloads live in Constraints.cuh.
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>

namespace NvOpenMM {

typedef long long mm_long;   // fixed-point force accumulator type (matches OpenMM)

// The fixed-point scale that accumulator is encoded with. This is an OpenMM ABI
// constant rather than a choice of ours: everything that writes the force buffer
// encodes with it -- OpenMM's own kernels (`(mixed) 0x100000000`, e.g.
// platforms/common/src/kernels/langevinMiddle.cc:9) and cupo's, which encodes with
// the same 2^32 scale -- so Part1 has to DECODE with the same exponent or every
// force is off by a power of two.
//
// Stated as the exponent so the value and its documentation cannot drift apart.
// 2^32 is exactly representable in float and double alike, so the decode divide is
// exact and costs the round trip nothing beyond the encode's own truncation.
static constexpr int   MM_FORCE_SCALE_BITS = 32;
static constexpr float MM_FORCE_SCALE = (float) (1ull << MM_FORCE_SCALE_BITS);   // 2^32

} // namespace NvOpenMM

// Vector operators CUDA does not provide. Defined for the concrete types both
// precision namespaces use: float3/double3 (mixed3) scalar multiply/scale, and
// float4 (the center-of-mass partials) accumulate.
//
// These stay in the GLOBAL namespace deliberately -- do not move them into
// NvOpenMM with the rest of this header. float3/float4 are CUDA builtins declared
// at global scope, so argument-dependent lookup for `a + b` searches the global
// namespace and nothing else. cub::BlockReduce<float4>::Sum() reaches operator+
// that way, from inside cuda::std::plus<> where no using-declaration of ours is in
// scope, so a namespaced operator+ is simply invisible to it. (Observed: moving
// them fails the centre-of-mass kernels with "no instance of cuda::std::plus<void>::
// operator() matches the argument list".)
static __device__ __forceinline__ float3  operator*(float3 a, float s)  { return make_float3(a.x*s, a.y*s, a.z*s); }
static __device__ __forceinline__ double3 operator*(double3 a, double s) { return make_double3(a.x*s, a.y*s, a.z*s); }
static __device__ __forceinline__ float3&  operator*=(float3& a, float s)  { a.x*=s; a.y*=s; a.z*=s; return a; }
static __device__ __forceinline__ double3& operator*=(double3& a, double s) { a.x*=s; a.y*=s; a.z*=s; return a; }
static __device__ __forceinline__ float4&  operator+=(float4& a, float4 b)  { a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; return a; }
static __device__ __forceinline__ double4& operator+=(double4& a, double4 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w; return a; }
static __device__ __forceinline__ float4   operator+(float4 a, float4 b) { return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }

#endif // NVOPENMM_COMMON_CUH
