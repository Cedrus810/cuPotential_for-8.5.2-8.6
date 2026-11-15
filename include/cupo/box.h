// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

/** Periodic box layout: FP32 [a_x, b_y, c_z, b_x, c_x, c_y] in nm.
 * Lattice vectors are a=(a_x,0,0), b=(b_x,b_y,0), c=(c_x,c_y,c_z).
 * Host APIs and the helpers below require all six values. Orthorhombic boxes
 * use zero for the last three. Device APIs configured for orthorhombic geometry
 * accept the first three values; triclinic device APIs require all six.
 */

#define CUPO_BOX_AX 0
#define CUPO_BOX_BY 1
#define CUPO_BOX_CZ 2
#define CUPO_BOX_BX 3
#define CUPO_BOX_CX 4
#define CUPO_BOX_CY 5

#define CUPO_BOX_ORTHORHOMBIC_FLOATS 3
#define CUPO_BOX_FLOATS 6

#ifdef __cplusplus
extern "C" {
#endif

/** Pack lower-triangular lattice vectors into the shared FP32 box layout.
 * @param[in] a_x X component of vector a, in nm.
 * @param[in] b_x X component of vector b, in nm.
 * @param[in] b_y Y component of vector b, in nm.
 * @param[in] c_x X component of vector c, in nm.
 * @param[in] c_y Y component of vector c, in nm.
 * @param[in] c_z Z component of vector c, in nm.
 * @param[out] out Host FP32 array of CUPO_BOX_FLOATS elements.
 */
static inline void cupo_box_from_vectors(double a_x, double b_x, double b_y,
                                         double c_x, double c_y, double c_z,
                                         float out[CUPO_BOX_FLOATS])
{
    out[CUPO_BOX_AX] = (float)a_x;
    out[CUPO_BOX_BY] = (float)b_y;
    out[CUPO_BOX_CZ] = (float)c_z;
    out[CUPO_BOX_BX] = (float)b_x;
    out[CUPO_BOX_CX] = (float)c_x;
    out[CUPO_BOX_CY] = (float)c_y;
}

/** Check whether the box is orthorhombic.
 * @param[in] box Host FP32 array in the six-value box layout.
 * @return 1 if b_x, c_x and c_y are exactly zero; otherwise 0.
 */
static inline int cupo_box_is_orthorhombic(const float box[CUPO_BOX_FLOATS])
{
    return box[CUPO_BOX_BX] == 0.0f && box[CUPO_BOX_CX] == 0.0f && box[CUPO_BOX_CY] == 0.0f;
}

/** Check whether the box is in reduced lower-triangular form.
 * @param[in] box Host FP32 array in the six-value box layout.
 * @return 1 if the diagonal is positive, |b_x|<=a_x/2, |c_x|<=a_x/2 and
 * |c_y|<=b_y/2; otherwise 0.
 */
static inline int cupo_box_is_reduced(const float box[CUPO_BOX_FLOATS])
{
    const float ax = box[CUPO_BOX_AX], by = box[CUPO_BOX_BY], cz = box[CUPO_BOX_CZ];
    const float bx = box[CUPO_BOX_BX], cx = box[CUPO_BOX_CX], cy = box[CUPO_BOX_CY];
    return ax > 0.0f && by > 0.0f && cz > 0.0f &&
           2.0f * (bx < 0.0f ? -bx : bx) <= ax &&
           2.0f * (cx < 0.0f ? -cx : cx) <= ax &&
           2.0f * (cy < 0.0f ? -cy : cy) <= by;
}

/** Compute the volume of a lower-triangular box.
 * @param[in] box Host FP32 array in the six-value box layout, in nm.
 * @return FP64 volume a_x*b_y*c_z, in nm^3.
 */
static inline double cupo_box_volume(const float box[CUPO_BOX_FLOATS])
{
    return (double)box[CUPO_BOX_AX] * (double)box[CUPO_BOX_BY] * (double)box[CUPO_BOX_CZ];
}

#ifdef __cplusplus
}
#endif
