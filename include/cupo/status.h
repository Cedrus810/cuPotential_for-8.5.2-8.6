// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include "cupo/visibility.h"

/** C API status codes. C++ exceptions do not cross the C API boundary.
 * For asynchronous calls, success means work was enqueued;
 * check CUDA synchronization for device execution errors. */
typedef enum cupo_status {
    CUPO_SUCCESS = 0,          // The operation succeeded.
    CUPO_INVALID_PLAN = 1,     // A required plan handle is NULL.
    CUPO_ALLOC_FAILED = 2,     // Host or device allocation failed.
    CUPO_INVALID_TYPE = 3,     // Unknown real-grid element type.
    CUPO_INVALID_VALUE = 4,    // Invalid pointer, alignment, overlap, encoding, or buffer size.
    CUPO_INTERNAL_ERROR = 5,   // Unexpected internal failure.
    CUPO_ERROR = CUPO_INTERNAL_ERROR, // Generic error; see cupo_last_error() for details.
    CUPO_EXEC_FAILED = 6,      // CUDA kernel launch or execution failed.
    CUPO_SETUP_FAILED = 7,     // CUDA or FFT initialization failed.
    CUPO_INVALID_SIZE = 8,     // Unsupported real-grid length or support radius.
    CUPO_INVALID_DEVICE = 11,  // Wrong or unavailable CUDA device.
    CUPO_NO_WORKSPACE = 13,    // Execution requires a bound workspace.
    CUPO_NOT_SUPPORTED = 16   // Unsupported GPU architecture or CUDA operation.
} cupo_status;

#ifdef __cplusplus
extern "C" {
#endif
/** Read the calling thread's last runtime API error.
 * APIs documented as using this diagnostic replace it on failure and clear it
 * on success. Reading it does not clear it. Asynchronous device errors are
 * reported by CUDA synchronization instead.
 * @return Borrowed host NUL-terminated string, empty initially or after success.
 * Valid until the next updating API call on this thread; at most 4095 bytes.
 */
CUPO_API const char* cupo_last_error(void);
#ifdef __cplusplus
}
#endif
