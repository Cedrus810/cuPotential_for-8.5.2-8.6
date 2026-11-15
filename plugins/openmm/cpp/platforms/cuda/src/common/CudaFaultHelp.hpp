// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NV_CUDA_FAULT_HELP_H_
#define NV_CUDA_FAULT_HELP_H_

#include <cuda_runtime.h>

#include <string>

namespace NvOpenMM {

/**
 * Whether a CUDA error means a kernel died on the device: a __trap(), which is
 * how cupo's contract checks end, or a memory fault. After one, every later CUDA
 * call fails, so the call that reports it is usually not where it happened.
 *
 * The same list as cupo's own device-fault check. Kept here rather
 * than asked of cupo, so the plugin needs nothing from cupo's C API to explain
 * its own errors.
 */
inline bool nvIsDeviceFault(cudaError_t err)
{
    switch (err) {
        case cudaErrorLaunchFailure:      // what __trap() becomes
        case cudaErrorIllegalInstruction:
        case cudaErrorIllegalAddress:
        case cudaErrorMisalignedAddress:
        case cudaErrorHardwareStackError:
        case cudaErrorInvalidPc:
        case cudaErrorAssert:
            return true;
        default:
            return false;
    }
}

/**
 * What the plugin adds to a CUDA error that means a kernel died on the device:
 * how to find that kernel, and how to make the failure happen at the same step
 * every run. Empty for any other error.
 *
 * Written for a plugin user. The site in the error is often OpenMM's, or the
 * plugin's own check, because a fault surfaces at whatever CUDA call syncs next;
 * and the way to make a run reproducible is this plugin's two environment
 * variables, with no change to the script.
 */
inline std::string nvDeviceFaultHelp(cudaError_t err)
{
    if (!nvIsDeviceFault(err))
        return "";
    return
        "\n"
        "A kernel died on the device (one of cupo's contract checks, or a memory "
        "fault). The call above is where it was noticed, at the next CUDA call to "
        "sync, and may be OpenMM's or the plugin's rather than the one that "
        "failed.\n"
        "\n"
        "To find the kernel, rerun with a GPU core dump:\n"
        "\n"
        "    CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 "
        "CUDA_COREDUMP_FILE=/tmp/cupo_%h_%p.nvcudmp python3 <your script>\n"
        "    cuda-gdb -batch -ex 'target cudacore /tmp/cupo_<host>_<pid>.nvcudmp' "
        "-ex 'info cuda kernels' -ex 'info cuda threads'\n"
        "\n"
        "'info cuda kernels' names the kernel, 'info cuda threads' the block and "
        "thread. There are no locals at -O3, but the thread index usually names "
        "the bad element: for cupo's per-atom kernels, the atom.\n"
        "\n"
        "Or run the script under compute-sanitizer: slower, no setup, and it also "
        "names the launch and an offset into the kernel ('+0x...'; the function "
        "named with it may be an unrelated math helper).\n"
        "\n"
        "The released libraries have no source lines. For file and line, report "
        "the kernel, the offset and the plugin version.\n"
        "\n"
        "If the failure is intermittent, make every run fail at the same step "
        "first: CUPO_OPENMM_PLUGIN_RNG_SEED=42 "
        "CUPO_OPENMM_STOCK_FORCE_DETERMINISTIC_FORCES=1, with no change to the "
        "script.";
}

}  // namespace NvOpenMM

#endif  // NV_CUDA_FAULT_HELP_H_
