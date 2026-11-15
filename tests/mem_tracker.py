# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steady-state GPU memory leak detector.

Runs an operation in a loop. The first `warmup` ticks absorb one-time costs
(module loads, JIT). After warmup, asserts memory is stable.

Usage:
    with MemoryTracker(warmup=2) as mt:
        for i in range(10):
            # ... create workspace, compute, release ...
            mt.tick()
"""
import gc
import subprocess

MB = 1024 * 1024


def _readings():
    """The device's used and free memory (cudaMemGetInfo), and CuPy's pool."""
    import cupy as cp
    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.runtime.memGetInfo()
    return {"device_used": total - free, "device_free": free,
            "pool_used": pool.used_bytes(), "pool_free": pool.free_bytes(),
            "pool_total": pool.total_bytes()}


class MemoryTracker:
    """Assert that repeated GPU operations do not leak memory.

    What it reads is the GPU memory in use: the device's used memory
    (cudaMemGetInfo), which a leak through any allocator shows in, cupo's own
    cudaMalloc among them, less the free memory CuPy's pool keeps. After
    free_all_blocks() that is the free space of chunks the pool cannot return
    because something still lives in them. Allocations carved from that space
    leave the device's number unchanged. That hid a leak from the device-wide
    reading alone: an earlier test left a 5.5 GB chunk free in the pool, the
    leaking loop's first array, made before the first tick, was carved from it,
    so free_all_blocks() could no longer return the chunk, and the next arrays
    came out of it too. __enter__ also returns every free chunk first, so the
    loop starts from a pool that holds only live arrays.

    On exit it prints its readings, one line per tick. pytest shows that output
    only for a failing test.

    Parameters
    ----------
    warmup : int
        Number of initial ticks to skip (absorbs JIT / module loads).
    threshold : int
        Maximum allowed byte growth across post-warmup ticks (default 0).
    """

    def __init__(self, warmup=2, threshold=0):
        self.warmup = warmup
        self.threshold = threshold
        self._tick_count = 0
        self._measurements = []
        self._table = []

    def __enter__(self):
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        self._at_enter = _readings()
        return self

    def tick(self):
        import cupy as cp
        pool = cp.get_default_memory_pool()
        cp.cuda.runtime.deviceSynchronize()
        pool.free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        r = _readings()
        used = r["device_used"] - r["pool_free"]

        self._tick_count += 1
        self._table.append((self._tick_count, used, r))
        if self._tick_count > self.warmup:
            self._measurements.append(used)

    def report(self):
        """The readings as text: the pool at __enter__, then one line per tick."""
        def row(r):
            return ("device used %9.1f free %9.1f | pool used %8.1f free %8.1f total %8.1f"
                    % tuple(r[k] / MB for k in ("device_used", "device_free", "pool_used",
                                                "pool_free", "pool_total")))
        lines = ["MemoryTracker readings (MB), warmup %d, threshold %.1f:"
                 % (self.warmup, self.threshold / MB),
                 "  enter              " + row(self._at_enter)]
        for n, used, r in self._table:
            lines.append("  tick %2d%s in use %9.1f | %s"
                         % (n, " (warmup)" if n <= self.warmup else "         ", used / MB, row(r)))
        return "\n".join(lines)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.report())
        if exc_type is not None:
            return False
        # Hard-fail: tick() must have been called more than `warmup` times.
        # This catches loops that exit early or have too few iterations --
        # without this guard the tracker silently passes with no data.
        if self._tick_count <= self.warmup:
            raise RuntimeError(
                f"MemoryTracker: tick() called {self._tick_count} times, "
                f"need more than {self.warmup} (warmup) to detect leaks. "
                f"Increase iterations or remove the tracker.")
        growth = self._measurements[-1] - self._measurements[0]
        if growth > self.threshold:
            # Who else is on the GPU, the same query as the session's exclusivity check.
            try:
                apps = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader"], capture_output=True, text=True,
                    timeout=10).stdout.strip() or "none"
            except (OSError, subprocess.TimeoutExpired) as e:
                apps = "nvidia-smi failed: %s" % e
            raise AssertionError(
                f"GPU memory leak: {growth / 1024**2:.1f} MB growth over "
                f"{len(self._measurements)} steady-state ticks "
                f"(threshold {self.threshold / 1024**2:.1f} MB)\n"
                f"  measurements (MB): "
                f"{[round(m / 1024**2, 1) for m in self._measurements]}\n"
                f"  GPU processes: {apps}")
        return False
