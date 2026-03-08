"""I/O Benchmark for GaussianStreamer RAW playback.

Simulates the plugin's per-frame loading patterns and measures:
1. Metadata parsing: per-frame JSON vs batch pre-load
2. File I/O: synchronous sequential vs async (threaded) reads
3. Memory allocation: create/destroy vs reuse pattern

Uses actual RAW output files for realistic measurements.
"""

import json
import os
import sys
import time
import statistics
import concurrent.futures
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────

RAW_FOLDER = r"C:\Users\tommy\Desktop\w6\medias\video\aurora-timelapse_raw_top50"
NUM_FRAMES_TO_TEST = 100  # Test first N frames
WARM_UP_FRAMES = 5        # Warm-up before measuring

BIN_FILES = [
    "position.bin", "rotation.bin", "scaleOpacity.bin",
    "sh_0.bin", "sh_1.bin", "sh_2.bin", "sh_3.bin",
    "sh_4.bin", "sh_5.bin", "sh_6.bin", "sh_7.bin",
    "sh_8.bin", "sh_9.bin", "sh_10.bin", "sh_11.bin",
]


def get_frame_folders(raw_folder: str, count: int) -> list[str]:
    """Get sorted frame folder paths."""
    folders = []
    for i in range(count):
        folder = os.path.join(raw_folder, f"frame_{i:04d}")
        if os.path.isdir(folder):
            folders.append(folder)
    return folders


# ═══════════════════════════════════════════════════════════════
# Benchmark 1: Metadata Parsing
# ═══════════════════════════════════════════════════════════════

def bench_metadata_per_frame(frame_folders: list[str]) -> dict:
    """Current approach: parse metadata.json on each frame load."""
    times = []
    for folder in frame_folders:
        meta_path = os.path.join(folder, "metadata.json")
        t0 = time.perf_counter_ns()
        with open(meta_path) as f:
            metadata = json.load(f)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)  # ms
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "max_ms": max(times),
        "p99_ms": sorted(times)[int(len(times) * 0.99)],
        "total_ms": sum(times),
    }


def bench_metadata_batch(frame_folders: list[str]) -> dict:
    """Optimized: batch pre-load all metadata at once."""
    t0 = time.perf_counter_ns()
    all_metadata = {}
    for i, folder in enumerate(frame_folders):
        meta_path = os.path.join(folder, "metadata.json")
        with open(meta_path) as f:
            all_metadata[i] = json.load(f)
    t_load = (time.perf_counter_ns() - t0) / 1e6

    # Simulate per-frame access (dict lookup)
    times = []
    for i in range(len(frame_folders)):
        t0 = time.perf_counter_ns()
        metadata = all_metadata[i]
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)

    return {
        "preload_total_ms": t_load,
        "per_access_mean_us": statistics.mean(times) * 1000,  # microseconds
        "per_access_max_us": max(times) * 1000,
    }


# ═══════════════════════════════════════════════════════════════
# Benchmark 2: File I/O - Synchronous vs Async
# ═══════════════════════════════════════════════════════════════

def load_frame_sync(frame_folder: str) -> tuple[float, int]:
    """Synchronous: load all bin files sequentially (current UE approach)."""
    total_bytes = 0
    t0 = time.perf_counter_ns()
    for bin_file in BIN_FILES:
        path = os.path.join(frame_folder, bin_file)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read()
            total_bytes += len(data)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, total_bytes


def load_frame_async_threaded(frame_folder: str, executor) -> tuple[float, int]:
    """Async: load all bin files in parallel threads."""
    total_bytes = 0

    def read_file(path):
        with open(path, "rb") as f:
            return f.read()

    paths = [
        os.path.join(frame_folder, bf) for bf in BIN_FILES
        if os.path.isfile(os.path.join(frame_folder, bf))
    ]

    t0 = time.perf_counter_ns()
    futures = [executor.submit(read_file, p) for p in paths]
    for fut in concurrent.futures.as_completed(futures):
        data = fut.result()
        total_bytes += len(data)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, total_bytes


def bench_sync_io(frame_folders: list[str]) -> dict:
    """Benchmark synchronous sequential file reads."""
    # Warm up
    for folder in frame_folders[:WARM_UP_FRAMES]:
        load_frame_sync(folder)

    times = []
    total_bytes = 0
    for folder in frame_folders:
        ms, nbytes = load_frame_sync(folder)
        times.append(ms)
        total_bytes = nbytes  # Same per frame

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "max_ms": max(times),
        "min_ms": min(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
        "p99_ms": sorted(times)[int(len(times) * 0.99)],
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "bytes_per_frame": total_bytes,
        "throughput_MBps": total_bytes / (statistics.mean(times) / 1000) / 1e6,
    }


def bench_async_io(frame_folders: list[str]) -> dict:
    """Benchmark threaded parallel file reads."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # Warm up
    for folder in frame_folders[:WARM_UP_FRAMES]:
        load_frame_async_threaded(folder, executor)

    times = []
    total_bytes = 0
    for folder in frame_folders:
        ms, nbytes = load_frame_async_threaded(folder, executor)
        times.append(ms)
        total_bytes = nbytes

    executor.shutdown(wait=False)

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "max_ms": max(times),
        "min_ms": min(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
        "p99_ms": sorted(times)[int(len(times) * 0.99)],
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "bytes_per_frame": total_bytes,
        "throughput_MBps": total_bytes / (statistics.mean(times) / 1000) / 1e6,
    }


# ═══════════════════════════════════════════════════════════════
# Benchmark 3: Prefetch Pipeline Simulation
# ═══════════════════════════════════════════════════════════════

def bench_prefetch_pipeline(frame_folders: list[str]) -> dict:
    """Simulate async prefetch: while frame N renders, load frame N+1 in background."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def read_all_bins(folder):
        buffers = []
        for bf in BIN_FILES:
            path = os.path.join(folder, bf)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    buffers.append(f.read())
        return buffers

    # Warm up
    for folder in frame_folders[:WARM_UP_FRAMES]:
        read_all_bins(folder)

    # Simulate pipeline: main thread "processes" current frame
    # while background thread loads next frame
    frame_times = []  # Time main thread is blocked per frame

    # Start first frame load
    future = executor.submit(read_all_bins, frame_folders[0])
    current_data = future.result()

    for i in range(1, len(frame_folders)):
        t0 = time.perf_counter_ns()

        # Submit next frame load (background)
        future = executor.submit(read_all_bins, frame_folders[i])

        # Simulate "processing" current frame (memcpy equivalent ~2ms)
        # In real UE this would be texture creation
        _ = bytearray(len(current_data[0]))  # Simulate allocation + copy

        # Wait for next frame data
        current_data = future.result()

        t1 = time.perf_counter_ns()
        frame_times.append((t1 - t0) / 1e6)

    executor.shutdown(wait=False)

    return {
        "mean_ms": statistics.mean(frame_times),
        "median_ms": statistics.median(frame_times),
        "max_ms": max(frame_times),
        "p95_ms": sorted(frame_times)[int(len(frame_times) * 0.95)],
        "p99_ms": sorted(frame_times)[int(len(frame_times) * 0.99)],
    }


# ═══════════════════════════════════════════════════════════════
# Benchmark 4: Memory Allocation Pattern
# ═══════════════════════════════════════════════════════════════

def bench_alloc_create_destroy(frame_folders: list[str]) -> dict:
    """Current: allocate new buffers per frame, then discard."""
    times = []
    for folder in frame_folders:
        t0 = time.perf_counter_ns()
        buffers = []
        for bf in BIN_FILES:
            path = os.path.join(folder, bf)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                buf = bytearray(size)  # Simulate UTexture2D::CreateTransient
                buffers.append(buf)
        # Simulate destroy
        del buffers
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "max_ms": max(times),
    }


def bench_alloc_ring_buffer(frame_folders: list[str]) -> dict:
    """Optimized: pre-allocate ring buffer, reuse."""
    # Pre-allocate once
    sample_folder = frame_folders[0]
    ring_buffers = []
    for bf in BIN_FILES:
        path = os.path.join(sample_folder, bf)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            ring_buffers.append(bytearray(size))

    times = []
    for folder in frame_folders:
        t0 = time.perf_counter_ns()
        for i, bf in enumerate(BIN_FILES):
            path = os.path.join(folder, bf)
            if os.path.isfile(path) and i < len(ring_buffers):
                # Reuse existing buffer (simulate RHIUpdateTexture2D)
                # Just zero it to simulate overwrite
                ring_buffers[i][:16] = b'\x00' * 16  # Minimal touch
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "max_ms": max(times),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GaussianStreamer I/O Benchmark")
    print("=" * 70)

    if not os.path.isdir(RAW_FOLDER):
        print(f"ERROR: RAW folder not found: {RAW_FOLDER}")
        sys.exit(1)

    frame_folders = get_frame_folders(RAW_FOLDER, NUM_FRAMES_TO_TEST)
    print(f"RAW folder: {RAW_FOLDER}")
    print(f"Frames to test: {len(frame_folders)}")

    # Get frame size info
    sample_frame = frame_folders[0]
    frame_size = sum(
        os.path.getsize(os.path.join(sample_frame, bf))
        for bf in BIN_FILES
        if os.path.isfile(os.path.join(sample_frame, bf))
    )
    bin_count = sum(
        1 for bf in BIN_FILES
        if os.path.isfile(os.path.join(sample_frame, bf))
    )
    print(f"Bin files per frame: {bin_count}")
    print(f"Frame size: {frame_size / 1e6:.1f} MB")
    print(f"Target: 30 FPS = {1000/30:.1f} ms/frame budget")
    print()

    # ─── Benchmark 1: Metadata ───
    print("-" * 70)
    print("BENCHMARK 1: Metadata Parsing")
    print("-" * 70)

    print("\n[A] Per-frame JSON parse (current UE approach):")
    meta_per = bench_metadata_per_frame(frame_folders)
    print(f"    Mean: {meta_per['mean_ms']:.3f} ms/frame")
    print(f"    Median: {meta_per['median_ms']:.3f} ms/frame")
    print(f"    Max: {meta_per['max_ms']:.3f} ms")
    print(f"    P99: {meta_per['p99_ms']:.3f} ms")
    print(f"    Total for {len(frame_folders)} frames: {meta_per['total_ms']:.1f} ms")

    print("\n[B] Batch pre-load (optimized):")
    meta_batch = bench_metadata_batch(frame_folders)
    print(f"    Pre-load all: {meta_batch['preload_total_ms']:.1f} ms (one-time cost)")
    print(f"    Per-access: {meta_batch['per_access_mean_us']:.3f} us (microseconds)")
    print(f"    Speedup: {meta_per['mean_ms'] * 1000 / max(meta_batch['per_access_mean_us'], 0.001):.0f}x per frame")

    # ─── Benchmark 2: File I/O ───
    print()
    print("-" * 70)
    print("BENCHMARK 2: File I/O (per frame)")
    print("-" * 70)

    print("\n[A] Synchronous sequential reads (current UE approach):")
    sync = bench_sync_io(frame_folders)
    print(f"    Mean: {sync['mean_ms']:.1f} ms/frame")
    print(f"    Median: {sync['median_ms']:.1f} ms/frame")
    print(f"    Min: {sync['min_ms']:.1f} ms | Max: {sync['max_ms']:.1f} ms")
    print(f"    P95: {sync['p95_ms']:.1f} ms | P99: {sync['p99_ms']:.1f} ms")
    print(f"    Stdev: {sync['stdev_ms']:.1f} ms")
    print(f"    Throughput: {sync['throughput_MBps']:.0f} MB/s")
    fits_budget = sync['mean_ms'] < 33.3
    print(f"    Fits 30fps budget? {'YES' if fits_budget else 'NO'} ({sync['mean_ms']:.1f} vs 33.3 ms)")

    print("\n[B] Threaded parallel reads (4 workers):")
    async_res = bench_async_io(frame_folders)
    print(f"    Mean: {async_res['mean_ms']:.1f} ms/frame")
    print(f"    Median: {async_res['median_ms']:.1f} ms/frame")
    print(f"    Min: {async_res['min_ms']:.1f} ms | Max: {async_res['max_ms']:.1f} ms")
    print(f"    P95: {async_res['p95_ms']:.1f} ms | P99: {async_res['p99_ms']:.1f} ms")
    print(f"    Stdev: {async_res['stdev_ms']:.1f} ms")
    print(f"    Throughput: {async_res['throughput_MBps']:.0f} MB/s")
    speedup_io = sync['mean_ms'] / max(async_res['mean_ms'], 0.01)
    print(f"    Speedup vs sync: {speedup_io:.1f}x")

    # ─── Benchmark 3: Prefetch Pipeline ───
    print()
    print("-" * 70)
    print("BENCHMARK 3: Prefetch Pipeline (load N+1 while processing N)")
    print("-" * 70)

    prefetch = bench_prefetch_pipeline(frame_folders)
    print(f"    Mean: {prefetch['mean_ms']:.1f} ms/frame (main thread blocked)")
    print(f"    Median: {prefetch['median_ms']:.1f} ms/frame")
    print(f"    Max: {prefetch['max_ms']:.1f} ms")
    print(f"    P95: {prefetch['p95_ms']:.1f} ms | P99: {prefetch['p99_ms']:.1f} ms")
    prefetch_speedup = sync['mean_ms'] / max(prefetch['mean_ms'], 0.01)
    print(f"    Speedup vs sync baseline: {prefetch_speedup:.1f}x")

    # ─── Benchmark 4: Memory Allocation ───
    print()
    print("-" * 70)
    print("BENCHMARK 4: Memory Allocation Pattern")
    print("-" * 70)

    print("\n[A] Create/destroy per frame (current UE approach):")
    alloc_cd = bench_alloc_create_destroy(frame_folders)
    print(f"    Mean: {alloc_cd['mean_ms']:.3f} ms/frame")
    print(f"    Max: {alloc_cd['max_ms']:.3f} ms")

    print("\n[B] Ring buffer reuse (optimized):")
    alloc_ring = bench_alloc_ring_buffer(frame_folders)
    print(f"    Mean: {alloc_ring['mean_ms']:.3f} ms/frame")
    print(f"    Max: {alloc_ring['max_ms']:.3f} ms")
    alloc_speedup = alloc_cd['mean_ms'] / max(alloc_ring['mean_ms'], 0.0001)
    print(f"    Speedup: {alloc_speedup:.0f}x")

    # ─── Summary ───
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Frame budget at 30fps: 33.3 ms")
    print()

    current_total = sync['mean_ms'] + meta_per['mean_ms'] + alloc_cd['mean_ms']
    optimized_io = prefetch['mean_ms']
    optimized_meta = meta_batch['per_access_mean_us'] / 1000  # us -> ms
    optimized_alloc = alloc_ring['mean_ms']
    optimized_total = optimized_io + optimized_meta + optimized_alloc

    print(f"  {'Component':<25} {'Current':>10} {'Optimized':>10} {'Saved':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Metadata parsing':<25} {meta_per['mean_ms']:>9.2f}ms {optimized_meta:>9.3f}ms {meta_per['mean_ms'] - optimized_meta:>9.2f}ms")
    print(f"  {'File I/O':<25} {sync['mean_ms']:>9.1f}ms {optimized_io:>9.1f}ms {sync['mean_ms'] - optimized_io:>9.1f}ms")
    print(f"  {'Memory alloc':<25} {alloc_cd['mean_ms']:>9.2f}ms {optimized_alloc:>9.3f}ms {alloc_cd['mean_ms'] - optimized_alloc:>9.2f}ms")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<25} {current_total:>9.1f}ms {optimized_total:>9.1f}ms {current_total - optimized_total:>9.1f}ms")
    print()
    improvement_pct = (1 - optimized_total / current_total) * 100
    print(f"  Overall improvement: {improvement_pct:.0f}%")
    print(f"  Fits 30fps budget? Current={'YES' if current_total < 33.3 else 'NO'}, Optimized={'YES' if optimized_total < 33.3 else 'NO'}")
    print()
    print("  NOTE: This benchmark measures CPU-side I/O only.")
    print("  UE texture upload (CPU→GPU) adds additional overhead")
    print("  that cannot be measured outside the engine.")


if __name__ == "__main__":
    main()
