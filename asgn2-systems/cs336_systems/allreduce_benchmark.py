import os
import csv
import timeit
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CSV_PATH = Path(__file__).resolve().parent / "results" / "allreduce.csv"

FIELDS = [
    "world_size",
    "size_mb",
    "median_ms",
    "mean_ms",
    "std_ms",
    "throughput_gbps",
]

def setup(
    rank: int, 
    world_size: int, 
    backend: str
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if backend == "nccl":
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend, 
        rank=rank, 
        world_size=world_size, 
    )

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def distributed_benchmark(
    rank,
    world_size,
    sizes_mb,
    device,
    warmup=5,
    iters=20,
):
    results = []

    for size_mb in sizes_mb:

        # Create Tensor
        numel = size_mb * 1_000_000 // 4
        data = torch.zeros(
            numel,
            dtype=torch.float32,
            device=device,
        )

        # Warm-up
        for _ in range(warmup):
            dist.all_reduce(data, op=dist.ReduceOp.SUM)

        sync(device)

        # Benchmark
        times = []

        for _ in range(iters):
            if device.type == "cuda":
                dist.barrier(device_ids=[device.index])
            else:
                dist.barrier()

            sync(device)
            start = timeit.default_timer()

            dist.all_reduce(
                data,
                op=dist.ReduceOp.SUM,
                async_op=False,
            )

            sync(device)
            elapsed = (timeit.default_timer() - start) * 1000
            times.append(elapsed)

        # Gather rank timings
        all_times = [None] * world_size
        dist.all_gather_object(all_times, times)

        if rank == 0:
            slowest_times = [
                max(all_times[r][i] for r in range(world_size))
                for i in range(iters)
            ]

            median_ms = statistics.median(slowest_times)

            results.append({
                "world_size": world_size,
                "size_mb": size_mb,
                "median_ms": median_ms,
                "mean_ms": statistics.mean(slowest_times),
                "std_ms": statistics.stdev(slowest_times),
                "throughput_gbps": size_mb / median_ms,
            })

        del data

    return results

def print_results(results):
    if not results:
        return

    world_size = results[0]["world_size"]

    print(f"\n{'=' * 74}")
    print(f"World Size: {world_size}")
    print(f"{'=' * 74}")

    print(
        f"{'Size(MB)':>10} "
        f"{'Median(ms)':>12} "
        f"{'Mean(ms)':>12} "
        f"{'Std(ms)':>10} "
        f"{'GB/s':>10}"
    )

    for r in results:
        print(
            f"{r['size_mb']:>10} "
            f"{r['median_ms']:>12.3f} "
            f"{r['mean_ms']:>12.3f} "
            f"{r['std_ms']:>10.3f} "
            f"{r['throughput_gbps']:>10.3f}"
        )

    # Basic analysis
    if len(results) > 1:
        first = results[0]
        last = results[-1]

        size_ratio = last["size_mb"] / first["size_mb"]
        latency_ratio = last["median_ms"] / first["median_ms"]
        throughput_ratio = (
            last["throughput_gbps"] / first["throughput_gbps"]
        )

        print("\nAnalysis:")
        print(f"  Data size: {size_ratio:.1f}x")
        print(f"  Latency: {latency_ratio:.2f}x")
        print(f"  Throughput: {throughput_ratio:.2f}x")

def save_results(results, csv_path):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerows(results)

def distributed_train(
    rank,
    world_size,
    sizes_mb,
    backend,
):
    setup(rank, world_size, backend)

    try:
        if backend == "nccl":
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device("cpu")

        print(
            f"Rank {rank}/{world_size} -> {device}",
            flush=True,
        )
        
        results = distributed_benchmark(
            rank=rank,
            world_size=world_size,
            sizes_mb=sizes_mb,
            device=device,
        )

        if rank == 0:
            print_results(results)
            save_results(results, CSV_PATH)

    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZES = [2, 4, 6]
    SIZES_MB = [1, 10, 100, 1000]
    BACKEND = "gloo"

    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")

    if BACKEND == "nccl" and available_gpus == 0:
        raise RuntimeError("NCCL requires CUDA GPUs")

    # Initialize CSV once
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

    for world_size in WORLD_SIZES:

        if BACKEND == "nccl" and world_size > available_gpus:
            print(f"Skip {world_size}: insufficient GPUs")
            continue

        mp.spawn(
            fn=distributed_train,
            args=(world_size, SIZES_MB, BACKEND),
            nprocs=world_size,
            join=True,
        )

    print(f"\nCSV saved to: {CSV_PATH}")