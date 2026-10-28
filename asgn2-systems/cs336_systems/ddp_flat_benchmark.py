import os
import time
import statistics

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.ddp import NaiveDDP, FlatDDP


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

    if backend == "nccl":
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def make_model():
    # 先用小模型测试，正式实验换成 2.1.2 的 XL 模型
    layers = []

    for _ in range(8):
        layers.append(nn.Linear(512, 512))
        layers.append(nn.GELU())

    return nn.Sequential(*layers)


def benchmark(
    ddp_class,
    device,
    inputs,
    targets,
    warmup=5,
    iters=20,
):
    # 保证两种 DDP 的初始模型一致
    torch.manual_seed(42)

    model = ddp_class(make_model().to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    step_times = []
    comm_times = []

    for i in range(warmup + iters):

        if device.type == "cuda":
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()

        sync(device)
        step_start = time.perf_counter()

        # Forward
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = loss_fn(output, targets)

        # Backward
        loss.backward()
        sync(device)

        # Communication
        comm_start = time.perf_counter()

        model.finish_gradient_synchronization()

        sync(device)
        comm_end = time.perf_counter()

        # Optimizer
        optimizer.step()
        sync(device)

        step_end = time.perf_counter()

        if i >= warmup:
            step_times.append((step_end - step_start) * 1000)
            comm_times.append((comm_end - comm_start) * 1000)

    # 每一轮取所有 rank 中最慢的耗时
    step_tensor = torch.tensor(step_times, device=device)
    comm_tensor = torch.tensor(comm_times, device=device)

    dist.all_reduce(step_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(comm_tensor, op=dist.ReduceOp.MAX)

    return {
        "step_ms": statistics.median(step_tensor.tolist()),
        "comm_ms": statistics.median(comm_tensor.tolist()),
    }


def distributed_train(rank, world_size, backend):
    setup(rank, world_size, backend)

    try:
        device = torch.device(
            f"cuda:{rank}" if backend == "nccl" else "cpu"
        )

        # 各 rank 使用不同数据，但两种 DDP 用相同数据
        torch.manual_seed(2026 + rank)
        inputs = torch.randn(16, 512, device=device)
        targets = torch.randn(16, 512, device=device)

        results = {}

        for name, ddp_class in [
            ("Naive DDP", NaiveDDP),
            ("Flat DDP", FlatDDP),
        ]:
            results[name] = benchmark(
                ddp_class,
                device,
                inputs,
                targets,
            )

        if rank == 0:
            print_results(results)

    finally:
        dist.destroy_process_group()


def print_results(results):
    print(
        f"{'Method':<16}"
        f"{'Iteration(ms)':>16}"
        f"{'Comm(ms)':>14}"
        f"{'Comm %':>12}"
    )

    for name, result in results.items():
        step = result["step_ms"]
        comm = result["comm_ms"]

        print(
            f"{name:<16}"
            f"{step:>16.3f}"
            f"{comm:>14.3f}"
            f"{comm / step * 100:>11.2f}%"
        )

    naive = results["Naive DDP"]
    flat = results["Flat DDP"]

    print("\nAnalysis:")
    print(
        f"  Iteration speedup: "
        f"{naive['step_ms'] / flat['step_ms']:.2f}x"
    )
    print(
        f"  Communication speedup: "
        f"{naive['comm_ms'] / flat['comm_ms']:.2f}x"
    )


if __name__ == "__main__":
    WORLD_SIZE = 4
    BACKEND = "gloo"  # AutoDL GPU: "nccl"

    if BACKEND == "nccl":
        assert torch.cuda.device_count() >= WORLD_SIZE

    print(f"\n{'=' * 65}")
    print("DDP Benchmark Comparison")
    print(f"World Size: {WORLD_SIZE}, Backend: {BACKEND}")
    print(f"{'=' * 65}")

    mp.spawn(
        fn=distributed_train,
        args=(WORLD_SIZE, BACKEND),
        nprocs=WORLD_SIZE,
        join=True,
    )