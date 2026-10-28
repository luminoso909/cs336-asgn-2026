"""CS336 A2: DDP and ZeRO-1 benchmark with clean, rank-aggregated console output.

Run from the assignment2-systems project root, for example:
  PROFILE_METHOD=Async OPT_MODE=both BENCH_MODEL=small python -m cs336_systems.ddp_zero_benchmark_pretty

The file is delivered separately: copy it to cs336_systems/ddp_zero_benchmark_pretty.py.
"""

import csv
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import TransformerLM
from cs336_basics.utils import cross_entropy
from cs336_systems.ddp import NaiveDDP, FlatDDP, AsyncDDP
from cs336_systems.zero import ZeRO1


MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}
DDP_CLASSES = {"Naive": NaiveDDP, "Flat": FlatDDP, "Async": AsyncDDP}

TEST_MODELS = os.getenv("BENCH_MODEL", "small").split(",")
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "128"))
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "10000"))
WARMUP = int(os.getenv("BENCH_WARMUP", "5"))
STEPS = int(os.getenv("BENCH_STEPS", "30"))
PROFILE_METHOD = os.getenv("PROFILE_METHOD")  # Naive, Flat, Async, or unset (all)
OPT_MODE = os.getenv("OPT_MODE", "normal")  # normal, zero1, or both
CSV_PATH = Path(__file__).resolve().parent / "results" / "ddp_zero_compare.csv"
GIB = 1024 ** 3


def setup(rank: int, world_size: int) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device: torch.device) -> dict | None:
    if device.type != "cuda":
        return None
    sync(device)
    return {
        "allocated": torch.cuda.memory_allocated(device) / GIB,
        "peak": torch.cuda.max_memory_allocated(device) / GIB,
        "reserved": torch.cuda.memory_reserved(device) / GIB,
    }


def memory_probe(ddp_model, optimizer, inputs, targets, device):
    """One *unmeasured* training step; observe AdamW's lazy state allocation."""
    if device.type != "cuda":
        return None
    measurements = {"after_init": memory_snapshot(device)}
    torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)
    outputs = ddp_model(inputs)
    loss = cross_entropy(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    del outputs, loss

    measurements["before_step"] = memory_snapshot(device)
    optimizer.step()
    measurements["after_step"] = memory_snapshot(device)

    inner = optimizer.local_optimizer if isinstance(optimizer, ZeRO1) else optimizer
    state_bytes = 0
    if inner is not None:
        for state in inner.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    state_bytes += value.numel() * value.element_size()

    measurements["state_gib"] = state_bytes / GIB
    measurements["params_gib"] = sum(
        p.numel() * p.element_size() for p in ddp_model.parameters()
    ) / GIB
    measurements["grads_gib"] = sum(
        p.grad.numel() * p.grad.element_size()
        for p in ddp_model.parameters() if p.grad is not None
    ) / GIB
    return measurements


def benchmark_step(ddp_model, optimizer, inputs, targets, device):
    is_cuda = device.type == "cuda"
    if is_cuda:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    sync(device)
    t0 = time.perf_counter()
    if is_cuda:
        events[0].record()
        torch.cuda.nvtx.range_push("Forward")

    optimizer.zero_grad(set_to_none=True)
    outputs = ddp_model(inputs)
    loss = cross_entropy(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
    if is_cuda:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Backward")

    loss.backward()
    if is_cuda:
        torch.cuda.nvtx.range_pop()
        events[1].record()
    else:
        t1 = time.perf_counter()

    # Do NOT synchronize CUDA here: Async DDP should be allowed to overlap work.
    if is_cuda:
        torch.cuda.nvtx.range_push("Finish Sync")
    ddp_model.finish_gradient_synchronization()
    if is_cuda:
        torch.cuda.nvtx.range_pop()
        events[2].record()
    else:
        t2 = time.perf_counter()

    optimizer.step()  # For ZeRO1, includes broadcasts of updated parameters.
    if is_cuda:
        events[3].record()
        sync(device)
        t3 = time.perf_counter()
        pre_finish = events[0].elapsed_time(events[1])
        finish_tail = events[1].elapsed_time(events[2])
        optimizer_ms = events[2].elapsed_time(events[3])
    else:
        t3 = time.perf_counter()
        pre_finish = (t1 - t0) * 1000
        finish_tail = (t2 - t1) * 1000
        optimizer_ms = (t3 - t2) * 1000

    return {
        "total_ms": (t3 - t0) * 1000,
        "pre_finish_ms": pre_finish,
        "finish_tail_ms": finish_tail,
        "optimizer_ms": optimizer_ms,
    }


def aggregate_timings(all_rank_measurements, steps: int):
    metrics = list(all_rank_measurements[0][0].keys())
    return {
        key: statistics.median([
            max(rank_data[step][key] for rank_data in all_rank_measurements)
            for step in range(steps)
        ])
        for key in metrics
    }


def format_gib(value):
    return f"{value:.3f}" if value is not None else "--"


def print_memory_table(memory_by_rank):
    if not memory_by_rank or all(m is None for m in memory_by_rank):
        print("  GPU memory: N/A (CPU + Gloo; CUDA memory accounting skipped)")
        return

    def get_value(record, key):
        if record is None:
            return None
        if isinstance(key, tuple):
            stage, field = key
            return record[stage][field]
        return record[key]

    rank_headers = "".join(f"{'Rank ' + str(i):>12}" for i in range(len(memory_by_rank)))
    print(f"  {'Metric (GiB)':<24}{rank_headers}{'Max':>12}")
    print("  " + "-" * (24 + 12 * (len(memory_by_rank) + 1)))
    rows = [
        ("After init / allocated", ("after_init", "allocated")),
        ("Before step / allocated", ("before_step", "allocated")),
        ("After step / allocated", ("after_step", "allocated")),
        ("Peak since init", ("after_step", "peak")),
        ("Optimizer state", "state_gib"),
        ("Model parameters", "params_gib"),
        ("Gradients", "grads_gib"),
    ]
    for label, key in rows:
        values = [get_value(record, key) for record in memory_by_rank]
        maximum = max((v for v in values if v is not None), default=None)
        print(f"  {label:<24}" + "".join(f"{format_gib(v):>12}" for v in values)
              + f"{format_gib(maximum):>12}")
    print("  Peak = max allocated since reset immediately after model initialization.")


def print_report(model_size, results, world_size, backend):
    line = "=" * 100
    print(f"\n{line}")
    print(f"  CS336 A2 | DDP x Optimizer Benchmark | Model: {model_size} | World size: {world_size} | {backend.upper()}")
    print(f"  Warmup: {WARMUP}  |  Timed steps: {STEPS}  |  Batch/rank: {BATCH_SIZE}  |  Sequence: {SEQ_LEN}")
    print(line)
    for r in results:
        print(f"\n[MEMORY] DDP={r['ddp']} | Optimizer={r['optimizer']}")
        print_memory_table(r["memory_by_rank"])

    print("\n[TIME] Median of per-step slowest rank (milliseconds; lower is better)")
    print(f"  {'DDP':<9}{'Optimizer':<12}{'Total':>13}{'PreFinish':>14}{'Sync tail':>13}{'Opt + comm':>14}{'vs normal':>13}")
    print("  " + "-" * 88)
    baseline_by_ddp = {
        r["ddp"]: r["timings"]["total_ms"] for r in results if r["optimizer"] == "normal"
    }
    for r in results:
        t = r["timings"]
        baseline = baseline_by_ddp.get(r["ddp"])
        speedup = f"{baseline / t['total_ms']:.2f}x" if baseline else "--"
        print(f"  {r['ddp']:<9}{r['optimizer']:<12}"
              f"{t['total_ms']:>13.2f}{t['pre_finish_ms']:>14.2f}"
              f"{t['finish_tail_ms']:>13.2f}{t['optimizer_ms']:>14.2f}{speedup:>13}")

    paired_ddp = sorted({r["ddp"] for r in results})
    for method in paired_ddp:
        normal = next((r for r in results if r["ddp"] == method and r["optimizer"] == "normal"), None)
        zero1 = next((r for r in results if r["ddp"] == method and r["optimizer"] == "zero1"), None)
        if normal and zero1:
            fraction = (1 - zero1["timings"]["total_ms"] / normal["timings"]["total_ms"]) * 100
            wording = "faster" if fraction >= 0 else "slower"
            print(f"\n[COMPARE] {method}: ZeRO1 is {abs(fraction):.1f}% {wording} in total iteration time (this run).")
            if normal["memory_by_rank"][0] is not None:
                normal_max = max(x["state_gib"] for x in normal["memory_by_rank"])
                zero_max = max(x["state_gib"] for x in zero1["memory_by_rank"])
                savings = (1 - zero_max / normal_max) * 100 if normal_max > 0 else 0
                print(f"          Per-rank max optimizer-state memory: {normal_max:.3f} -> {zero_max:.3f} GiB ({savings:.1f}% less).")


def save_results(model_size, results, world_size, backend):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "world_size", "backend", "ddp", "optimizer",
        "total_ms", "pre_finish_ms", "finish_tail_ms", "optimizer_ms",
        "max_after_init_gib", "max_before_step_gib", "max_after_step_gib",
        "max_peak_gib", "max_optimizer_state_gib", "total_optimizer_state_gib",
    ]
    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        for r in results:
            memory = r["memory_by_rank"]
            def maximum(stage, key):
                if memory[0] is None:
                    return None
                return max(m[stage][key] for m in memory)
            states = None if memory[0] is None else [m["state_gib"] for m in memory]
            writer.writerow({
                "model": model_size,
                "world_size": world_size,
                "backend": backend,
                "ddp": r["ddp"],
                "optimizer": r["optimizer"],
                **r["timings"],
                "max_after_init_gib": maximum("after_init", "allocated"),
                "max_before_step_gib": maximum("before_step", "allocated"),
                "max_after_step_gib": maximum("after_step", "allocated"),
                "max_peak_gib": maximum("after_step", "peak"),
                "max_optimizer_state_gib": max(states) if states is not None else None,
                "total_optimizer_state_gib": sum(states) if states is not None else None,
            })


def run(rank, world_size, model_size, steps, warmup):
    device = setup(rank, world_size)
    try:
        config = MODEL_CONFIGS[model_size]
        torch.manual_seed(2026 + rank)
        tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN + 1), device=device)
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        methods = ({PROFILE_METHOD: DDP_CLASSES[PROFILE_METHOD]}
                   if PROFILE_METHOD is not None else DDP_CLASSES)
        modes = ["normal", "zero1"] if OPT_MODE == "both" else [OPT_MODE]
        results = []

        for name, ddp_class in methods.items():
            for mode in modes:
                if rank == 0:
                    print(f"[RUN] {model_size:<7} | DDP={name:<5} | Optimizer={mode:<6} | {warmup} warmup + {steps} timed steps", flush=True)
                torch.manual_seed(42)
                model = TransformerLM(
                    VOCAB_SIZE, SEQ_LEN, config["d_model"], config["num_layers"],
                    config["num_heads"], config["d_ff"], 10000.0,
                ).to(device)
                ddp_model = ddp_class(model)
                opt_kwargs = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=0.1, foreach=False)
                optimizer = (torch.optim.AdamW(ddp_model.parameters(), **opt_kwargs)
                             if mode == "normal" else
                             ZeRO1(ddp_model.parameters(), optimizer_cls=torch.optim.AdamW,
                                   **opt_kwargs))

                memory = memory_probe(ddp_model, optimizer, inputs, targets, device)
                memory_by_rank = [None] * world_size
                dist.all_gather_object(memory_by_rank, memory)

                measurements = []
                for step in range(warmup + steps):
                    if device.type == "cuda":
                        dist.barrier(device_ids=[device.index])
                    else:
                        dist.barrier()
                    t = benchmark_step(ddp_model, optimizer, inputs, targets, device)
                    if step >= warmup:
                        measurements.append(t)
                    if rank == 0 and (step + 1) % 10 == 0:
                        print(f"      step {step + 1}/{warmup + steps}", flush=True)

                all_measurements = [None] * world_size
                dist.all_gather_object(all_measurements, measurements)
                if rank == 0:
                    results.append({
                        "ddp": name,
                        "optimizer": mode,
                        "timings": aggregate_timings(all_measurements, steps),
                        "memory_by_rank": memory_by_rank,
                    })
                del optimizer, ddp_model, model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if rank == 0:
            print_report(model_size, results, world_size, dist.get_backend())
            save_results(model_size, results, world_size, dist.get_backend())
            print(f"\n[CSV] {CSV_PATH}", flush=True)
    finally:
        dist.destroy_process_group()


def main():
    if PROFILE_METHOD is not None and PROFILE_METHOD not in DDP_CLASSES:
        raise ValueError(f"PROFILE_METHOD must be one of {list(DDP_CLASSES)}")
    if OPT_MODE not in ("normal", "zero1", "both"):
        raise ValueError("OPT_MODE must be normal, zero1, or both")
    if WARMUP < 0 or STEPS < 1 or WORLD_SIZE < 1:
        raise ValueError("WARMUP >= 0, STEPS >= 1, WORLD_SIZE >= 1 required")
    if torch.cuda.is_available() and torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(f"Need {WORLD_SIZE} CUDA devices, found {torch.cuda.device_count()}")
    for model_size in TEST_MODELS:
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_size}")
        mp.spawn(run, args=(WORLD_SIZE, model_size, STEPS, WARMUP),
                 nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
