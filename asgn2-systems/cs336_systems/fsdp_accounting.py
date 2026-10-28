
"""CS336 A2 FSDP memory and timing measurements; run on 2 CUDA GPUs.

Example:
  FSDP_MODE=fsdp FSDP_DTYPE=bf16 python -m cs336_systems.fsdp_accounting
  FSDP_MODE=ddp FSDP_DTYPE=bf16 python -m cs336_systems.fsdp_accounting

For Nsight:
  FSDP_MODE=fsdp nsys profile -t cuda,nvtx -o fsdp_trace \
      python -m cs336_systems.fsdp_accounting

CUDA devices are required for the assignment's official memory/profile results.
"""

import os
import time
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import TransformerLM
from cs336_systems.zero import FSDP


MODE = os.getenv("FSDP_MODE", "fsdp")
DTYPE = os.getenv("FSDP_DTYPE", "bf16")

BATCH = int(os.getenv("FSDP_BATCH", "1"))
SEQ = int(os.getenv("FSDP_SEQ", "128"))
WARMUP = int(os.getenv("FSDP_WARMUP", "3"))
STEPS = int(os.getenv("FSDP_STEPS", "10"))

VOCAB = 10000


def measure(device, label):
    torch.cuda.synchronize(device)

    return {
        "label": label,
        "alloc_gib": (
            torch.cuda.memory_allocated(device) / (1024 ** 3)
        ),
        "peak_gib": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        ),
        "reserved_gib": (
            torch.cuda.memory_reserved(device) / (1024 ** 3)
        ),
    }


def worker(rank, world):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29562",
        rank=rank,
        world_size=world,
    )

    try:
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": None,
        }[DTYPE]

        torch.manual_seed(42)

        model = TransformerLM(
            VOCAB,
            SEQ,
            2560,
            32,
            32,
            10240,
            10000.0,
        ).to(device)

        if MODE == "fsdp":
            wrapped = FSDP(
                model,
                compute_dtype=dtype,
            )

        elif MODE == "ddp":
            wrapped = (
                torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[rank],
                )
            )

        else:
            raise ValueError(MODE)

        optim = torch.optim.AdamW(
            wrapped.parameters(),
            lr=1e-4,
            foreach=False,
        )

        torch.manual_seed(123 + rank)

        tokens = torch.randint(
            0,
            VOCAB,
            (BATCH, SEQ + 1),
            device=device,
        )

        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        snaps = [
            measure(device, "after_init")
        ]

        duration = []

        for step in range(WARMUP + STEPS):

            dist.barrier(device_ids=[rank])
            torch.cuda.synchronize(device)

            t0 = time.perf_counter()

            optim.zero_grad(set_to_none=True)

            with torch.cuda.nvtx.range("forward"):
                output = wrapped(inputs)

                loss = F.cross_entropy(
                    output.reshape(-1, VOCAB).float(),
                    targets.reshape(-1),
                )

            with torch.cuda.nvtx.range("backward"):
                loss.backward()

            if MODE == "fsdp":
                wrapped.finish_gradient_synchronization()

            if step == 0:
                del output, loss

                snaps.append(
                    measure(device, "before_first_step")
                )

            with torch.cuda.nvtx.range("optimizer_step"):
                optim.step()

            torch.cuda.synchronize(device)

            if step == 0:
                snaps.append(
                    measure(device, "after_first_step")
                )

            if step >= WARMUP:
                duration.append(
                    (time.perf_counter() - t0) * 1e3
                )

        records = [None] * world

        dist.all_gather_object(
            records,
            {
                "rank": rank,
                "memory": snaps,
                "times": duration,
            },
        )

        if rank == 0:
            print(
                f"\nFSDP ACCOUNTING | "
                f"mode={MODE} | "
                f"dtype={DTYPE} | xl | {world} GPUs"
            )

            for rec in records:
                print(f"Rank {rec['rank']}")

                for m in rec["memory"]:
                    print(
                        f"  {m['label']:20} "
                        f"alloc={m['alloc_gib']:.3f} GiB "
                        f"peak={m['peak_gib']:.3f} GiB "
                        f"reserved={m['reserved_gib']:.3f} GiB"
                    )

            slowest = [
                max(r["times"][i] for r in records)
                for i in range(STEPS)
            ]

            print(
                "Step time (median of slowest rank): "
                f"{statistics.median(slowest):.3f} ms"
            )

            max_peak = max(
                max(x["peak_gib"] for x in r["memory"])
                for r in records
            )

            print(f"Max rank peak: {max_peak:.3f} GiB")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if torch.cuda.device_count() < 2:
        raise SystemExit(
            "This XL accounting/profile script requires two CUDA GPUs."
        )

    mp.spawn(
        worker,
        args=(2,),
        nprocs=2,
        join=True,
    )
