"""Minimal two-rank test for FSDP weight sharding and Linear forward.

Place this file under cs336_systems/ and run:
    python -m cs336_systems.test_fsdp_shard

This tests forward numerics but intentionally does NOT test backward.
"""

import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_basics.model import Embedding, Linear
from cs336_systems.zero import FSDP


WORLD_SIZE = 2


def worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        # Intentionally initialize different weights on each rank.
        # FSDP is expected to broadcast Rank 0's weights first.
        torch.manual_seed(42 + rank)
        model = Linear(8, 4)  # (4, 8) = 32 elements

        expected = model.weight.detach().clone()
        dist.broadcast(expected, src=0)

        fsdp = FSDP(model)
        shard = fsdp.module.weight.detach()

        # Verify each rank keeps 16 FP32 elements (not the full 32).
        assert tuple(shard.shape) == (16,), (
            f"Rank {rank}: got shard shape {tuple(shard.shape)}, expected (16,)"
        )
        assert shard.dtype == torch.float32

        # Check actual shard values against the original Rank 0 weight.
        expected_shard = expected.flatten()[rank * 16:(rank + 1) * 16]
        torch.testing.assert_close(shard, expected_shard)

        # Reconstruct the original matrix in the *test* using all_gather.
        pieces = [torch.empty_like(shard) for _ in range(world_size)]
        dist.all_gather(pieces, shard)
        reconstructed = torch.cat(pieces).reshape(4, 8)
        torch.testing.assert_close(reconstructed, expected)

        # Test the user's FSDP.forward against the original full weight.
        torch.manual_seed(123)
        x = torch.randn(2, 8)
        with torch.no_grad():
            actual = fsdp(x)
            reference = torch.nn.functional.linear(x, expected)
        torch.testing.assert_close(actual, reference)
        assert tuple(actual.shape) == (2, 4)

        print(
            f"[PASS] Rank {rank} | "
            f"shard shape={tuple(shard.shape)} | "
            f"dtype={shard.dtype} | "
            f"forward shape={tuple(actual.shape)} | "
            f"first 4 values={shard[:4].tolist()}",
            flush=True,
        )
        dist.barrier()
        if rank == 0:
            print("[PASS] Shards reconstruct original weight; forward matches full Linear.", flush=True)
    finally:
        dist.destroy_process_group()


def main() -> None:
    # Pick an available local port, so this is less likely to conflict with
    # previous DDP benchmark runs.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    mp.spawn(worker, args=(WORLD_SIZE, port), nprocs=WORLD_SIZE, join=True)
    print("\nFSDP sharding + forward test PASSED.")


if __name__ == "__main__":
    main()
