"""CUDA benchmark: uv run python -u -m cs336_systems.flash_benchmark."""

import gc

import torch
import triton
from triton.testing import do_bench

from cs336_systems.flash_attention import BenchmarkAttention, FlashAttentionTriton


# ============================================================
# 1. Benchmark Configuration
# ============================================================

DEVICE = "cuda:0"
BATCH_SIZE = 1
IS_CAUSAL = True
SEQ_LENS = [2**i for i in range(7, 17)]
HEAD_DIMS = [16, 32, 64, 128]
DTYPES = [torch.bfloat16, torch.float32]
SEED = 42

# do_bench uses milliseconds, not iteration counts.
WARMUP_MS = 25
REP_MS = 100

IMPLEMENTATIONS = [
    ("PyTorch", BenchmarkAttention.apply),
    ("Triton", FlashAttentionTriton.apply),
]


# ============================================================
# 2. Benchmark One Implementation
# ============================================================

def benchmark_attention(implementation, inputs, grad_output):
    """Return forward, backward, forward+backward times in ms and status."""
    timings = [None, None, None]
    output = None
    status = "OK"

    def forward():
        return implementation(*inputs, IS_CAUSAL)

    def forward_backward():
        forward().backward(grad_output)

    try:
        # do_bench calls the function before timing, excluding first compilation.
        timings[0] = do_bench(
            forward, warmup=WARMUP_MS, rep=REP_MS, return_mode="mean"
        )

        # Build the graph outside timing; reuse it to measure backward alone.
        output = forward()
        timings[1] = do_bench(
            lambda: output.backward(grad_output, retain_graph=True),
            warmup=WARMUP_MS, rep=REP_MS, return_mode="mean",
            grad_to_none=inputs,
        )

        # Release the retained graph before timing fresh forward/backward steps.
        output = None
        for tensor in inputs:
            tensor.grad = None
        timings[2] = do_bench(
            forward_backward, warmup=WARMUP_MS, rep=REP_MS,
            return_mode="mean", grad_to_none=inputs,
        )
    except torch.cuda.OutOfMemoryError:
        status = "OOM"
    finally:
        output = None
        for tensor in inputs:
            tensor.grad = None

    return timings, status


# ============================================================
# 3. Table Output
# ============================================================

def print_row(name, dtype, seq_len, head_dim, timings, status):
    values = [f"{value:.3f}" if value is not None else "OOM" for value in timings]
    dtype_name = str(dtype).removeprefix("torch.")
    print(
        f"{name:<10} {dtype_name:<10} {seq_len:>8} {head_dim:>6} "
        f"{values[0]:>14} {values[1]:>14} {values[2]:>14} {status:>8}",
        flush=True,
    )


# ============================================================
# 4. Run All Configurations
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA GPU; run it on Dell or a cloud GPU.")

    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    gpu_name = torch.cuda.get_device_name(device)

    print("=" * 112)
    print("FlashAttention Benchmark")
    print(f"GPU: {gpu_name} | Device: {device}")
    if "B200" not in gpu_name:
        print("Hardware note: this is not the B200 required by the assignment.")
    print(f"PyTorch: {torch.__version__} | Triton: {triton.__version__} | CUDA: {torch.version.cuda}")
    print(f"Batch: {BATCH_SIZE} | Causal: {IS_CAUSAL} | Seed: {SEED} | Triton tiles: 16 x 16")
    print(f"do_bench warmup: {WARMUP_MS} ms | rep: {REP_MS} ms | Statistic: mean | Unit: ms")
    print("Both implementations use the same torch.compile'd PyTorch backward.")
    print("OOM marks the failed phase and any remaining unmeasured phases.")
    print("=" * 112)
    print(
        f"{'Impl':<10} {'Dtype':<10} {'Seq len':>8} {'D':>6} "
        f"{'Forward (ms)':>14} {'Backward (ms)':>14} {'Fwd+Bwd (ms)':>14} {'Status':>8}",
        flush=True,
    )
    print("-" * 112, flush=True)

    for dtype in DTYPES:
        for seq_len in SEQ_LENS:
            for head_dim in HEAD_DIMS:
                # Reset between shapes so compiled backward does not hit its
                # recompile limit and silently switch to eager execution.
                torch.compiler.reset()
                inputs = []
                grad_output = None
                try:
                    # Allocate all random inputs before any measured operation.
                    shape = (BATCH_SIZE, seq_len, head_dim)
                    for _ in range(3):
                        inputs.append(torch.randn(shape, device=device, dtype=dtype, requires_grad=True))
                    grad_output = torch.randn(shape, device=device, dtype=dtype)
                except torch.cuda.OutOfMemoryError:
                    for name, _ in IMPLEMENTATIONS:
                        print_row(name, dtype, seq_len, head_dim, [None] * 3, "OOM")
                else:
                    for name, implementation in IMPLEMENTATIONS:
                        timings, status = benchmark_attention(implementation, inputs, grad_output)
                        print_row(name, dtype, seq_len, head_dim, timings, status)
                        # The helper has returned, so failed graphs can be freed
                        # before trying the other implementation on the same data.
                        gc.collect()
                        torch.cuda.empty_cache()
                finally:
                    inputs.clear()
                    grad_output = None
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
