import torch
import timeit
import gc
from statistics import mean, pstdev

from model import MultiheadSelfAttention

# ============================================================
# 1. Benchmark Configuration
# ============================================================

D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]

BATCH_SIZE = 8
NUM_HEADS = 1
THETA = 10000.0

WARMUP_ITERS = 10
BENCHMARK_ITERS = 100
SEED = 42

MiB = 1024 ** 2
COMPILE = True

# ============================================================
# 2. Device Utilities
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def sync(device):
    """Wait for all queued GPU operations to finish."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def allocated_memory_mib(device):
    """Currently allocated PyTorch device memory (MiB)."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / MiB
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / MiB
    return None

# ============================================================
# 3. Benchmark a Single Attention Configuration
# ============================================================

def benchmark_attention(d_model, seq_len, device):
    torch.manual_seed(SEED)

    model_benchmark = MultiheadSelfAttention(
        d_model, NUM_HEADS, seq_len, THETA
    ).to(device)

    model = torch.compile(model_benchmark) if COMPILE else model_benchmark

    model.train()

    # --------------------------------------------------------
    # Step 1: Warm-up
    # Exclude first-run initialization overhead from timing.
    # --------------------------------------------------------

    for _ in range(WARMUP_ITERS):
        inputs = torch.rand((BATCH_SIZE, seq_len, d_model), device=device)

        outputs = model(inputs)
        outputs.sum().backward()
        model.zero_grad(set_to_none=True)

        del inputs, outputs

    sync(device)

    # --------------------------------------------------------
    # Step 2: Forward / Backward Timing
    # Allocate inputs before timing.
    # Synchronize GPU before and after each measured operation.
    # --------------------------------------------------------

    forward_times = []
    backward_times = []
    for _ in range(BENCHMARK_ITERS):
        inputs = torch.rand((BATCH_SIZE, seq_len, d_model), device=device)

        sync(device)
        start = timeit.default_timer()

        outputs: torch.Tensor = model(inputs)

        sync(device)
        forward_end = timeit.default_timer()

        outputs.sum().backward()

        sync(device)
        backward_end = timeit.default_timer()

        forward_times.append((forward_end - start) * 1000)
        backward_times.append((backward_end - forward_end) * 1000)

        model.zero_grad(set_to_none=True)
        del inputs, outputs

    # --------------------------------------------------------
    # Step 3: Saved Tensor Memory
    # Run separately to avoid hook overhead affecting timings.
    #
    # saved_bytes: Logical size of tensors saved by autograd.
    # memory_before_bwd: Total active PyTorch device allocation
    #                    immediately before backward.
    # --------------------------------------------------------

    saved_bytes = 0

    def pack_hook(tensor):
        nonlocal saved_bytes

        # Exclude directly saved model parameters.
        if not isinstance(tensor, torch.nn.Parameter):
            saved_bytes += tensor.numel() * tensor.element_size()

        return tensor

    def unpack_hook(tensor):
        return tensor

    inputs = torch.randn((BATCH_SIZE, seq_len, d_model), device=device, requires_grad=True)

    with torch.autograd.graph.saved_tensors_hooks(
        pack_hook, unpack_hook
    ):
        outputs = model(inputs)

    sync(device)

    # Measure before backward starts, not after it finishes.
    saved_memory_mib = saved_bytes / MiB
    memory_before_bwd = allocated_memory_mib(device)

    outputs.sum().backward()
    sync(device)

    model.zero_grad(set_to_none=True)
    del inputs, outputs

    # --------------------------------------------------------
    # Step 4: Return Results
    # --------------------------------------------------------

    return {
        "forward_mean": mean(forward_times),
        "forward_std": pstdev(forward_times),
        "backward_mean": mean(backward_times),
        "backward_std": pstdev(backward_times),
        "saved_mib": saved_memory_mib,
        "allocated_mib": memory_before_bwd,
    }

def print_failed_row(d_model, seq_len, status):
    print(
        f"{d_model:>7} {seq_len:>8} "
        f"{'—':>20} {'—':>20} "
        f"{'—':>13} {'—':>16} "
        f"{status:>8}",
        flush=True,
    )

def main():
    import gc

    device = get_device()

    # ============================================================
    # 1. Configuration
    # ============================================================

    # Based on previous experiments:
    # seq_len = 4096 succeeded, while 8192 failed.
    # Skip larger configurations to avoid CUDA errors.
    MAX_SAFE_SEQ_LEN = 16384

    results = []
    oom_seq_threshold = None

    # ============================================================
    # 2. Print Header
    # ============================================================

    print("=" * 108)
    print("PyTorch Attention Benchmark")
    print(f"Device: {device} | Batch: {BATCH_SIZE} | Heads: {NUM_HEADS}")
    print(f"Warm-up: {WARMUP_ITERS} | Timed iterations: {BENCHMARK_ITERS}")
    print(f"Safe sequence limit: {MAX_SAFE_SEQ_LEN}")
    print("Time unit: ms | Memory unit: MiB")
    print("=" * 108)

    print(
        f"{'d_model':>7} {'seq_len':>8} "
        f"{'Forward (ms)':>20} {'Backward (ms)':>20} "
        f"{'Saved (MiB)':>13} {'Allocated (MiB)':>16} "
        f"{'Status':>8}"
    )
    print("-" * 108)

    # ============================================================
    # 3. Helper: Print Status
    # ============================================================

    def print_status(d_model, seq_len, status):
        print(
            f"{d_model:>7} {seq_len:>8} "
            f"{'—':>20} {'—':>20} "
            f"{'—':>13} {'—':>16} "
            f"{status:>8}",
            flush=True,
        )

    # ============================================================
    # 4. Run Benchmarks
    # ============================================================

    for d_model in D_MODELS:
        for seq_len in SEQ_LENS:

            # Skip known high-risk sequence lengths.
            if seq_len > MAX_SAFE_SEQ_LEN:
                print_status(d_model, seq_len, "SKIP")

                results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "status": "SKIP",
                })
                continue

            # Skip subsequent configurations after a real OOM.
            if (
                oom_seq_threshold is not None
                and seq_len >= oom_seq_threshold
            ):
                print_status(d_model, seq_len, "SKIP")

                results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "status": "SKIP",
                })
                continue

            try:
                result = benchmark_attention(
                    d_model, seq_len, device
                )

                forward_str = (
                    f"{result['forward_mean']:.2f} "
                    f"± {result['forward_std']:.2f}"
                )
                backward_str = (
                    f"{result['backward_mean']:.2f} "
                    f"± {result['backward_std']:.2f}"
                )

                allocated = result["allocated_mib"]
                allocated_str = (
                    f"{allocated:.2f}"
                    if allocated is not None else "N/A"
                )

                print(
                    f"{d_model:>7} {seq_len:>8} "
                    f"{forward_str:>20} "
                    f"{backward_str:>20} "
                    f"{result['saved_mib']:>13.2f} "
                    f"{allocated_str:>16} "
                    f"{'OK':>8}",
                    flush=True,
                )

                results.append({
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "status": "OK",
                    **result,
                })

            except RuntimeError as e:

                is_oom = (
                    isinstance(e, torch.cuda.OutOfMemoryError)
                    or "out of memory" in str(e).lower()
                )

                if is_oom:
                    print_status(d_model, seq_len, "OOM")

                    oom_seq_threshold = (
                        seq_len
                        if oom_seq_threshold is None
                        else min(oom_seq_threshold, seq_len)
                    )

                    results.append({
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "status": "OOM",
                    })

                    gc.collect()

                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif device.type == "mps":
                        torch.mps.empty_cache()

                else:
                    print(
                        f"\nCUDA / Runtime Error: "
                        f"d_model={d_model}, seq_len={seq_len}\n"
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    raise

    # ============================================================
    # 5. Summary
    # ============================================================

    print("=" * 108)

    ok_count = sum(r["status"] == "OK" for r in results)
    oom_count = sum(r["status"] == "OOM" for r in results)
    skip_count = sum(r["status"] == "SKIP" for r in results)

    print(
        f"Summary: "
        f"{ok_count} OK | "
        f"{oom_count} OOM | "
        f"{skip_count} SKIP"
    )

    print("=" * 108)

    return results

if __name__ == "__main__":
    main()

