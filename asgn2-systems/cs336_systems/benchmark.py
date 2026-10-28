"""CS336 A2 Transformer benchmark, restored from a complete user chat paste.

The original --compile type=bool bug is fixed: --compile False now means False.
Measurements retain the original cumulative timing/reporting convention.
"""

import argparse
import timeit
from statistics import pstdev

import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, ProfilerActivity

from model import TransformerLM
from optimizer import AdamW
from cs336_basics.utils import cross_entropy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Testing a CS336 A2 Model Benchmark")

    # Training scheduling
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    )

    # Model params
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--theta", type=float, default=10000.0)

    # Optim params
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Restored original CLI --compile True / False, but parse it correctly.
    def parse_bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in {"true", "1", "yes", "on"}:
            return True
        if s.lower() in {"false", "0", "no", "off"}:
            return False
        raise argparse.ArgumentTypeError("--compile must be True or False")

    parser.add_argument("--compile", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-compile", action="store_false", dest="compile")

    return parser.parse_args(argv)


def sync(args):
    if args.device == "cuda":
        torch.cuda.synchronize()
    elif args.device == "mps":
        torch.mps.synchronize()


def get_batch(args):
    token_ids = torch.randint(
        0, args.vocab_size, (args.batch_size, args.context_length + 1)
    ).to(device=args.device)
    inputs = token_ids[..., :-1]
    targets = token_ids[..., 1:]
    return inputs, targets


def train(args, model: TransformerLM, optimizer: AdamW):
    inputs, targets = get_batch(args)
    optimizer.zero_grad(set_to_none=True)

    with nvtx.range("forward") if args.device == "cuda" else _nullcontext():
        sync(args)
        start = timeit.default_timer()
        logits = model(inputs)
        sync(args)
        time1 = timeit.default_timer()

    with nvtx.range("backward") if args.device == "cuda" else _nullcontext():
        loss = cross_entropy(logits, targets)
        loss.backward()
        sync(args)
        time2 = timeit.default_timer()

    with nvtx.range("optimizer_step") if args.device == "cuda" else _nullcontext():
        optimizer.step()
        sync(args)
        time3 = timeit.default_timer()

    return time1 - start, time2 - start, time3 - start


from contextlib import nullcontext as _nullcontext


def test_benchmark():
    args = parse_args()
    print("Compile enabled:", args.compile)

    model_benchmark = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.theta,
    ).to(device=args.device)

    model = torch.compile(model_benchmark) if args.compile else model_benchmark

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    forOnly_elapsed = []
    forBack_elapsed = []
    forBackOptim_elapsed = []

    with nvtx.range("warmup") if args.device == "cuda" else _nullcontext():
        for _ in range(args.warmup_iters):
            train(args, model, optimizer)

    with nvtx.range("measured_runs") if args.device == "cuda" else _nullcontext():
        for _ in range(args.max_iters):
            elapsed1, elapsed2, elapsed3 = train(args, model, optimizer)
            forOnly_elapsed.append(elapsed1)
            forBack_elapsed.append(elapsed2)
            forBackOptim_elapsed.append(elapsed3)

    print(
        "Mean: \n"
        f"ForOnly: {sum(forOnly_elapsed)/len(forOnly_elapsed):.3f}s, "
        f"ForBack: {sum(forBack_elapsed)/len(forBack_elapsed):.3f}s, "
        f"ForBackOptim: {sum(forBackOptim_elapsed)/len(forBackOptim_elapsed):.3f}s"
    )
    print(
        "Std: \n"
        f"ForOnly: {pstdev(forOnly_elapsed):.3f}s, "
        f"ForBack: {pstdev(forBack_elapsed):.3f}s , "
        f"ForBackOptim: {pstdev(forBackOptim_elapsed):.3f}s"
    )

    activities = [ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print("\n=== PyTorch Profiler Results ===")
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(3):
            train(args, model, optimizer)

    sort_key = "cuda_time_total" if args.device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))


if __name__ == "__main__":
    print(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    test_benchmark()
