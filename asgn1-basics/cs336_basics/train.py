import torch
import argparse
import numpy as np
import os
import math
import pickle
from pathlib import Path
from datetime import datetime

from numpy.typing import NDArray
import time
from torch.utils.tensorboard import SummaryWriter

from .configs import parse_args
from .tokenizer import Tokenizer
from .model import TransformerLM
from .optimizer import AdamW
from .utils import cross_entropy, learning_rate_schedule, gradient_clipping
from .data import get_batch, save_checkpoint, load_checkpoint

def prepare_data(args, special_tokens):
    train_txt = Path(args.train_data)
    valid_txt = Path(args.valid_data)

    # 自动生成 .npy 输出路径
    train_npy = train_txt.with_name(
        f"{train_txt.stem}_{args.vocab_size}.npy"
    )
    valid_npy = valid_txt.with_name(
        f"{valid_txt.stem}_{args.vocab_size}.npy"
    )

    # 加载 BPE 词表
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    with open(args.merges_path, "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(
        vocab,
        merges,
        special_tokens
    )

    assert len(tokenizer.vocab) == args.vocab_size

    # 首次运行时自动编码
    if not train_npy.exists():
        print("开始编码训练集...", flush=True)
        tokenize_file(tokenizer, str(train_txt), train_npy)

    if not valid_npy.exists():
        print("开始编码验证集...", flush=True)
        tokenize_file(tokenizer, str(valid_txt), valid_npy)

    # 训练阶段加载 token IDs
    train_data = np.load(train_npy, mmap_mode="r")
    valid_data = np.load(valid_npy, mmap_mode="r")

    print(f"Train: {train_data.shape}")
    print(f"Valid: {valid_data.shape}")

    return train_data, valid_data

def tokenize_file(
    tokenizer: Tokenizer,
    input_path: str,
    output_path: Path,
):
    special_token = "<|endoftext|>"
    temp_path = output_path.with_suffix(".bin")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    token_count = 0

    # 第一阶段：将 token IDs 流式写入二进制文件
    with open(input_path, "r", encoding="utf-8") as f, \
         open(temp_path, "wb") as out:

        buffer = ""

        for line in f:
            buffer += line

            # 只在完整特殊 Token 边界编码
            while special_token in buffer:
                document, buffer = buffer.split(
                    special_token, 1
                )

                ids = tokenizer.encode(
                    document + special_token
                )

                array = np.asarray(ids, dtype=np.uint16)
                array.tofile(out)
                token_count += len(ids)

        if buffer:
            ids = tokenizer.encode(buffer)
            array = np.asarray(ids, dtype=np.uint16)
            array.tofile(out)
            token_count += len(ids)

    # 第二阶段：转换为 .npy
    arr = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(token_count,)
    )

    raw = np.memmap(
        temp_path,
        mode="r",
        dtype=np.uint16,
        shape=(token_count,)
    )

    chunk_size = 1_000_000

    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        arr[start:end] = raw[start:end]

    arr.flush()

    del arr, raw
    temp_path.unlink()

    print(
        f"Tokenization completed: {input_path}, "
        f"{token_count:,} tokens"
    )

# ———————————————————————————————————————————————————————————————————————————————————————————— #
def train_model(args):
    if not 0 <= args.warmup_iters < args.max_iters:
        raise ValueError(
            "需要满足 0 <= warmup_iters < max_iters, "
            f"当前为 {args.warmup_iters}, {args.max_iters}"
        )

    # set ramdom seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare checkpoint directory
    os.makedirs(args.checkpoint_path, exist_ok=True)

    device = torch.device(args.device)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    writer = SummaryWriter(
        log_dir=f"runs/tinystories_{run_name}"
    )

    # prepare data, model and optimizer
    with open(args.special_tokens_path, "rb") as f:
        special_tokens = pickle.load(f)
    train_data, valid_data = prepare_data(args, special_tokens)

    model = TransformerLM(
        vocab_size = args.vocab_size, 
        context_length = args.context_length, 
        d_model = args.d_model, 
        num_layers = args.num_layers, 
        num_heads = args.num_heads, 
        d_ff = args.d_ff, 
        rope_theta = args.theta
    ).to(device)

    optimizer = AdamW(
        model.parameters(), 
        lr = args.lr_max, 
        betas = (args.beta1, args.beta2), 
        eps = args.eps, 
        weight_decay = args.weight_decay
    )

    start_iter = 1
    if args.resume is not None:
        completed_iter, start_loss = load_checkpoint(
            args.resume, 
            model, 
            optimizer, 
        )
        start_iter = completed_iter + 1
        print(f"Resumed from iteration {completed_iter}, loss {start_loss}, continue from iteration {start_iter}")

    model.train()
    start_time = time.perf_counter()
    last_log_time = time.perf_counter()
    for it in range(start_iter, args.max_iters+1):
        # update new learning rate
        lr = learning_rate_schedule(
            it, 
            args.lr_max, 
            args.lr_min, 
            args.warmup_iters, 
            args.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ———————————————————————————————————↓ core process ↓———————————————————————————————————— #
        # get data
        inputs, targets = get_batch(
            train_data, 
            args.batch_size, 
            args.context_length, 
            device
        )

        # training process: forward -> loss -> backward -> grad_clip -> optim_step
        logits = model(inputs)
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip_max_norm)
        optimizer.step()

        # ———————————————————————————————————↑ core process ↑———————————————————————————————————— #

        # print log
        if it % args.log_interval == 0:
            elapsed = time.perf_counter() - start_time

            writer.add_scalar("train/loss", loss.item(), it)
            writer.add_scalar("train/loss_vs_time", loss.item(), elapsed)
            writer.add_scalar("train/lr", lr, it)

            tokens_processed = (args.batch_size * args.context_length * args.log_interval)
            tokens_per_sec = tokens_processed / (time.perf_counter() - last_log_time)
            writer.add_scalar("train/tokens_per_sec", tokens_per_sec, it)
            last_log_time = time.perf_counter()

        # evaluate loss
        if it % args.val_interval == 0 and it > 0:
            val_loss = evaluate(model, valid_data, args, device)
            writer.add_scalar("valid/loss", val_loss, it)
            writer.add_scalar(
                "valid/loss_vs_time",
                val_loss,
                time.perf_counter() - start_time
            )
            print(f"Iter {it}: Val Loss {val_loss:.4f}, Val PPL {math.exp(val_loss):.2f}")
            model.train()

        # checkpoint
        if it % args.save_interval == 0 and it > 0:
            save_checkpoint(
                os.path.join(args.checkpoint_path, f"save_{it}.pt"), 
                model, 
                optimizer, 
                it, 
                loss = loss.item()
            )
            save_checkpoint(
                os.path.join(args.checkpoint_path, "latest.pt"),         
                model, 
                optimizer, 
                it, 
                loss = loss.item()
            )

    writer.flush()

    if start_iter <= args.max_iters:
        save_checkpoint(
            os.path.join(args.checkpoint_path, "latest.pt"),
            model,
            optimizer,
            it,
            loss=loss.item()
        )

    writer.close()

@torch.no_grad()
def evaluate(model:torch.nn.Module, val_data:NDArray, args:argparse, device:torch.device | str):
    model.eval()
    total_loss = 0
    num_batches = 10
    for _ in range(num_batches):
        inputs, targets = get_batch(val_data, args.batch_size, args.context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    return total_loss / num_batches
# ———————————————————————————————————————————————————————————————————————————————————————————— #


if __name__ == "__main__":
    args = parse_args()
    train_model(args)