from bpe import BPE
from tokenizer import Tokenizer
from model import TransformerLM
from optimizer import cross_entropy, SGD, AdamW, learning_rate_schedule, gradient_clipping

from data_loading import get_batch
from checkpoint import save_checkpoint, load_checkpoint

import torch
import argparse
import numpy as np
import os
import math

from numpy.typing import NDArray

def parse_args():
    parser = argparse.ArgumentParser(description = "Training a CS336 A1 Transformer LM")
    # 使用 .add_argument(“--参数名”, type, required:bool（required 表示必须传，否则报错）, default:Any)
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # 数据与分词器
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)

    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--theta', type=float, default=10000.0)

    # 优化器参数
    parser.add_argument('--lr_max', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=1e-5, help="Min learning rate for cosine decay")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # 训练调度
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000)

    # 日志参数
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')

    return parser.parse_args()      # 直接变成一个 namespace 对象，类似字典，只不过参数自动变成 arg.属性


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1 训练数据：内存高效加载
    # 采用 mmap_mode = 'r' 表示返回一个 np.memmap 只读对象，只有访问的部分才会被载入内存，从而保证内存占用固定
    train_data = np.load(args.train_data, mmap_mode = 'r')
    val_data = np.load(args.val_data, mmap_mode = 'r')

    # 2 初始化模型、优化器、调度器
    model = TransformerLM(
        vocab_size = args.vocab_size, 
        context_length = args.context_length, 
        num_layers = args.num_layers, 
        d_model = args.d_model, 
        num_heads = args.num_heads, 
        d_ff = args.d_ff, 
        theta = args.theta
    ).to(device)

    optimizer = AdamW(
        model.parameters(), 
        lr = args.lr_max, 
        betas = (args.beta1, args.beta2), 
        eps = args.eps, 
        weight_decay = args.weight_decay)


    start_iter = 1
    if os.path.exists(os.path.join(args.checkpoint_path, 'latest.pt')):
        start_iter, start_loss = load_checkpoint(os.path.join(args.checkpoint_path, 'latest.pt'), model, optimizer)
        print(f"Resumed from iteration {start_iter}, loss {start_loss}")

    # 3 训练循环
    model.train()
    for iteration in range(start_iter, args.max_iters+1):
        lr = learning_rate_schedule(iteration, args.lr_max, args.lr_min, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        # 获取数据
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        # 前向传播
        logits = model(inputs)
        # 计算损失：注意这里的输入形状和 cross_entropy 函数的要求不同：此处 logits(batch_size, seq_len, vocab_size)，targets(batch_size, seq_len)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        gradient_clipping(model.parameters(), args.grad_clip)
        # 优化器步进
        optimizer.step()


        # 日志记录
        if iteration % args.log_interval == 0:
            print(f"Iter {iteration}: Loss {loss.item():.4f}, LR {lr:.2e}")

        # 验证与保存
        if iteration % args.val_interval == 0 and iteration > 0:
            val_loss = evaluate(model, val_data, args, device)
            print(f"Iter {iteration}: Val Loss {val_loss:.4f}, Val PPL {math.exp(val_loss):.2f}")
            model.train()

        if iteration % args.save_interval == 0 and iteration > 0:
            save_checkpoint(model, optimizer, iteration, os.path.join(args.checkpoint_path, f"save_{iteration}.pt"), loss = loss.item())
            save_checkpoint(model, optimizer, iteration, os.path.join(args.checkpoint_path, "latest.pt"), loss = loss.item())


@torch.no_grad()
def evaluate(model:torch.nn.Module, val_data:NDArray, args:argparse, device:str):
    args = parse_args()
    model.eval()
    total_loss = 0
    num_batches = 10 # 每次验证只算几个batch，避免耗时过长
    for _ in range(num_batches):
        inputs, targets = get_batch(val_data, args.batch_size, args.context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
    return total_loss / num_batches


if __name__ == "__main__":
    main()