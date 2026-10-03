import torch
import timeit
import argparse
from contextlib import nullcontext

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, cross_entropy

# 运行命令 python cs336_systems/benchmark.py --mode full

def parse_args():
    parser = argparse.ArgumentParser(description = "CS336 A2 End-to-End Benchmark")

    # 模型参数
    parser.add_argument('--vocab_size', type = int, default = 10000)
    parser.add_argument('--batch_size', type = int, default = 4)
    parser.add_argument('--context_length', type = int, default = 512)
    parser.add_argument('--d_model', type = int, default = 512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--theta', type=float, default=10000.0)

    # 优化器参数
    parser.add_argument('--lr_max', type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # benchmark 参数
    parser.add_argument('--warmup_steps', type = int, default = 5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "full"], required = True)
    # 如果 mode 参数不是 choices 里面的值，就会在 argparse 库这里直接报错

    # 混合精度开关
    parser.add_argument('--mixed_precision', action='store_true', help='Use BF16 mixed precision')
    return parser.parse_args()

def main():
    args = parse_args()
    assert torch.cuda.is_available()        # 必须采用 cuda 设备运行

    # 1 初始化模型和优化器
    model = TransformerLM(
        args.vocab_size, 
        args.context_length, 
        args.num_layers, 
        args.d_model, 
        args.num_heads, 
        args.d_ff, 
        args.theta).cuda()
    optimizer = AdamW(
        model.parameters(), 
        args.lr_max, 
        (args.beta1, args.beta2), 
        args.eps, 
        args.weight_decay)
    coompiled_model = torch.compile(model)

    # 2 初始化数据批次
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).cuda()
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).cuda()

    # 3 定义单步执行函数
    def step():
        optimizer.zero_grad(set_to_none=True)   # set_to_none：直接把 .grad 指针指向 None，即直接扔掉这个张量，省时省力省显存

        # 采用 autocast 进行混合精度运算，由于 matmul 和 Linear 一般只会在 model() 出现，所以只在这里用可以减少 autocast 导致的上下文管理性能开销
        with torch.autocast(device_type = "cuda", dtype = torch.float16):
            logits = model(x)
        
        if args.mode in ["forward_backward", "full"]:
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            if args.mode == "full":
                optimizer.step()

    # 4 warm up 步骤
    for _ in range(args.warmup_steps):
        step()
        torch.cuda.synchronize()

    # 开始记录显存历史
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # 5 正式测量
    times = []
    for _ in range(args.measure_steps):
        torch.cuda.synchronize()
        t0 = timeit.default_timer()

        step()
        
        torch.cuda.synchronize()
        t1 = timeit.default_timer()

        times.append(t1-t0)

    # 6 计算平均时间和标准差
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    print(f"Mode: {args.mode}, Mean: {mean_time:.4f}s, Std: {std_time:.4f}s")

    # 保存一个 pickle 文件，供 PyTorch 的在线工具加载
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    # 停止记录显存历史
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()