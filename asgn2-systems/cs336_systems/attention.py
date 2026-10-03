import torch
import timeit
from cs336_basics.model import MultiheadSelfAttention

batch_size = 8
d_models = [16, 32, 64, 128]
seq_lens = [256, 1024, 4096, 8192, 16384]
warm_up_steps = 5
measure_steps = 100

def benchmark_attention(x:torch.Tensor, model:MultiheadSelfAttention):
    # 预热阶段
    try:
        for _ in range(warm_up_steps):
            model.zero_grad()
            x.grad = None
            logits = model(x)
            logits.sum().backward()
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM]{e}：预热阶段显存不足！")
            torch.cuda.empty_cache()
        else:
            raise e
            

    fwd_times = []
    fwd_mems = []
    bwd_times = []
    bwd_mems = []

    # 前向部分
    try:
        for _ in range(measure_steps):
            torch.cuda.synchronize()

            t0 = timeit.default_timer()
            logits = model(x)
            torch.cuda.synchronize()
            t1 = timeit.default_timer()

            # torch.cuda.memory_allocated()：返回当前时刻，所有活着的 PyTorch 张量（参数、梯度、激活值、logits）占用的总字节数（记录当前的真实显存占用）。
            fwd_mem = torch.cuda.memory_allocated(device = "cuda") / (1024)**2

            # # 测量峰值显存的情况
            # torch.cuda.reset_peak_memory_stats()        # 在计时开始前，重置峰值统计
            # ...(上面的 t0-t1 计时部分)
            # fwd_mem = torch.cuda.max_memory_allocated(device="cuda") / (1024)**2


            fwd_times.append(t1-t0)
            fwd_mems.append(fwd_mem)
            del logits      # 删除输出和挂在输出上的计算图
        torch.cuda.empty_cache()        # 手动要求 PyTorch 把显存池里没被使用的显存全部还给系统，很费时间

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM]{e}：d_model={x.shape[-1]}, seq_len={x.shape[-2]} 前向传播阶段显存不足！")
            torch.cuda.empty_cache()
            return
        else:
            raise e

    # 反向部分
    try:
        for _ in range(measure_steps):
            # 前向部分不用清空梯度，反向部分产生梯度才需要每次循环时清空梯度
            model.zero_grad(set_to_none = True)
            x.grad = None

            logits = model(x)
            torch.cuda.synchronize()

            t2 = timeit.default_timer()
            logits.sum().backward()
            torch.cuda.synchronize()
            t3 = timeit.default_timer()

            bwd_mem = torch.cuda.memory_allocated(device = "cuda") / (1024)**2
            bwd_times.append(t3-t2)
            bwd_mems.append(bwd_mem)
            del logits
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM]{e}：d_model={x.shape[-1]}, seq_len={x.shape[-2]} 反向传播阶段显存不足！")
            torch.cuda.empty_cache()
            return
        else:
            raise e

    fwd_mean_time = sum(fwd_times) / len(fwd_times)
    fwd_std_time = (sum((t - fwd_mean_time) ** 2 for t in fwd_times) / len(fwd_times)) ** 0.5
    bwd_mean_time = sum(bwd_times) / len(bwd_times)
    bwd_std_time = (sum((t - bwd_mean_time) ** 2 for t in bwd_times) / len(bwd_times)) ** 0.5

    print(f"显存占用: 前向部分：{sum(fwd_mems)/len(fwd_mems):.2f} MB，反向部分：{sum(bwd_mems)/len(bwd_mems):.2f} MB")
    print(f"fwd_Total: {sum(fwd_times):.4f}, fwd_Mean: {fwd_mean_time:.4f}s, fwd_Std: {fwd_std_time:.4f}s")
    print(f"bwd_Total: {sum(bwd_times):.4f}, bwd_Mean: {bwd_mean_time:.4f}s, bwd_Std: {bwd_std_time:.4f}s")
    

if __name__ == "__main__":
    for d_model in d_models:
        for seq_len in seq_lens:
            try:
                print(f"\nTesting d_model={d_model}, seq_len={seq_len}")
                x = torch.randn((batch_size, seq_len, d_model), device = "cuda", requires_grad = True)
                # 本处也可以直接把 x 的梯度按钮关闭，因为该脚本（以及参数训练优化时同样）只需要模型的梯度而不需要数据的梯度（只有在输入数据本身成为我们需要优化或分析的对象时才需要）
                model = MultiheadSelfAttention(d_model, 1, seq_len).cuda()
                compiled_model = torch.compile(model)
                
            except Exception as e:
                print(f"Setup Failed: {e}")
                torch.cuda.empty_cache()
                continue

            benchmark_attention(x, model)
            benchmark_attention(x, compiled_model)
            del x, model, compiled_model
            torch.cuda.empty_cache()