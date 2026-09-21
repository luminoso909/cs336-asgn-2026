from einops import einsum, rearrange, reduce
import torch
from torch import nn
import math

from typing import Callable, Iterable, Optional
from jaxtyping import Bool, Float, Int
from torch import Tensor


def cross_entropy(
        input:Float[Tensor, "batch_size vocab_size"], 
        targets:Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    max_input = input.max(dim = -1, keepdim = True).values      # 或用 torch.amax() 不返回下标，只返回最大值
    stable_input = input - max_input
    log_sum_exp = max_input + torch.log(torch.sum(torch.exp(stable_input), dim = -1, keepdim = True))
    correct_logits = input[torch.arange(targets.shape[0]), targets]
    # 或者用 gather 方法：correct_logits = input.gather(1, rearrange(targets, "batch_size -> batch_size 1"))
    loss = log_sum_exp - correct_logits
    return torch.mean(loss)


class SGD(torch.optim.Optimizer):
    def __init__(self, 
            params:Iterable[nn.Parameter], 
            lr:float = 1e-3
    ):
        if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)      # 自动把 params 转为 param_groups

    def step(self,
            closure:Optional[Callable] = None
    ):
        loss = None if closure is None else closure()

        # 每个 group 是一个字典：{'params': [param1, param2, ...], lr': 1e-3,}
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state['t'] = t + 1

        return loss

# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1)

# for t in range(5000):
#     opt.zero_grad() # 重置所有可学习参数的梯度。
#     loss = (weights**2).mean() # 计算标量损失值。
#     print(loss.cpu().item())
#     loss.backward() # 运行反向传播，计算梯度。
#     opt.step() # 运行优化器步进。


class AdamW(torch.optim.Optimizer):
    def __init__(self, 
            params:Iterable[nn.Parameter] | Iterable[nn.Parameter], 
            lr:float = 1e-3, 
            betas:tuple[float, float] = (0.9, 0.999), 
            eps:float = 1e-8, 
            weight_decay:float = 0.01
    ):
        if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
        # defaults 是优化器的默认超参数配置（Python 字典），告诉 PyTorch：这个优化器的超参数和对应默认值
        defaults = {"lr":lr, "betas":betas, "eps":eps, "weight_decay":weight_decay}
        super().__init__(params, defaults)

    def step(self, closure:Optional[Callable] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None: continue

                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # 1. 计算偏差校正后的学习率 (对应算法第7行)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # 2. 应用权重衰减 (对应算法第8行)
                p.data -= lr * weight_decay * p.data

                # 3. 更新一阶和二阶矩 (对应算法第9, 10行)：原地运算，避免制造多余的垃圾内存
                exp_avg.mul_(beta1).add_(grad, alpha = 1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 4. 更新 Parameter 参数 (对应算法第11行)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data -= alpha_t * exp_avg / denom

        return loss

def learning_rate_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        cosine_param = math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (1 + cosine_param) / 2
    else:
        return min_learning_rate

def gradient_clipping(parameters:Iterable[torch.nn.Parameter], max_l2_norm:float):
    eps = 1e-6

    # 合并所有梯度拼成一个长向量 g
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads: return

    total_l2_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))

    if total_l2_norm >= max_l2_norm:
        for g in grads:
            g.mul_(max_l2_norm / (total_l2_norm + eps))
        