import torch
import math

from torch import Tensor
from jaxtyping import Float, Int
from typing import Iterable

# functions
def softmax(x: torch.Tensor, dim: int = -1):
    max_per_dim = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_per_dim)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def silu(x:torch.Tensor):
    return x * torch.sigmoid(x)

# loss evaluation
def cross_entropy(
    logits: Float[Tensor, "... vocab_size"], 
    targets: Int[Tensor, "..."], 
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    # 或者如下方式，也可以避免 softmax 时数值 probs 下溢导致概率化 0 的错误
    # x = logits - logits.max(dim=-1, keepdim=True).values
    # log_probs = x - torch.log(torch.exp(x).sum(dim=-1, keepdim=True))
    target_log_probs = torch.gather(log_probs, dim = -1, index = targets.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()

# Learning rate scheduling
def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        a_t = it * max_learning_rate / warmup_iters
    elif it > cosine_cycle_iters:
        a_t = min_learning_rate
    else:
        a_t = min_learning_rate + 0.5 * (
            1 + math.cos( (it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters) )
        ) * (max_learning_rate - min_learning_rate)
    return a_t

# Gradient Clipping
def gradient_clipping(
    params: Iterable[torch.nn.Parameter], 
    max_l2_norm: float, 
    eps: float = 1e-6, 
):
    params_list = [p for p in params if p.grad is not None]
    if not params:
        return
    grad = 0.0
    for p in params_list:
        grad += torch.sum(torch.pow(p.grad.data, 2))

    l2_norm = torch.sqrt(grad)

    if l2_norm >= max_l2_norm:
        scaled_factor = max_l2_norm / (l2_norm + eps)
        for p in params_list:
            p.grad.mul_(scaled_factor)

# Temperature Scaling
def temperature_scaling(x:torch.Tensor, temper:float, dim:int):
    if temper <= 0:
        raise ValueError("temperature must be positive")
    return softmax(x/temper, dim=dim)

# Top_p Sampling
def top_p_selected(probs:torch.Tensor, top_p:float):
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove_mask = cumulative_probs - sorted_probs >= top_p
    sorted_probs.masked_fill_(remove_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return sorted_probs, sorted_indices


    