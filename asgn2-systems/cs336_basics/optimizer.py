import torch
from torch import nn
import math

from typing import Callable, Iterable, Optional


class SGD(torch.optim.Optimizer):
    def __init__(self, 
            params:Iterable[nn.Parameter], 
            lr:float = 1e-3
    ):
        if lr <= 0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr}
        super().__init__(params, defaults)

    def step(self,
            closure:Optional[Callable] = None
    ):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None: continue

                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad
                p.data -= lr / math.sqrt(t+1) * grad
                state['t'] = t+1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params:Iterable[nn.Parameter], 
        lr:float, 
        betas:tuple[float, float], 
        eps:float, 
        weight_decay:float
    ):
        defaults = {
            "lr":lr, 
            "betas":betas, 
            "eps":eps, 
            "weight_decay":weight_decay
        }
        super().__init__(params, defaults)

    def step(
        self, 
        closure:Optional[Callable] = None
    ):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            decay_rate = group["weight_decay"]

            for p in group["params"]:
                grad = p.grad
                if grad is None: continue
                
                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))

                p.data -= lr * decay_rate * p.data
                
                alpha_t = lr * math.sqrt(1-(beta2)**t) / (1-beta1**t)
                m_hat = beta1 * m + (1-beta1) * grad
                v_hat = beta2 * v + (1-beta2) * grad * grad
                p.data -= alpha_t * m_hat / (torch.sqrt(v_hat) + eps)

                state['t'] = t+1
                state['m'] = m_hat
                state['v'] = v_hat

        return loss