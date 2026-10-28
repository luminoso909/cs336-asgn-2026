import torch
from torch import nn, Tensor

import torch.distributed as dist
from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module        # module 是已经初始化好输入的模型
        self.world_size = dist.get_world_size()

        # state_dict() 返回模型的状态字典，默认浅拷贝
        for param in self.module.state_dict().values():
            dist.broadcast(param, src=0)

    def forward(self, *args, **kwargs):
        # *args 接收位置参数，**kwargs 接收关键字参数：在这里是用于封装通用模型包装器，包括参数的收集和展开
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.grad is None:
                continue

            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)   # all_reduce 为原地修改
            param.grad.div_(self.world_size)


class FlatDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        for param in self.module.state_dict().values():
            dist.broadcast(param, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        grads = [
            p.grad
            for p in self.module.parameters()
            if p.grad is not None
        ]

        flat: Tensor = _flatten_dense_tensors(grads)

        dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=False)   # all_reduce 为原地修改
        flat.div_(self.world_size)

        new_grads = _unflatten_dense_tensors(flat, grads)

        for old, new in zip(grads, new_grads):
            old.copy_(new)


class AsyncDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        for param in self.module.state_dict().values():
            dist.broadcast(param, src=0)

        self.handles = []
        self.hook_handles = []

        for p in self.module.parameters():
            if p.requires_grad:
                h = p.register_post_accumulate_grad_hook(
                    self._on_grad_ready
                )
                self.hook_handles.append(h)

    def _on_grad_ready(self, param):

        if param.grad is None:
            return

        # 先除以 world_size
        param.grad.div_(self.world_size)

        # 异步启动 All-Reduce
        handle = dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )

        self.handles.append(handle)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        self.handles.clear()