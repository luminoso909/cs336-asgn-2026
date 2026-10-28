import math
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import Embedding, Linear


class ZeRO1(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs,
    ):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.optimizer_cls = optimizer_cls

        # 每个 Rank 都维护一个负载记录，由于贪心算法是有确定性的，所以每个 Rank 的记录都相同
        # loads: 每个 Rank 当前分配了多少参数元素; owners: 各个参数所属的 Rank
        self.loads = [0] * self.world_size
        self.owners = {}

        # 防止基类初始化期间提前执行分片逻辑
        # True：父类正在初始化，只登记参数组，先不要分片
        # False：初始化完了，以后新增参数组需要真正执行分片
        self._initializing = True
        super().__init__(params, defaults=kwargs)
        self._initializing = False

        local_groups = []
        for group in self.param_groups:
            local_params = []
            for param in group["params"]:

                owner = min(
                    range(self.world_size),
                    key = lambda r: self.loads[r]
                )
                self.owners[param] = owner
                self.loads[owner] += param.numel()

                if self.owners[param] == self.rank:
                    local_params.append(param)
                    
            if local_params:
                local_group = {
                    **group,
                    "params": local_params,
                }

                local_groups.append(local_group)

        self.optimizer_kwargs = kwargs

        if local_groups:
            self.local_optimizer = optimizer_cls(
                local_groups,
                **kwargs,
            )
        else:
            self.local_optimizer = None

    def step(self, closure=None, **kwargs):
        result = None

        if self.local_optimizer is not None:
            result = self.local_optimizer.step(
                closure=closure,
                **kwargs,
            )

        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    owner = self.owners[param]
                    dist.broadcast(param, src=owner)

        return result

    def add_param_group(self, param_group):     # 此函数仅限增加一个新参数组
        super().add_param_group(param_group)

        if self._initializing:
            return

        group = self.param_groups[-1]
        local_params = []
        for param in group["params"]:
            owner = min(
                range(self.world_size),
                key=lambda r: self.loads[r],
            )

            self.owners[param] = owner
            self.loads[owner] += param.numel()

            if owner == self.rank:
                local_params.append(param)

        if not local_params:
            return

        local_group = {
            **group,
            "params": local_params,
        }       # 把新参数组复制一份，但删掉不属于我的参数，保留该参数组原来的优化器超参数。

        if self.local_optimizer is None:
            self.local_optimizer = self.optimizer_cls(
                [local_group],
                **self.optimizer_kwargs,
            )
        else:
            self.local_optimizer.add_param_group(local_group)


@dataclass(eq=False)
class _Meta:
    name: str
    layer: nn.Module
    param: nn.Parameter
    shape: torch.Size
    numel: int
    shard_size: int
    kind: str


class _ShardedOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, shard, fsdp, meta, bias):
        ctx.fsdp = fsdp
        ctx.meta = meta
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x)

        # All-Gather：恢复当前层完整权重
        weight = fsdp._gather(meta, "forward")

        if meta.kind == "linear":
            x_compute = x.to(weight.dtype)
            b = bias.to(weight.dtype) if bias is not None else None
            out = F.linear(x_compute, weight, b)

        else:
            out = F.embedding(
                x,
                weight,
                padding_idx=getattr(meta.layer, "padding_idx", None),
            )

        # 释放当前层的完整权重
        del weight

        # 当前层完成后，预取接下来两层的权重
        fsdp._prefetch(meta, "forward")

        return out

    @staticmethod
    def backward(ctx, dout):
        (x,) = ctx.saved_tensors
        fsdp = ctx.fsdp
        meta = ctx.meta

        if meta.kind == "linear":

            # Backward 重新 All-Gather 权重
            weight = fsdp._gather(meta, "backward")

            # 输入梯度
            dx = (
                dout.to(weight.dtype) @ weight
                if ctx.needs_input_grad[0]
                else None
            )

            if dx is not None:
                dx = dx.to(x.dtype)

            # 权重梯度
            x2d = x.to(dout.dtype).reshape(-1, x.shape[-1])
            g2d = dout.reshape(-1, dout.shape[-1])

            dw = g2d.T @ x2d

            # bias 梯度
            dbias = (
                g2d.sum(dim=0).float()
                if ctx.has_bias
                else None
            )

            del weight

        else:
            # Embedding 的输入是整数 token ids，不需要 dx
            dx = None
            dbias = None

            dw = torch.zeros(
                meta.shape,
                dtype=dout.dtype,
                device=dout.device,
            )

            ids = x.reshape(-1)
            grads = dout.reshape(-1, dout.shape[-1])

            padding_idx = getattr(
                meta.layer,
                "padding_idx",
                None,
            )

            if padding_idx is not None:
                mask = ids != padding_idx
                ids = ids[mask]
                grads = grads[mask]

            dw.index_add_(0, ids, grads)

        # Reduce-Scatter：只返回当前 Rank 的梯度分片
        dshard = fsdp._scatter_grad(meta, dw.float())

        # 预取反向传播即将使用的前两层
        fsdp._prefetch(meta, "backward")

        return dx, dshard, None, None, dbias


class FSDP(nn.Module):

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if not dist.is_initialized():
            raise RuntimeError(
                "Initialize distributed process group first"
            )

        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype

        self.meta = {}
        self.layers = []
        self.pending = {}
        self.hooks = []

        linear_types = (Linear, nn.Linear)
        embedding_types = (Embedding, nn.Embedding)

        sharded = set()

        # 1. 分片 Linear / Embedding 权重
        for name, layer in self.module.named_modules():

            if not isinstance(
                layer,
                linear_types + embedding_types,
            ):
                continue

            weight = layer.weight

            if not isinstance(weight, nn.Parameter):
                raise TypeError(
                    f"{name}.weight must be an nn.Parameter"
                )

            # 目前只支持 Embedding 的常规稠密梯度
            if isinstance(layer, embedding_types):
                if (
                    getattr(layer, "sparse", False)
                    or getattr(layer, "scale_grad_by_freq", False)
                    or getattr(layer, "max_norm", None) is not None
                ):
                    raise NotImplementedError(
                        "Embedding sparse/scale_grad_by_freq/"
                        "max_norm are not supported"
                    )

            with torch.no_grad():

                # 所有 Rank 从相同的初始权重开始
                dist.broadcast(weight.data, src=0)

                shape = weight.shape
                numel = weight.numel()

                shard_size = (
                    numel + self.world_size - 1
                ) // self.world_size

                # 展平、补零
                flat = F.pad(
                    weight.detach().float().flatten(),
                    (
                        0,
                        shard_size * self.world_size - numel,
                    ),
                )

                start = self.rank * shard_size
                end = start + shard_size

                # FP32 Master Shard，保留 Parameter 身份
                weight.data = flat[start:end].clone()

            kind = (
                "linear"
                if isinstance(layer, linear_types)
                else "embedding"
            )

            key = f"{name}.weight" if name else "weight"

            meta = _Meta(
                name=key,
                layer=layer,
                param=weight,
                shape=shape,
                numel=numel,
                shard_size=shard_size,
                kind=kind,
            )

            self.meta[key] = meta
            self.layers.append(meta)
            sharded.add(id(weight))

            # 用自定义 Autograd Function 替换该层计算
            def wrapped_forward(
                mod,
                x,
                _meta=meta,
                _fsdp=self,
            ):
                return _ShardedOp.apply(
                    x,
                    mod.weight,
                    _fsdp,
                    _meta,
                    getattr(mod, "bias", None),
                )

            layer.forward = types.MethodType(
                wrapped_forward,
                layer,
            )

        # 2. 小参数（RMSNorm / bias 等）保持复制
        for param in self.module.parameters():

            if id(param) in sharded:
                continue

            with torch.no_grad():
                dist.broadcast(param.data, src=0)

            if not param.requires_grad:
                continue

            def avg_hook(p):
                if p.grad is not None:
                    dist.all_reduce(
                        p.grad,
                        op=dist.ReduceOp.SUM,
                    )
                    p.grad.div_(self.world_size)

            handle = param.register_post_accumulate_grad_hook(
                avg_hook
            )

            self.hooks.append(handle)

    def _start_gather(self, meta, phase):
        key = (phase, meta.name)

        if key in self.pending:
            return

        local = (
            meta.param.detach()
            .to(self.compute_dtype or torch.float32)
            .contiguous()
        )

        recv = [
            torch.empty_like(local)
            for _ in range(self.world_size)
        ]

        # Gloo 同步；NCCL 异步
        async_op = dist.get_backend() != "gloo"

        work = dist.all_gather(
            recv,
            local,
            async_op=async_op,
        )

        # 保留通信缓冲区，直到完成
        self.pending[key] = (work, recv, local)

    def _gather(self, meta, phase):
        key = (phase, meta.name)

        self._start_gather(meta, phase)

        work, recv, local = self.pending.pop(key)

        if work is not None:
            work.wait()

        return (
            torch.cat(recv)[:meta.numel]
            .reshape(meta.shape)
        )

    def _prefetch(self, meta, phase):
        i = self.layers.index(meta)

        if phase == "forward":
            upcoming = self.layers[i + 1:i + 3]

        else:
            upcoming = list(
                reversed(
                    self.layers[max(0, i - 2):i]
                )
            )

        for next_meta in upcoming:
            self._start_gather(next_meta, phase)

    def _scatter_grad(self, meta, grad):

        padded = F.pad(
            grad.flatten(),
            (
                0,
                meta.shard_size * self.world_size - meta.numel,
            ),
        ).contiguous()

        out = torch.empty(
            meta.shard_size,
            dtype=torch.float32,
            device=grad.device,
        )

        if dist.get_backend() == "gloo":
            # CPU 兼容路径：All-Reduce + 本地切片
            dist.all_reduce(
                padded,
                op=dist.ReduceOp.SUM,
            )

            start = self.rank * meta.shard_size
            out.copy_(
                padded[start:start + meta.shard_size]
            )

        else:
            # NCCL：真正的 Reduce-Scatter
            dist.reduce_scatter_tensor(
                out,
                padded,
                op=dist.ReduceOp.SUM,
            )

        return out / self.world_size

    def _wait_all(self):
        for work, recv, local in self.pending.values():
            if work is not None:
                work.wait()

        self.pending.clear()

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)

        self._wait_all()
        return out

    def finish_gradient_synchronization(self):
        self._wait_all()

    @torch.no_grad()
    def gather_full_parameters(self):
        result = {}

        for name, param in self.module.named_parameters():

            if name in self.meta:
                meta = self.meta[name]

                recv = [
                    torch.empty_like(param)
                    for _ in range(self.world_size)
                ]

                dist.all_gather(recv, param.detach())

                result[name] = (
                    torch.cat(recv)[:meta.numel]
                    .reshape(meta.shape)
                    .clone()
                )

            else:
                result[name] = param.detach().clone()

        return result