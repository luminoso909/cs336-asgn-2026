"""Learning baseline: parameter-sharded Linear/Embedding FSDP (no prefetch yet).

Supports CS336 Linear/Embedding, standard bias-free nn.Linear, and default nn.Embedding.
Not final submission: prefetch and asynchronous communication still need implementing.
"""
from __future__ import annotations
import math
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class _Meta:
    param: nn.Parameter
    shape: torch.Size
    numel: int
    shard_numel: int
    kind: str

class _ShardedOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, owner, meta, bias):
        ctx.owner = owner
        ctx.meta = meta
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, shard)
        weight = owner._gather_weight(meta)
        if meta.kind == "linear":
            out = F.linear(x, weight, bias.to(x.dtype) if bias is not None else None)
        else:
            out = F.embedding(x, weight)
        # ``weight`` not saved: its memory is released after this function returns.
        return out

    @staticmethod
    def backward(ctx, dout):
        x, shard = ctx.saved_tensors
        owner, meta = ctx.owner, ctx.meta
        weight = owner._gather_weight(meta) if meta.kind == "linear" else None
        if meta.kind == "linear":
            dx = dout @ weight if ctx.needs_input_grad[0] else None
            x2d = x.reshape(-1, x.shape[-1])
            g2d = dout.reshape(-1, dout.shape[-1])
            dw = g2d.transpose(0, 1) @ x2d
            dbias = g2d.sum(dim=0) if ctx.has_bias else None
        else:
            dx = None
            dw = torch.zeros(meta.shape, dtype=dout.dtype, device=dout.device)
            dw.index_add_(0, x.reshape(-1), dout.reshape(-1, dout.shape[-1]))
            dbias = None
        dshard = owner._reduce_scatter_grad(meta, dw.to(torch.float32))
        return dx, dshard, None, None, dbias

class FSDP(nn.Module):
    """Restricted educational FSDP: cs336 Linear/Embedding and bias-free nn.Linear.

    This is an eager, synchronous correctness baseline. It does NOT yet implement
    the assignment's required asynchronous prefetch or overlapped communication.
    """
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Initialize distributed process group first")
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype
        self._metas: dict[str, _Meta] = {}
        self._replicated_hook_handles = []

        from cs336_basics.model import Linear, Embedding
        self._linear_types = (Linear, nn.Linear)
        self._embedding_types = (Embedding, nn.Embedding)
        for name, child in self.module.named_modules():
            if isinstance(child, self._linear_types + self._embedding_types):
                if isinstance(child, nn.Linear) and child.bias is not None:
                    raise NotImplementedError("This learning version supports bias-free Linear only")
                param = child.weight
                if not param.requires_grad:
                    continue
                with torch.no_grad():
                    dist.broadcast(param.data, src=0)
                    original_shape = param.data.shape
                    full_flat = param.data.float().reshape(-1)
                    true_n = full_flat.numel()
                    shard_n = math.ceil(true_n / self.world_size)
                    padded = torch.zeros(shard_n * self.world_size, device=param.device, dtype=torch.float32)
                    padded[:true_n] = full_flat
                    param.data = padded[self.rank*shard_n:(self.rank+1)*shard_n].clone()
                kind = "linear" if isinstance(child, self._linear_types) else "embedding"
                meta = _Meta(param, original_shape, true_n, shard_n, kind)
                param_name = f"{name}.weight" if name else "weight"
                self._metas[param_name] = meta
                def new_forward(mod, x, _owner=self, _meta=meta):
                    return _ShardedOp.apply(x, mod.weight, _owner, _meta, getattr(mod, 'bias', None))
                child.forward = types.MethodType(new_forward, child)
            else:
                for param in child.parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    with torch.no_grad():
                        dist.broadcast(param.data, src=0)
                    def avg_hook(p):
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.div_(self.world_size)
                    self._replicated_hook_handles.append(param.register_post_accumulate_grad_hook(avg_hook))

    def _gather_weight(self, meta: _Meta) -> torch.Tensor:
        local = meta.param.detach().to(self.compute_dtype or meta.param.dtype).contiguous()
        all_shards = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(all_shards, local)
        return torch.cat(all_shards)[:meta.numel].reshape(meta.shape)

    def _reduce_scatter_grad(self, meta: _Meta, grad: torch.Tensor) -> torch.Tensor:
        flat = grad.reshape(-1)
        padded = torch.zeros(meta.shard_numel * self.world_size, dtype=torch.float32, device=flat.device)
        padded[:meta.numel] = flat
        out = torch.empty(meta.shard_numel, dtype=torch.float32, device=flat.device)
        try:
            dist.reduce_scatter_tensor(out, padded, op=dist.ReduceOp.SUM)
        except RuntimeError as e:
            if dist.get_backend() != 'gloo' or 'support' not in str(e).lower():
                raise
            dist.all_reduce(padded, op=dist.ReduceOp.SUM)
            out.copy_(padded[self.rank*meta.shard_numel:(self.rank+1)*meta.shard_numel])
        return out / self.world_size

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # Synchronous teaching implementation, no pending handles yet.
        pass

    @torch.no_grad()
    def gather_full_parameters(self):
        result = {}
        for name, p in self.module.named_parameters():
            if name in self._metas:
                meta = self._metas[name]
                shards = [torch.empty_like(p) for _ in range(self.world_size)]
                dist.all_gather(shards, p)
                result[name] = torch.cat(shards)[:meta.numel].reshape(meta.shape).clone()
            else:
                result[name] = p.detach().clone()
        return result
