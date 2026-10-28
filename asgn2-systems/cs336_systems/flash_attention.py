import math
import torch
import triton
import triton.language as tl

from torch import Tensor
from jaxtyping import Float

def flash_bwd_pytorch(
    Q: Float[Tensor, "b nq d_k"],
    K: Float[Tensor, "b nk d_k"],
    V: Float[Tensor, "b nk d_v"],
    O: Float[Tensor, "b nq d_v"],
    dO: Float[Tensor, "b nq d_v"],
    L: Float[Tensor, "b nq"],
    is_causal: bool=False, 
) -> tuple[
    Float[Tensor, "b nq d_k"],
    Float[Tensor, "b nk d_k"],
    Float[Tensor, "b nk d_v"],
]:
    scale = 1.0 / math.sqrt(Q.shape[-1])

    q = Q.float()
    k = K.float()
    v = V.float()
    do = dO.float()

    D = (O.float() * do).sum(dim=-1, keepdim=True)
    S = (q @ k.transpose(-2, -1)) * scale       # 重算 S

    if is_causal:       # 类似前向的过程
        q_idx = torch.arange(Q.shape[1], device=Q.device)
        k_idx = torch.arange(K.shape[1], device=K.device)
        allowed = q_idx[:, None] >= k_idx[None, :]
        S += torch.where(allowed[None, :, :], 0.0, -1e6)

    P = torch.exp(S - L.float().unsqueeze(-1))  # 重算 P（由于有 L 所以无需调用 softmax）
    dV = P.transpose(-2, -1) @ do
    dP = do @ v.transpose(-2, -1)
    dS =  P * (dP - D)
    dQ = (dS @ k) * scale
    dK = (dS.transpose(-2, -1) @ q) * scale

    return (
        dQ.to(Q.dtype),
        dK.to(K.dtype),
        dV.to(V.dtype),
    )

compiled_flash_bwd_pytorch = torch.compile(flash_bwd_pytorch)

class BenchmarkAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        Q: Float[Tensor, "b nq d_k"],
        K: Float[Tensor, "b nk d_k"],
        V: Float[Tensor, "b nk d_v"],
        is_causal: bool = False, 
    ) -> Float[Tensor, "b nq d_v"]:
        D = Q.shape[-1]

        S = Q @ K.transpose(-2, -1) / math.sqrt(D)

        if is_causal:
            mask = torch.tril(torch.ones(S.shape, dtype=torch.bool, device=S.device))
            S = torch.masked_fill(S, ~mask, float("-inf"))
            
        P = torch.softmax(S, dim=-1)
        O = P @ V
        L = torch.logsumexp(S, dim=-1)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O
    
    @staticmethod
    def backward(
        ctx, 
        grad_output: Float[Tensor, "b nq d_v"],
    ) -> tuple[
        Float[Tensor, "b nq d_k"],
        Float[Tensor, "b nk d_k"],
        Float[Tensor, "b nk d_v"],
        None,
    ]:
        L, Q, K, V, O = ctx.saved_tensors
                
        dQ, dK, dV = compiled_flash_bwd_pytorch(
            Q, K, V, O, grad_output, L, ctx.is_causal
        )

        return dQ, dK, dV, None



class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        Q: Float[Tensor, "b nq d_k"],
        K: Float[Tensor, "b nk d_k"],
        V: Float[Tensor, "b nk d_v"],
        is_causal=False, 
    ) -> Float[Tensor, "b nq d_v"]:
        B, Nq, D = Q.shape
        Dv = V.shape[-1]
        Nk = K.shape[1]

        BQ: int = 16
        BK: int = 16

        L = torch.empty((B, Nq), device=Q.device, dtype=torch.float32)
        O = torch.empty((B, Nq, Dv), device=Q.device, dtype=Q.dtype)

        for b in range(B):
            for qs in range(0, Nq, BQ):
                qe = min(qs + BQ, Nq)
                q = Q[b, qs:qe, :]
                rows = qe - qs

                m = torch.full((rows,), float("-inf"), device=Q.device, dtype=torch.float32)
                l = torch.zeros((rows,), device=Q.device, dtype=torch.float32)
                acc = torch.zeros((rows, Dv), device=Q.device, dtype=torch.float32)

                for ks in range(0, Nk, BK):
                    ke = min(ks + BK, Nk)
                    k = K[b, ks:ke, :]
                    v = V[b, ks:ke, :]

                    s = q @ k.T / math.sqrt(D)
                    if is_causal:
                        q_idx = torch.arange(qs, qe, device=Q.device)
                        k_idx = torch.arange(ks, ke, device=Q.device)
                        allowed = q_idx[:, None] >= k_idx[None, :]
                        s = torch.where(allowed, s, -1e6)

                    tile_max = torch.max(s, dim=-1).values
                    m_old = m
                    m_new = torch.maximum(m, tile_max)
                    alpha = torch.exp(m_old - m_new)
                    p_tilde = torch.exp(s - m_new.unsqueeze(-1))
                    l = alpha * l + p_tilde.sum(dim=-1)
                    acc = alpha.unsqueeze(-1) * acc + p_tilde @ v
                    m = m_new

                L[b, qs:qe] = m + torch.log(l)
                O[b, qs:qe, :] = acc / l.unsqueeze(-1)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O
    
    @staticmethod
    def backward(
        ctx, 
        grad_output: Float[Tensor, "b nq d_v"],
    ) -> tuple[
        Float[Tensor, "b nq d_k"],
        Float[Tensor, "b nk d_k"],
        Float[Tensor, "b nk d_v"],
        None,
    ]:
        L, Q, K, V, O = ctx.saved_tensors
                
        dQ, dK, dV = compiled_flash_bwd_pytorch(
            Q, K, V, O, grad_output, L, ctx.is_causal
        )

        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D_KEY: tl.constexpr,
    D_VALUE: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_KEY),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_KEY),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_KEY),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_KEY),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb, 
        shape=(N_KEYS, D_VALUE), 
        strides=(stride_vk, stride_vd), 
        offsets=(0, 0), 
        block_shape=(K_TILE_SIZE, D_VALUE), 
        order=(1, 0), 
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D_VALUE),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_VALUE),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q_indices = (query_tile_index * Q_TILE_SIZE+ tl.arange(0, Q_TILE_SIZE))
    m = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D_VALUE), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(
            K_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        v = tl.load(
            V_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        s = tl.dot(
            q, tl.trans(k),
            input_precision="ieee",
        ) * scale

        key_indices = (
            j * K_TILE_SIZE 
            + tl.arange(0, K_TILE_SIZE)
        )
        
        if IS_CAUSAL:
            # 经过广播，得到逐元素对比的 (tile_q, tile_k) 型 bool 矩阵（再不存在尾部截断的情况下一般是方阵）
            allowed = q_indices[:, None] >= key_indices[None, :]
            s = s + tl.where(
                allowed,
                0.0,
                -1e6,       # 如果采用 -inf 可能导致更新 alpha 的时候存在 m_old-m_new 结果为 NaN 的情况从而数据污染
            )

        s = tl.where(
            key_indices[None, :] < N_KEYS, 
            s, 
            -float("inf")
        )

        # —————————————————————— Core Part —————————————————————————— #
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m-m_new)
        p_tilde = tl.exp(s-m_new[:, None])
        l = alpha * l + tl.sum(p_tilde, axis=1)
        acc = alpha[:, None] * acc + tl.dot(
            p_tilde.to(v.dtype), 
            v, 
            input_precision="ieee",
        )
        # "ieee": 使用常规 FP32 精度的矩阵运算路径
        # "tf32": 利用 TF32 精度的 Tensor Core 路径，通常更快但误差更大
        # "tf32x3": 用多次 TF32 相关运算提高 FP32 结果精度
        m = m_new
        # —————————————————————— Core Part —————————————————————————— #

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    o = acc / l[:, None]
    lse = m + tl.log(l)

    tl.store(
        O_block_ptr,
        o.to(O_ptr.type.element_ty),
        boundary_check=(0, 1),
    )

    tl.store(
        L_block_ptr,
        lse.to(L_ptr.type.element_ty),
        boundary_check=(0,),
    )

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "b nq d_k"],
        K: Float[Tensor, "b nk d_k"],
        V: Float[Tensor, "b nk d_v"],
        is_causal=False, 
    ) -> Float[Tensor, "b nq d_v"]:
        B, Nq, Dk = Q.shape
        Dv = V.shape[-1]
        Nk = K.shape[1]

        BQ = 16
        BK = 16

        O = torch.empty(
            (B, Nq, Dv),
            device=Q.device,
            dtype=Q.dtype
        )

        L = torch.empty(
            (B, Nq),
            device=Q.device,
            dtype=torch.float32,
        )
        grid = (triton.cdiv(Nq, BQ), B)
        # triton.cdiv 用于普通 python 代码，计算 grid 大小
        # tl.cdiv 用于 @triton.jit 内部，计算 K/V Tile 的循环次数

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,

            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *L.stride(),

            Nq, Nk,
            1.0 / math.sqrt(Dk),

            Dk,
            Dv,
            BQ,
            BK,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal

        return O


    @staticmethod
    def backward(
        ctx, 
        grad_output: Float[Tensor, "b nq d_v"],
    ) -> tuple[
        Float[Tensor, "b nq d_k"],
        Float[Tensor, "b nk d_k"],
        Float[Tensor, "b nk d_v"],
        None,
    ]:
        L, Q, K, V, O = ctx.saved_tensors
                
        dQ, dK, dV = compiled_flash_bwd_pytorch(
            Q, K, V, O, grad_output, L, ctx.is_causal
        )

        return dQ, dK, dV, None
    
