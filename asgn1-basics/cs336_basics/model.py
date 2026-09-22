from einops import einsum, rearrange, reduce
import torch
from torch import nn
import math

from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, 
            in_features:int, 
            out_features:int, 
            device:torch.device | None = None, 
            dtype:torch.dtype | None = None, 
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype))

        # 作业要求采用 Linear weights: N(μ=0, σ²=2/(d_in+d_out)) truncated at [-3σ, 3σ]
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean = 0.0, std = std, a = -3*std, b = 3*std)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')

class Embedding(nn.Module):
    def __init__(self,
            num_embeddings:int,         # 词汇表大小
            embedding_dim:int,          # 嵌入向量的维度
            device:torch.device | None = None, 
            dtype:torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), device = device, dtype = dtype))

        # 作业要求采用 Embedding: N(μ=0, σ²=1) truncated at [-3, 3]
        torch.nn.init.trunc_normal_(self.embeddings, mean = 0.0, std = 1, a = -3.0, b = 3.0)

    # 相比输入张量，输出是直接在其后面加上一维表示 token_embeddings
    def forward(self, token_ids:Float[torch.LongTensor, "..."]) -> Float[Tensor, "... embedding_dim"]:
        return self.embeddings[token_ids]       # torch.Tensor 的高阶索引法则

# 均方根层归一化 (Root Mean Square Layer Normalization)
class RMSNorm(nn.Module):
    def __init__(self, 
            d_model:int, 
            eps:float = 1e-5, 
            device:torch.device | None = None, 
            dtype:torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # 作业要求采用 RMSNorm: 1，即初始化是 torch.ones()，元素 全为 1
        self.gains = nn.Parameter(torch.ones((d_model, ), device = device, dtype = dtype))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # 校验 d_model 防止传错维度时静默算错
        assert x.shape[-1] == self.d_model

        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_a = torch.sqrt(torch.sum(x ** 2, dim = -1, keepdim = True) / self.d_model + self.eps)
        gains = self.gains.to(torch.float32)
        result = x * gains / rms_a
        # 或用 torch.rsqrt() 直接计算出来倒数平方根

        return result.to(in_dtype)      # Transformer 里每一层的输入输出 dtype 应该保持一致


def silu(x:torch.Tensor):
    return x * torch.sigmoid(x)

# 逐位置前馈网络 (Position-Wise Feed-Forward Network)
class SwiGLU(nn.Module):
    def __init__(self, 
            d_model:int, 
            d_ff:int, 
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x:Float[Tensor, "... d_model"], ) -> Float[Tensor, "... d_model"]:    # 门控机制
        gate = silu(self.w1(x))     # (..., d_ff)，过 SiLU
        value = self.w3(x)          # (..., d_ff)，不过激活
        return self.w2(gate * value)  # (..., d_model)，此处做 hadamard 积（对应位置矩阵乘法）

        # 或者采用 einsum 一条公式得到结论
        # return einsum(silu(einsum(x, self.w1.weight, "... d_model, d_ff d_model -> ... d_ff")) * einsum(x, self.w3.weight, "... d_model, d_ff d_model -> ... d_ff"), self.w2.weight, "... d_ff, d_model d_ff -> ... d_model")


# 旋转位置编码(Rotary Position Embeddings, RoPE)
class RoPE(nn.Module):
    def __init__(self, 
            theta:float, 
            d_k:int, 
            max_seq_len:int, 
            device = None
    ):
        super().__init__()
        assert d_k % 2 == 0     # 在 RoPE 中 d_k 必须是偶数
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device = device) / d_k))
        pos = torch.arange(max_seq_len, device = device)
        angles = einsum(freqs, pos, "d_k_half, seq_len -> seq_len d_k_half")

        # 角度、三角函数矩阵的形状都为 (seq_len, d_k_half)
        # register_buffer 自动跟随参数转移设备（cpu，gpu等）
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        
    def forward(self, 
            x: Float[Tensor, "... seq_len d_k"], 
            token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        # 通过对 token_positions(..., seq_len) 高级索引，cos 和 sin 的形状都变为 (..., seq_len, d_k_half)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        # 两个 out 矩阵的形状都为 (..., seq_len, d_k_half)

        return rearrange(torch.stack((out1, out2), dim = -1), "... seq_len d_k_half pair -> ... seq_len (d_k_half pair)")



def softmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    exp = torch.exp(x - torch.max(x, dim = dim, keepdim=True).values)
    return exp / torch.sum(exp, dim = dim, keepdim = True)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self, 
            Q:Float[Tensor, "... queries d_k"], 
            K: Float[Tensor, "... keys d_k"], 
            V: Float[Tensor, "... keys d_v"], 
            mask:Bool[Tensor, " ... queries keys"] | None = None    # mask 是一个 bool 型矩阵
    ) -> Float[Tensor, " ... queries d_v"]:
        
        d_k = K.shape[-1]
        scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

        # 这里不能写 if mask 来判断（ mask != None 时会判断为 Tensor 而不是 bool:True，多元素张量转为 torch.bool 后不能被 if 判断）
        if mask is not None:
            scores = scores.masked_fill_(~mask, float('-inf'))      # 值为 True 的位置被覆盖（变为 -inf）

        attn_weight = softmax(scores, dim = -1)
        return einsum(attn_weight, V, "... queries keys, ... keys d_v -> ... queries d_v")

class MultiheadSelfAttention(nn.Module):
    def __init__(self, 
            d_model:int, 
            num_heads:int, 
            max_seq_len:int | None = None, 
            theta:float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_heads
        self.max_seq_len = max_seq_len

        self.d_head = d_model // num_heads

        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)

        # self.w_qkv = Linear(d_model, 3 * d_model)

        self.rope = RoPE(theta, self.d_head, self.max_seq_len) if max_seq_len is not None else None
        self.sdpa = ScaledDotProductAttention()


    def forward(self, 
            in_features: Float[Tensor, "... seq_len d_model"],
            token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
    

        # 这三个矩阵的形状都为 (..., seq_len, d_model) = (..., seq_len, d_model) @ (d_model, d_model)
        Q = self.w_q(in_features)
        K = self.w_k(in_features)
        V = self.w_v(in_features)

        # 拆分多头
        Q = rearrange(Q, "... seq_len (num_head d_head) -> ... num_head seq_len d_head", num_head = self.num_head)
        K = rearrange(K, "... seq_len (num_head d_head) -> ... num_head seq_len d_head", num_head = self.num_head)
        V = rearrange(V, "... seq_len (num_head d_head) -> ... num_head seq_len d_head", num_head = self.num_head)

        # RoPE 旋转处理：仅对 Q 和 K
        
        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # tril：保留下三角部分；triu：保留上三角部分
        seq_len = in_features.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype = torch.bool, device = in_features.device))

        # 此时 Q，K，V 形状为 (..., num_head, seq_len, d_head)，mask 形状为 (seq_len, seq_len)
        
        out = self.sdpa(Q, K, V, mask)
        # 输出形状为 (..., num_head, seq_len, d_head)
        out = rearrange(out, "... num_head seq_len d_head -> ... seq_len (num_head d_head)")
        return self.w_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, 
            d_model:int, 
            num_heads:int, 
            d_ff:int, 
            max_seq_len:int | None = None, 
            theta:float = 10000.0, 
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_head = d_model // num_heads
        
        self.max_seq_len = max_seq_len

        self.ffn = SwiGLU(d_model, d_ff)
        self.msa = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta = theta)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, 
            in_features: Float[Tensor, "batch sequence_length d_model"]
    ) -> Float[Tensor, "batch sequence_length d_model"]:

        seq_len = in_features.shape[-2]
        token_positions = torch.arange(seq_len, device=in_features.device)
        
        sublayer1 = in_features + self.msa(self.norm1(in_features), token_positions = token_positions)
        sublayer2 = sublayer1 + self.ffn(self.norm2(sublayer1))
        return sublayer2

class TransformerLM(nn.Module):
    def __init__(self, 
            vocab_size:int, 
            context_length:int, 
            num_layers:int, 

            d_model:int, 
            num_heads:int, 
            d_ff:int, 
            theta:float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

        # self.transformerblock:dict[int, TransformerBlock] = {}
        # for i in range(num_layers):
        #     self.transformerblock[i] = TransformerBlock(d_model, num_heads, d_ff, context_length, theta)

        self.transformerblock = nn.ModuleList(
                [TransformerBlock(d_model, num_heads, d_ff, context_length, theta) 
                 for _ in range(num_layers)])       # 采用 ModuleList 存子模块更好，具体类型类似普通 list
        
        self.embeddings = Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, 
            in_indices: Int[Tensor, "batch_size sequence_length"]
    ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:

        in_features = self.embeddings(in_indices)
        for num_layer in range(self.num_layers):
            in_features = self.transformerblock[num_layer](in_features)

        out = self.linear(self.norm(in_features))
        return out

    @torch.no_grad()
    def generate(
        self,
        prompt: Int[Tensor, "batch prompt_len"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> Int[Tensor, "batch total_len"]:
        """
        自回归生成。
        prompt:           (batch, prompt_len) 的 LongTensor（已 encode 的 token id）
        max_new_tokens:   最多生成多少个新 token
        temperature:      采样温度，越小越确定；= 0 时退化为贪心
        top_p:            nucleus sampling 阈值，1.0 表示不启用
        eos_token_id:     遇到它则标记该序列结束（整 batch 都结束就提前 break）
        """
        was_training = self.training
        self.eval()
        device = prompt.device
        out = prompt

        # 每个样本是否已经产生了 EOS
        finished = torch.zeros(prompt.shape[0], dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # 1. 只保留最后 context_length 个 token
            idx = out[:, -self.context_length:]

            # 2. 取最后一个位置的 logits
            logits = self(idx)[:, -1, :]            # (batch, vocab_size)

            # 3. temperature 缩放
            if temperature > 0:
                logits = logits / temperature

            # 4. 采样
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)   # (batch, 1)
            else:
                if top_p < 1.0:
                    logits = top_p_filter(logits, top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # 5. 已经结束的序列，之后一直输出 EOS
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
                finished = finished | (next_token.squeeze(-1) == eos_token_id)

            out = torch.cat([out, next_token], dim=1)

            # 6. 全部结束就提前退出
            if eos_token_id is not None and finished.all():
                break

        if was_training:
            self.train()
        return out

def top_p_filter(
    logits: Float[Tensor, "batch vocab"],
    top_p: float,
) -> Float[Tensor, "batch vocab"]:
    """Nucleus sampling：只保留累计概率达到 top_p 的最小 token 集合，其余置为 -inf。"""
    # 按 logits 降序排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 累计概率超过 top_p 的 token 都丢掉（右移一位，保留第一个越过阈值的 token）
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # 把顺序还原到原 vocab 索引
    return torch.empty_like(sorted_logits).scatter_(-1, sorted_indices, sorted_logits)
