import torch
from torch import nn, Tensor
import math
from jaxtyping import Float, Bool, Int
from einops import einsum, rearrange
import pickle

from .utils import softmax, silu, temperature_scaling, top_p_selected
from .tokenizer import Tokenizer

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features), 
                device = device, 
                dtype = dtype
            )
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight, 
            mean = 0.0, 
            std = std, 
            a = -3*std, 
            b = 3*std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return out

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim), 
                device = device, 
                dtype = dtype
            )
        )
        torch.nn.init.trunc_normal_(
            self.embeddings, 
            mean = 0.0, 
            std = 1, 
            a = -3, 
            b = 3
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gains = nn.Parameter(
            torch.ones(
                d_model, 
                device = device, 
                dtype = dtype
            )
        )

    def forward(self, x:Float[Tensor, "... seq_len d_model"]) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(
            torch.mean(torch.pow(x, 2), dim = -1, keepdim = True) + self.eps
        )
        out = rms * x * self.gains      # (B, seq_len, d_model) * (B, seq_len, 1) * (1, 1, d_model)
        
        return out.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x:Float[Tensor, "... seq_len d_model"]):
        gate = silu(self.w1(x))
        return self.w2(gate * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super().__init__()
        assert d_k % 2 == 0 
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        position = torch.arange(0, max_seq_len, device=device).float()
        angles = einsum(position, freqs, "max_seq_len, d_k_half -> max_seq_len d_k_half")

        cos_cache = torch.cos(angles).to(device=device)
        sin_cache = torch.sin(angles).to(device=device)
        # register_buffer: 把一个不需要学习、但属于模型状态的 Tensor 注册到 nn.Module 中
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(
        self, 
        x:Float[Tensor, "... seq_len d_model"], 
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos

        out = rearrange(
            torch.stack((y_even, y_odd), dim = -1), 
            "... d_k_half pair -> ... (d_k_half pair)"
        )
        return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        queries:Float[Tensor, " ... queries d_k"], 
        keys:Float[Tensor, " ... keys d_k"], 
        values:Float[Tensor, " ... keys d_v"],
        mask:Bool[Tensor, " ... queries keys"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        self.d_k = queries.shape[-1]
        scores = einsum(queries, keys, "... q d_k, ... k d_k -> ... q k") / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill_(~mask, float('-inf'))
        attn_scores = softmax(scores, dim = -1)

        out = einsum(attn_scores, values, "... q k, ... k d_v -> ... q d_v")
        return out

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model:int, 
        num_heads:int, 
        max_seq_len:int, 
        theta: float = 10000.0, 
        device=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)

        self.sdp_attn = ScaledDotProductAttention()
        self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device)
        causal_mask = torch.tril(
            torch.ones((max_seq_len, max_seq_len))
        ).to(dtype=bool, device=device)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(
        self, 
        x:Float[Tensor, " ... seq_len d_model"], 
        token_positions:Int[Tensor, " ... sequence_length"] | None = None, 
        is_casual:bool = True, 
    ) -> Float[Tensor, " ... seq_len d_model"]:
        queries = self.w_q(x)
        keys = self.w_k(x)
        values = self.w_v(x)

        mh_q = rearrange(queries, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", d_head = self.d_head)
        mh_k = rearrange(keys, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", d_head = self.d_head)
        mh_v = rearrange(values, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", d_head = self.d_head)

        if token_positions is not None:
            mh_q = self.rope(mh_q, token_positions)
            mh_k = self.rope(mh_k, token_positions)

        if is_casual:
            seq_len = x.shape[-2]
            mask = self.causal_mask[:seq_len, :seq_len]
            mh_out = self.sdp_attn(mh_q, mh_k, mh_v, mask)
        else:
            mh_out = self.sdp_attn(mh_q, mh_k, mh_v)

        out = rearrange(mh_out, "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)")
        out = self.w_o(out)
        return out
        
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model:int, 
        num_heads: int, 
        d_ff:int, 
        max_seq_len: int, 
        theta: float
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mha = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x:Float[Tensor, " ... seq_len d_model"]):
        token_positions = torch.arange(x.shape[-2], device=x.device)
        y = x + self.mha(self.norm1(x), token_positions)
        out = y + self.swiglu(self.norm2(y))
        return out
        
class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.embedding = Embedding(vocab_size, d_model)

        self.transformerblock = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(
        self, 
        token_ids:Int[Tensor, "... context_length"]
    ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        
        embeddings = self.embedding(token_ids)
        out = embeddings
        for layer in self.transformerblock:
            out = layer(out)
        out = self.linear(self.norm(out))
        return out

    def _load_state(self, path:dict) -> Tokenizer:
        vocab_path = path["vocab"]
        merges_path = path["merges"]
        special_tokens_path = path["special_tokens"]
        checkpoint_path = path["checkpoint"]
        
        with open(special_tokens_path, mode='rb') as f:
            special_tokens = pickle.load(f)

        tokenizer = Tokenizer.from_files(
            vocab_path,
            merges_path,
            special_tokens
        )

        checkpoint = torch.load(
            checkpoint_path,
            weights_only=True
        )
        self.load_state_dict(checkpoint["model"])
        return tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompt: str, 
        tokenizer: Tokenizer, 
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> Int[Tensor, "batch total_len"]:
        self.eval()

        input_ids: Int[Tensor, "batch prompt_len"] = tokenizer.encode(prompt)
        token_ids = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=next(self.parameters()).device
        )
        eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.reversed_vocab[b"<|endoftext|>"]

        for _ in range(max_new_tokens):
            idx_cond = token_ids[:, -self.context_length:]
            logits = self.forward(idx_cond)[..., -1, :]
            probs: Float[Tensor, "batch vocab_size"] = temperature_scaling(
                logits, 
                temperature, 
                dim=-1
            )
            sorted_probs, sorted_indices = top_p_selected(probs, top_p)
            # multinomial 按照你给出的概率权重，从多个候选项中随机抽取索引
            sampled_index = torch.multinomial(
                sorted_probs,
                num_samples=1
            )
            predicted = sorted_indices.gather(
                dim=-1,
                index=sampled_index
            )
            token_ids = torch.concat((token_ids, predicted), dim=-1)
            if eos_token_id is not None:
                if (predicted == eos_token_id).all():
                    break

        text = tokenizer.decode(token_ids[0].tolist())

        return text