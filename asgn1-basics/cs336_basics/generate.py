import torch
from tokenizer import Tokenizer
from model import *

def generate(
        model:torch.nn.Module, 
        prompt:str, 
        max_tokens:int, 
        device:str, 
        temperature:float = 1.0, 
        top_p:float = 1.0, 
        eos_token_id:int = None
):
    """
    prompt_tokens: 初始提示词的 token ID 列表 (List[int])
    max_tokens: 最大生成 token 数
    temperature: 温度参数 tau
    top_p: 核采样阈值 p
    eos_token_id: 结束符的 ID (如 <|endoftext|> 的 ID)
    """

    model.eval()
    tokenizer = Tokenizer.from_files("../vocab.pkl", "../merges.pkl", ["<|endoftext|>"])
    ids:list[int] = tokenizer.encode(prompt)
    tokens = torch.tensor(ids, dtype = torch.long, device = device).unsqueeze_(0)

    with torch.no_grad():
        for _ in range(max_tokens):
            # 1. 准备输入，转换为 Tensor 并添加 batch 维度
            input_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, seq_len)
            
            # 2. 截断过长的序列（防止超过 context_length）
            # 假设模型有 context_length 限制，如果超出，只取最后 context_length 个 token
            # input_tensor = input_tensor[:, -model.context_length:] 
            
            # 3. 前向传播
            # 注意：需要传入 token_positions 用于 RoPE
            seq_len = input_tensor.shape[1]
            token_positions = torch.arange(seq_len, device=device).unsqueeze(0)
            logits = model(input_tensor, token_positions=token_positions) # (1, seq_len, vocab_size)
            
            # 4. 取最后一个位置的 logits
            next_token_logits = logits[0, -1, :] # (vocab_size,)
            
            # 5. 应用温度缩放
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # 6. 计算概率分布
            probs = softmax(next_token_logits, dim=-1) # 使用你之前写的 softmax 函数
            
            # 7. 应用 Top-p 采样
            if top_p < 1.0:
                # 降序排序
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                # 计算累积概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 标记需要移除的 token (累积概率 > top_p)
                # 注意：我们要保留第一个超过 top_p 的 token，所以将 mask 向右移一位
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                # 将需要移除的 token 概率置为 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0.0
                
                # 重新归一化
                probs = probs / probs.sum()
            
            # 8. 采样
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)
            
            # 9. 检查是否遇到结束符
            if eos_token_id is not None and next_token == eos_token_id:
                break
                
    return tokens

