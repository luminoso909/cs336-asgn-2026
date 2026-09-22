import numpy as np
from numpy.typing import NDArray
import torch

def get_batch(
        x:NDArray, 
        batch_size:int, 
        context_length:int, 
        device:str
) -> tuple[torch.LongTensor, torch.LongTensor]:
    starts = np.random.randint(low = 0, high = len(x)-context_length, size = batch_size)
    
    inputs_np = np.stack([x[i:i+context_length] for i in starts])
    targets_np = np.stack([x[i+1:i+context_length+1] for i in starts])

    # 注意返回类型是 torch.LongTensor，所以要求在从 numpy 转为 tensor 后使用 .long() 方法转为 Longtensor 类型
    inputs = torch.from_numpy(inputs_np).long().to(device = device)
    targets = torch.from_numpy(targets_np).long().to(device = device)

    return inputs, targets