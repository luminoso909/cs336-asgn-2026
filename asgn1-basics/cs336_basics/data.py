import torch
import numpy as np

from numpy.typing import NDArray
import typing, os

# provide (inputs, targets) batch
def get_batch(
    dataset: NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(dataset) > context_length
    selected = np.random.choice(
        len(dataset)-context_length, 
        batch_size, 
    )

    X, Y = [], []
    for i in selected:
        X.append(dataset[i:i+context_length])
        Y.append(dataset[i+1:i+context_length+1])

    return (
        torch.as_tensor(np.stack(X), dtype=torch.long, device=device), 
        torch.as_tensor(np.stack(Y), dtype=torch.long, device=device)
    )

# checkpoint：建议使用 dict 类型进行保存
def save_checkpoint(
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    loss: float | None = None,   
):

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration, 
        "loss": loss
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
) -> tuple[int, float | None]:
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    iteration = checkpoint.get("iteration", 0)
    loss = checkpoint.get("loss", None)

    return iteration, loss
