import torch
import typing
import os

def save_checkpoint(
        model:torch.nn.Module, 
        optimizer:torch.optim.Optimizer, 
        iteration:int,      # 训练迭代次数（训练步数）：已经执行了多少次优化器更新
        out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes],      # 保存路径
        loss:float | None = None, 
):
    checkpoint = {
        "model":model.state_dict(), 
        "optimizer":optimizer.state_dict(), 
        "iteration":iteration, 
        'loss': loss}
    torch.save(checkpoint, out)


def load_checkpoint(
        src:str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
        model:torch.nn.Module, 
        optimizer:torch.optim.Optimizer
) -> tuple[int, float]:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    iteration = checkpoint.get("iteration", 0)
    loss = checkpoint.get("loss", None)

    return iteration, loss