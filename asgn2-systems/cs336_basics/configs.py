import argparse
import torch

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description = "Training a CS336 A1 Transformer LM")
    # 使用 .add_argument(“--参数名”, type, required:bool（required 表示必须传，否则报错）, default:Any)

    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # 数据与分词器
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--valid_data', type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)
    parser.add_argument("--special_tokens_path", type=str, required=True)

    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--theta', type=float, default=10000.0)

    # 优化器参数
    parser.add_argument('--lr_max', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--lr_min', type=float, default=1e-5, help="Min learning rate for cosine decay")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip_max_norm', type=float, default=1.0)

    # 训练调度
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000)

    # 日志参数
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')

    return parser.parse_args(argv)      # 直接变成一个 namespace 对象，类似字典，只不过参数自动变成 arg.属性
