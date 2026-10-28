import pickle
import os

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.train import parse_args
from cs336_basics.data import load_checkpoint


def generate(prompt: str):
    args = parse_args()

    with open("data/bpe_checkpoints/special_tokens.pkl", mode='rb') as f:
        special_tokens = pickle.load(f)

    tkzr = Tokenizer.from_files(
        "data/bpe_checkpoints/bpe_vocab_50000.pkl", 
        "data/bpe_checkpoints/bpe_merges_50000.pkl", 
        special_tokens
        )
    model = TransformerLM(
        vocab_size = args.vocab_size, 
        context_length = args.context_length, 
        d_model = args.d_model, 
        num_layers = args.num_layers, 
        num_heads = args.num_heads, 
        d_ff = args.d_ff, 
        rope_theta = args.theta
    ).to(args.device)
    eos_token_id = tkzr.reversed_vocab[]
    state_dict = load_checkpoint(os.path.join(args.checkpoint_path, "latest.pt"))
    model.load_state_dict(state_dict)

    inputs: list[int] = tkzr.encode(prompt)
    out = model.generate(
        inputs, 
        max_new_tokens = 512, 
        temperature = 0.80, 
        top_p = 0.9, 
        eos_token_id = eos_token_id
    )
    return tkzr.decode(out)


if __name__ == "__main__":
    prompt = str(input())
    text = generate(prompt)
    print(text)