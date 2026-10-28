import regex as re
import pickle
import time
from pathlib import Path
import hashlib
import json


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE:
    def __init__(
        self, 
        vocab_size: int, 
        special_tokens: list[str] | None, 
    ):
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.vocab_size = vocab_size
        self.words_freq = {}
        
        self.vocab: dict[int, bytes] = self._init_vocab(self.special_tokens)
        self.merges: list[tuple[bytes, bytes]] = []

    @staticmethod
    def _init_vocab(
        special_tokens: list[str] | None
    ) -> dict[int, bytes]:
        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])

        if special_tokens is not None:
            for i in range(len(special_tokens)):
                vocab[256+i] = special_tokens[i].encode("utf-8")
        return vocab
    
    def _pre_tokenization(
        self, 
        content: str
    ):
        if self.special_tokens:
            pattern = '|'.join(
                re.escape(token) for token in sorted(
                    self.special_tokens, key=lambda x:len(x), reverse=True
                )
            )
            chunks = re.split(pattern, content)
        else:
            chunks = [content]
    
        count = 0 
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                word = match.group()
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                self.words_freq[word_bytes] = self.words_freq.get(word_bytes, 0) + 1

                count += 1
                if count % 1_000_000 == 0:
                    print(f"已处理 {count:,} 个 pre-tokens", flush=True)

    @staticmethod
    def _count_pair(
        words_freq: dict[tuple[bytes, ...], int]
    ) -> dict[tuple[bytes, bytes], int]:
        pairs_freq = {}
        for word_bytes, freq in words_freq.items():
            for i in range(len(word_bytes)-1):
                pair = tuple([word_bytes[i], word_bytes[i+1]])
                pairs_freq[pair] = pairs_freq.get(pair, 0) + freq
        return pairs_freq

    @staticmethod
    def _find_best_pair(
        pairs_freq: dict[tuple[bytes, bytes], int]
    ) -> tuple[bytes, bytes]:
        best_pair = max(pairs_freq, key = lambda pair: (pairs_freq[pair], pair))
        return best_pair

    @staticmethod
    def _merge_pair(
        word_bytes: tuple[bytes, ...], 
        best_pair: tuple[bytes, bytes]
    ) -> tuple[bytes, ...]:
        new_word_bytes = []
        pos = 0
        while pos < len(word_bytes):
            if pos < len(word_bytes)-1 and tuple([word_bytes[pos], word_bytes[pos+1]]) == best_pair:
                new_word_bytes.append(word_bytes[pos] + word_bytes[pos+1])
                pos += 2
            else:
                new_word_bytes.append(word_bytes[pos])
                pos += 1
        return tuple(new_word_bytes)

    def _update_words_freq(
        self, words_freq: dict[tuple[bytes, ...], int], 
        best_pair: tuple[bytes, bytes]
    ) -> dict[tuple[bytes, ...], int]:
        new_words_freq = {}
        for word_bytes, freq in words_freq.items():
            new_word_bytes = self._merge_pair(word_bytes, best_pair)
            new_words_freq[new_word_bytes] = new_words_freq.get(new_word_bytes, 0) + freq
        return new_words_freq

    def train_bpe(
        self, 
        input_path: str
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        start = time.perf_counter()
        # 三份状态放在同一目录；按输入文件及特殊 token 隔离，避免错用其他语料的存档
        cache_path = self._get_cache_path(input_path, self.special_tokens)
        checkpoint_dir = Path("./data/bpe_checkpoints") / cache_path.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 从已有的最大有效词表快照恢复；目标大小可以高于已保存词表
        resumed = False
        saved_sizes = sorted(
            (int(p.stem.split("_")[-1]) for p in checkpoint_dir.glob("bpe_vocab_*.pkl")
             if p.stem.split("_")[-1].isdigit()),
            reverse=True,
        )
        for size in saved_sizes:
            if size > self.vocab_size:
                continue
            vocab_path = checkpoint_dir / f"bpe_vocab_{size}.pkl"
            merges_path = checkpoint_dir / f"bpe_merges_{size}.pkl"
            freq_path = checkpoint_dir / f"bpe_words_freq_{size}.pkl"
            if not (merges_path.exists() and freq_path.exists()):
                continue
            self.vocab, self.merges, self.words_freq = self.load_data(
                vocab_path, merges_path, freq_path
            )
            if len(self.vocab) != size:
                raise ValueError(f"Invalid BPE checkpoint: {vocab_path}")
            resumed = True
            print(f"已从词表大小 {size} 恢复训练", flush=True)
            break

        # ———————————————————————————————————————————————————————————————————————————————————— #

        if not resumed:
            print("开始读取语料...", flush=True)
            with open(input_path, mode='r', encoding="utf-8") as f:
                content = f.read()
            print(f"读取完成：{time.perf_counter() - start:.2f}s")

        # ———————————————————————————————————————————————————————————————————————————————————— #

        if not resumed:
            print("开始预分词...", flush=True)
            if cache_path.exists():
                print("加载预分词缓存...", flush=True)
                with cache_path.open("rb") as f:
                    self.words_freq = pickle.load(f)
            else:
                print("开始读取和预分词...", flush=True)
                self.words_freq = {}
                self._pre_tokenization(content)
                with cache_path.open("wb") as f:
                    pickle.dump(self.words_freq, f, protocol=pickle.HIGHEST_PROTOCOL)
            del content
            print(
                f"预分词完成：{time.perf_counter() - start:.2f}s, 不同 pre-token 数量：{len(self.words_freq)}",
                flush=True,
            )

        # ———————————————————————————————————————————————————————————————————————————————————— #

        print("开始 BPE 合并训练...", flush=True)
        merge_start = time.perf_counter()

        while len(self.vocab) < self.vocab_size:
            if len(self.vocab) % 50 == 0:
                print(
                    f"""当前词表大小: {len(self.vocab)}, 当前进度：{100 * len(self.vocab)/self.vocab_size} %, 合并阶段耗时 {time.perf_counter() - merge_start:.1f}s""", 
                    flush=True
                )
                
            if len(self.vocab) in {10000, 20000, 30000}:
                size = len(self.vocab)
                self.save_data(
                    self.vocab, self.merges, self.words_freq,
                    checkpoint_dir / f"bpe_vocab_{size}.pkl",
                    checkpoint_dir / f"bpe_merges_{size}.pkl",
                    checkpoint_dir / f"bpe_words_freq_{size}.pkl",
                )

            pairs_freq = self._count_pair(self.words_freq)
            if not pairs_freq:
                break

            best_pair = self._find_best_pair(pairs_freq)
            new_words_freq = self._update_words_freq(self.words_freq, best_pair)
            self.words_freq = new_words_freq

            self.vocab[len(self.vocab)] = best_pair[0]+best_pair[1]
            self.merges.append(best_pair)

        print(
            f"""BPE 训练完成!! \n 最终词表大小: {len(self.vocab)}, 进度：{100 * len(self.vocab)/self.vocab_size} %, 总耗时：{time.perf_counter() - start:.2f}s""", 
            flush=True
        )
        
        size = len(self.vocab)
        self.save_data(
            self.vocab, self.merges, self.words_freq,
            checkpoint_dir / f"bpe_vocab_{size}.pkl",
            checkpoint_dir / f"bpe_merges_{size}.pkl",
            checkpoint_dir / f"bpe_words_freq_{size}.pkl",
        )
        return self.vocab, self.merges

    @staticmethod
    def save_data(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        words_freq: dict[tuple[bytes, ...], int],
        vocab_path: str | Path,
        merges_path: str | Path,
        words_freq_path: str | Path,
    ):
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(merges_path, 'wb') as f:
            pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(words_freq_path, 'wb') as f:
            pickle.dump(words_freq, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_data(vocab_path, merges_path, words_freq_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_path, 'rb') as f:
            merges = pickle.load(f)
        with open(words_freq_path, 'rb') as f:
            words_freq = pickle.load(f)
        return vocab, merges, words_freq

    def _get_cache_path(self, input_path: str | Path, special_tokens: list[str] | None):
        path = Path(input_path).resolve()
        stat = path.stat()

        config = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "special_tokens": special_tokens,
            "pattern": PAT,
        }

        key = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:16]

        cache_dir = Path("./data/bpe_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        return cache_dir / f"words_freq_{key}.pkl"

def train(input_path: str, vocab_size: int, special_tokens: list[str]):
    bpe = BPE(vocab_size, special_tokens)
    vocab, merges = bpe.train_bpe(input_path)

if __name__ == "__main__":
        train(
            input_path="./data/TinyStoriesV2-GPT4-train.txt",
            vocab_size=50000,
            special_tokens=["<|endoftext|>"],
        )