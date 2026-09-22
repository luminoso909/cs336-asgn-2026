# from typing import Iterable, Iterator
# import pickle
# import regex as re

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# class Tokenizer:
#     def __init__(
#             self, 
#             vocab:dict[int, bytes], 
#             merges:list[tuple[bytes, bytes]], 
#             special_tokens:list[str] | None
#     ):
#         self.vocab = vocab
#         self.merges = merges
#         self.special_tokens = special_tokens if special_tokens else []
#         self.reversed_vocab = {v:k for k, v in self.vocab.items()}
#         self.safe_token = [' ', '\n']

#         if special_tokens is not None:
#             for token in special_tokens:
#                 token_byte = token.encode('utf-8')
#                 if token_byte not in self.reversed_vocab:
#                     new_id = len(self.vocab)
#                     self.vocab[new_id] = token_byte
#                     self.reversed_vocab[token_byte] = new_id

 

#     @classmethod
#     def from_files(
#             cls, 
#             vocab_filepath:str, 
#             merges_filepath:str, 
#             special_tokens:list[str] | None = None
#     ) -> 'Tokenizer':
#         with open(vocab_filepath, mode = 'rb') as f:
#             vocab = pickle.load(f)
#         with open(merges_filepath, mode = 'rb') as f:
#             merges = pickle.load(f)

#         return cls(vocab, merges, special_tokens)

#     def encode(self, text:str) -> list[int]:
#         if self.special_tokens:
#             sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
#             pattern = "|".join(re.escape(tok) for tok in sorted_specials)
#             chunks = re.split(f"({pattern})", text)
#         else:
#             chunks = [text]

#         past_words_id = {}
#         word_ids = []

#         for chunk in chunks:
#             if chunk not in self.special_tokens:
#                 for match in re.finditer(PAT, chunk):
#                     word = match.group()

#                     if word in past_words_id.keys():
#                         for i in range(len(past_words_id[word])):
#                             word_ids.append(past_words_id[word][i])
#                         continue

#                     word_byte = tuple(bytes([byte]) for byte in word.encode('utf-8'))
#                     new_word_byte = self.merge_byte(word_byte, self.merges)

#                     word_id = []
#                     for token in new_word_byte:
#                         word_id.append(self.reversed_vocab[token])

#                     past_words_id[word] = word_id
#                     word_ids += word_id
                    
#             else:
#                 word_ids.append(self.reversed_vocab[chunk.encode('utf-8')])
 
#         return word_ids
        

#     def merge_byte(self, word_byte, merges):
#         for pair in merges:              # ← 外层是 merges，不是 word_byte
#             new_word_byte = []
#             i = 0
#             while i < len(word_byte):
#                 if i < len(word_byte)-1 and (word_byte[i], word_byte[i+1]) == pair:
#                     new_word_byte.append(pair[0] + pair[1])
#                     i += 2
#                 else:
#                     new_word_byte.append(word_byte[i])
#                     i += 1
#             word_byte = tuple(new_word_byte)
#         return word_byte

#     def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
#         buffer = ""

#         for line in iterable:
#             buffer += line

#             if buffer[-1] in self.safe_token:
#                 yield from self.encode(buffer)
#                 buffer = ""

#         if buffer:
#             yield from self.encode(buffer)

#     def decode(self, word_ids: list[int]) -> str:
#         byte_list = [self.vocab[i] for i in word_ids]

#         return b"".join(byte_list).decode('utf-8', errors="replace")




from typing import Iterable, Iterator
import pickle
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # bytes -> id
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

        # pair -> rank（用 dict 加速查表，替代 list 的 O(n) 查找）
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # 特殊 token 加入 vocab
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.reversed_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = tok_bytes
                self.reversed_vocab[tok_bytes] = new_id

        # 编译特殊 token 的分割正则（按长度从长到短，避免短 token 抢先匹配）
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pat = "|".join(re.escape(tok) for tok in sorted_specials)
            self.special_pattern = re.compile(f"({pat})")
            self.special_set = set(self.special_tokens)
        else:
            self.special_pattern = None
            self.special_set = set()

        # 编码缓存：word(str) -> list[int]
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    # ---------- 核心：BPE 合并 ----------
    def _merge_word_bytes(self, word_bytes: list[bytes]) -> list[bytes]:
        """
        对单个 pre-token 的字节序列做 BPE 合并。
        规则：反复找出当前序列里 rank 最小的相邻 pair 合并，直到没有可合并的 pair。
        """
        tokens = list(word_bytes)
        while len(tokens) >= 2:
            best_rank = None
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                r = self.merge_ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_idx = i
            if best_rank is None:
                break
            tokens[best_idx : best_idx + 2] = [tokens[best_idx] + tokens[best_idx + 1]]
        return tokens

    def _encode_word(self, word: str) -> list[int]:
        """对一个 pre-token（不含特殊 token）编码。带缓存。"""
        cached = self._cache.get(word)
        if cached is not None:
            return cached

        word_bytes = [bytes([b]) for b in word.encode("utf-8")]
        merged = self._merge_word_bytes(word_bytes)
        ids = [self.reversed_vocab[tok] for tok in merged]
        self._cache[word] = ids
        return ids

    # ---------- encode ----------
    def encode(self, text: str) -> list[int]:
        if self.special_pattern is not None:
            chunks = self.special_pattern.split(text)
        else:
            chunks = [text]

        out: list[int] = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_set:
                out.append(self.reversed_vocab[chunk.encode("utf-8")])
                continue
            for match in re.finditer(PAT, chunk):
                word = match.group()
                out.extend(self._encode_word(word))
        return out

    # ---------- encode_iterable ----------
    def encode_iterable(self, iterable: Iterable[str], chunk_size: int = 1 << 22) -> Iterator[int]:
        """
        流式编码。把输入流缓冲到约 1MB 再编码，避免逐行切断跨行 token。
        """
        buf = ""
        for piece in iterable:
            buf += piece
            if len(buf) >= chunk_size:
                yield from self.encode(buf)
                buf = ""
        if buf:
            yield from self.encode(buf)

    # ---------- decode ----------
    def decode(self, ids: list[int]) -> str:
        parts = [self.vocab[i] for i in ids]
        return b"".join(parts).decode("utf-8", errors="replace")