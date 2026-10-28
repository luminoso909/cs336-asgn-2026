import regex as re
import pickle
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.merges_rank = {pair:rank for rank, pair in enumerate(merges)}

        self.special_tokens = special_tokens if special_tokens is not None else []
        self.reversed_vocab: dict[bytes, int] = {token:idx for idx, token in self.vocab.items()}

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.reversed_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.reversed_vocab[token_bytes] = new_id


    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        with open(vocab_path, mode='rb') as f:
            vocab = pickle.load(f)
        with open(merges_path, mode='rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pattern = '(' + '|'.join(
                re.escape(token) for token in sorted(
                    self.special_tokens, key=lambda x:len(x), reverse=True
                )
            ) + ')'
            chunks = re.split(pattern, text)
        else:
            chunks = [text]

        text_ids: list[int] = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                text_ids.append(self.reversed_vocab[chunk.encode("utf-8")])
                continue
            for match in re.finditer(PAT, chunk):
                words = match.group()
                
                word_bytes = [bytes([b]) for b in words.encode("utf-8")]

                # # old algorithm: O(len(merges) * len(word_bytes))
                # for merge in self.merges:
                #     word_bytes = self._merge_bytes(merge, word_bytes)

                # new algorithm: O(len(word_bytes)^2)
                word_bytes = self._merge_bytes_updated(word_bytes)
                
                for byte in word_bytes:
                    text_ids.append(self.reversed_vocab[byte])

        return text_ids

    def _merge_bytes(self, merge: tuple[bytes, bytes], word_bytes: list[bytes]):
        new_word_bytes = []
        pos = 0
        while pos < len(word_bytes):
            if pos < len(word_bytes)-1 and (word_bytes[pos], word_bytes[pos+1]) == merge:
                new_word_bytes.append(merge[0]+merge[1])
                pos += 2
            else:
                new_word_bytes.append(word_bytes[pos])
                pos += 1
        return new_word_bytes

    def _merge_bytes_updated(self, word_bytes: list[bytes]):
        new_word_bytes = word_bytes
        while True:
            selected_merge = None
            pos1, pos2 = 0, 0
            while pos1 < len(new_word_bytes)-1:
                if (new_word_bytes[pos1], new_word_bytes[pos1+1]) in self.merges_rank:
                    merge = (new_word_bytes[pos1], new_word_bytes[pos1+1])
                    if (
                        selected_merge is None
                        or self.merges_rank[merge] < self.merges_rank[selected_merge]
                    ):
                        selected_merge = merge

                pos1 += 1

            if selected_merge is None:
                break

            pos2 = 0
            tmp_word_bytes = []
            while pos2 < len(new_word_bytes):
                if pos2 < len(new_word_bytes)-1 and (new_word_bytes[pos2], new_word_bytes[pos2+1]) == selected_merge:
                    tmp_word_bytes.append(selected_merge[0]+selected_merge[1])
                    pos2 += 2
                else:
                    tmp_word_bytes.append(new_word_bytes[pos2])
                    pos2 += 1
            new_word_bytes = tmp_word_bytes

        return new_word_bytes

    # boundary problem
    def encode_iterable(self, iterable: Iterable[str], chunk_size: int = 1 << 22) -> Iterator[int]:
        # buf = ""
        # for piece in iterable:
        #     buf += piece
        #     if len(buf) >= chunk_size:
        #         yield from self.encode(buf)
        #         buf = ""
        # if buf:
        #     yield from self.encode(buf)
        
        for piece in iterable:
            yield from self.encode(piece)

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[i] for i in ids]
        text = b''.join(bytes_list).decode("utf-8", errors="replace")
        return text