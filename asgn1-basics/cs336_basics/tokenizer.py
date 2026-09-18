from typing import Iterable, Iterator
import pickle
import regex as re
from cs336_basics.bpe import BPE

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
            self, 
            vocab:dict[int, bytes], 
            merges:list[tuple[bytes, bytes]], 
            special_tokens:list[str] | None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.reversed_vocab = {v:k for k, v in self.vocab.items()}
        self.safe_token = ['', ' ', '\n']

        if special_tokens:
            for token in special_tokens:
                token_byte = token.encode('utf-8')
                if token_byte not in self.reversed_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token_byte
                    self.reversed_vocab[token_byte] = new_id

 

    @classmethod
    def from_files(
            cls, 
            vocab_filepath:str, 
            merges_filepath:str, 
            special_tokens:list[str] | None = None
    ) -> 'Tokenizer':
        with open(vocab_filepath, mode = 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, mode = 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text:str) -> list[int]:
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(tok) for tok in sorted_specials)
            chunks = re.split(f"({pattern})", text)
        else:
            chunks = [text]

        past_words_id = {}
        word_ids = []

        for chunk in chunks:
            if chunk not in self.special_tokens:
                for match in re.finditer(PAT, chunk):
                    word = match.group()

                    if word in past_words_id.keys():
                        for i in range(len(past_words_id[word])):
                            word_ids.append(past_words_id[word][i])
                        continue

                    word_byte = tuple(bytes([byte]) for byte in word.encode('utf-8'))
                    new_word_byte = self.merge_byte(word_byte, self.merges)

                    word_id = []
                    for token in new_word_byte:
                        word_id.append(self.reversed_vocab[token])

                    past_words_id[word] = word_id
                    word_ids += word_id
                    
            else:
                word_ids.append(self.reversed_vocab[chunk.encode('utf-8')])
 
        return word_ids
        

    # def merge_byte(self, word_byte:tuple[bytes, ...], merges:list[tuple[bytes, bytes]]) -> tuple[bytes, ...]:
    #     new_word_byte = [word_byte[0]]

    #     i = 0
    #     j = 1
    #     while j < len(word_byte):

    #         if (new_word_byte[i], word_byte[j]) in merges:
    #             new_word_byte.append(new_word_byte[i]+word_byte[j])
    #             new_word_byte.pop(i)
    #         else:
    #             new_word_byte.append(word_byte[j])
    #             i += 1
    #         j += 1

    #     return tuple(new_word_byte)

    def merge_byte(self, word_byte, merges):
        for pair in merges:              # ← 外层是 merges，不是 word_byte
            new_word_byte = []
            i = 0
            while i < len(word_byte):
                if i < len(word_byte)-1 and (word_byte[i], word_byte[i+1]) == pair:
                    new_word_byte.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word_byte.append(word_byte[i])
                    i += 1
            word_byte = tuple(new_word_byte)
        return word_byte

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        
        for line in iterable:
            buffer += line

            if buffer[-1] in self.safe_token:
                for token_id in self.encode(buffer):
                    yield token_id
                buffer = ""

        if buffer:
            for token_id in self.encode(buffer):
                yield token_id


    def decode(self, word_ids: list[int]) -> str:
        byte_list = [self.vocab[i] for i in word_ids]

        return b"".join(byte_list).decode('utf-8', errors="replace")




