import regex as re
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE:
    def __init__(self, input_path:str, vocab_size:int, special_tokens:list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def init_vocab(self) -> dict[int, bytes]:
        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])

        if self.special_tokens:
            for i in range(len(self.special_tokens)):
                vocab[256+i] = self.special_tokens[i].encode('utf-8')

        return vocab

    def pre_tokenization(self, content:str) -> dict[tuple[bytes, ...], int]:
        if self.special_tokens:
            pattern = '|'.join(re.escape(tok) for tok in self.special_tokens)
            chunks = re.split(pattern, content)
        else:
            chunks = [content]

        word_freqs = {}
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                word = match.group()
                word_bytes = tuple(bytes([byte]) for byte in word.encode('utf-8'))
                word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1

        return word_freqs
    
    def count_pairs(self, word_freqs:dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
        pair_freqs = {}

        for word_byte, freq in word_freqs.items():
            for i in range(len(word_byte)-1):
                pair = (word_byte[i], word_byte[i+1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

        return pair_freqs

    def merge_pair(self, word_freqs:dict[tuple[bytes, ...], int], best_pair:tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
        new_word_freqs = {}

        for word_byte, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word_byte):
                if i < len(word_byte)-1 and (word_byte[i], word_byte[i+1]) == best_pair:
                    new_word.append(word_byte[i] + word_byte[i+1])
                    i += 2
                else:
                    new_word.append(word_byte[i])
                    i += 1

            new_word_freqs[tuple(new_word)] = freq

        return new_word_freqs
        
    # 增量更新：修正方案：用增量更新（ where_to_update + pair_counts 维护）
    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(self.input_path, mode='r', encoding='utf-8') as f:
            content = f.read()

        vocab = self.init_vocab()
        word_freqs = self.pre_tokenization(content)

        merges = []

        while len(vocab) < self.vocab_size:
            
            if len(vocab) % 300 == 0:
                print(f"当前词表大小: {len(vocab)}, 当前进度：{len(vocab)/self.vocab_size}")

            pair_counts = self.count_pairs(word_freqs)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key = lambda p:(pair_counts[p], p))
            word_freqs = self.merge_pair(word_freqs, best_pair)

            vocab[len(vocab)] = best_pair[0] + best_pair[1]
            merges.append(best_pair)
               
        return vocab, merges

    def save_file(self, vocab:dict[int, bytes], merges:list[tuple[bytes, bytes]], vocab_path:str, merges_path:str):
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        with open(merges_path, 'wb') as f:
            pickle.dump(merges, f)
