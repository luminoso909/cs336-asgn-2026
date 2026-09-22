import regex as re
import pickle

# TODO: 当跑完整数据集时，用 find_chunk_boundaries 和 multiprocessing 加速 pre_tokenization

# GPT-2 采用的正则语言
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE:
    def __init__(self, input_path:str, vocab_size:int, special_tokens:list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    # 构建字节词表(dict)：数字 -> utf-8字节表述，输出词表
    def init_vocab(self) -> dict[int, bytes]:
        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])
            # 函数 bytes([value1, value2]) 的使用 其中 value 必须是 0～255 的数字；用字节类型有效解决“未登陆”词的问题


        if self.special_tokens:
            for i in range(len(self.special_tokens)):
                vocab[256+i] = self.special_tokens[i].encode('utf-8')

        return vocab

    # 预分词：把读取后的文本先根据 special_tokens 分开成 chunks，然后把每个 chunk 的每个 word 转为字节化 tuple 表述并统计 words 的总出现次，输出为：字节化 tuple word -> 统计次数
    def pre_tokenization(self, content:str) -> dict[tuple[bytes, ...], int]:
        if self.special_tokens:
            pattern = '|'.join(re.escape(tok) for tok in self.special_tokens)
            chunks = re.split(pattern, content)
            # chunks = re.split(f"({pattern})", content) 此时返回的列表会把分隔符本身也包含进去，类似 ['Hello world', '[MASK]', 'This is a test', '<SEP>', 'End of text']
        
        else:
            chunks = [content]

        word_freqs = {}
        for chunk in chunks:
            for match in re.finditer(PAT, chunk):
                word = match.group()
                word_bytes = tuple(bytes([byte]) for byte in word.encode('utf-8'))
                # dict 的 key 必须是不可变类型，所以要用 tuple 作为 dict 的 key 来统计单词的次数，tuple 中存了这个单词每个字符的 utf-8 的字节化表述
                word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1

        return word_freqs

    # 统计相邻对：把两个相邻字母作为一组，统计这些两元素小组的出现频率
    def count_pairs(self, word_freqs:dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
        pair_freqs = {}

        for word_byte, freq in word_freqs.items():
            for i in range(len(word_byte)-1):
                pair = (word_byte[i], word_byte[i+1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

        return pair_freqs
    
    # 这里有问题：合并相邻对：通过 best_pair 比对 word_freqs 的 key（是一个 tuple）中频率最高的两个字符，然后合并他们，形成一个新的 word_freqs（由于一个二元 tuple 中两个字符成对出现，所以两者的 freq 相同），word tuple 的长度 - 1
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

    # 训练 bpe：通过输入文本、词表长度、特殊 token，得到一个包含 merged pair 的新词表（序号 -> byte 化字符）vocab 以及一系列 best_pair 在 merge 前的二元 tuple
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
            word_freqs = self.merge_pair(word_freqs, best_pair)     # 更新词频字典

            vocab[len(vocab)] = best_pair[0] + best_pair[1]
            merges.append(best_pair)
               
        return vocab, merges

    def save_file(self, vocab:dict[int, bytes], merges:list[tuple[bytes, bytes]], vocab_path:str, merges_path:str):
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        with open(merges_path, 'wb') as f:
            pickle.dump(merges, f)
