"""Hash-based tokenizer for the 0.6M TinyTransformer core."""

from typing import List


class HashTokenizer:
    """Fixed-size hash vocabulary tokenizer for reproducibility.

    Maps tokens to hash-based IDs within a fixed vocabulary size.
    Reserves ID 0 for PAD.
    """

    def __init__(self, vocab_size: int = 2960, max_len: int = 128):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def encode(self, text: str) -> List[int]:
        toks = text.lower().split()
        ids = []
        for t in toks[: self.max_len]:
            h = (hash(t) % (self.vocab_size - 1)) + 1  # reserve 0 for PAD
            ids.append(h)
        ids += [0] * (self.max_len - len(ids))
        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]
