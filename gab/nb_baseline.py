"""Multinomial Naive Bayes baseline for action classification (E1)."""

import math
from collections import Counter, defaultdict
from typing import List, Optional, Tuple


ACTIONS = ["SEARCH", "MEMORY_READ", "EXEC", "TEACHER", "ANSWER"]


def tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.strip()]


class MultinomialNB:
    """Naive Bayes classifier with Laplace smoothing for action prediction."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_counts: Counter = Counter()
        self.word_counts: defaultdict = defaultdict(Counter)
        self.total_words: Counter = Counter()
        self.vocab: set = set()

    def fit(self, data: List[Tuple[str, str]]):
        for text, y in data:
            self.class_counts[y] += 1
            for w in tokenize(text):
                self.vocab.add(w)
                self.word_counts[y][w] += 1
                self.total_words[y] += 1

    def predict(self, text: str) -> str:
        V = len(self.vocab)
        N = sum(self.class_counts.values())
        best_y, best_score = None, -1e18
        words = tokenize(text)
        for y in ACTIONS:
            logp = math.log(
                (self.class_counts[y] + self.alpha) / (N + self.alpha * len(ACTIONS))
            )
            denom = self.total_words[y] + self.alpha * V
            for w in words:
                logp += math.log((self.word_counts[y][w] + self.alpha) / denom)
            if logp > best_score:
                best_score, best_y = logp, y
        return best_y

    def predict_proba(self, text: str) -> dict:
        """Return log-probabilities for each action."""
        V = len(self.vocab)
        N = sum(self.class_counts.values())
        words = tokenize(text)
        scores = {}
        for y in ACTIONS:
            logp = math.log(
                (self.class_counts[y] + self.alpha) / (N + self.alpha * len(ACTIONS))
            )
            denom = self.total_words[y] + self.alpha * V
            for w in words:
                logp += math.log((self.word_counts[y][w] + self.alpha) / denom)
            scores[y] = logp
        return scores
