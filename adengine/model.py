"""A from-scratch logistic-regression CTR model.

predict():  feature vector -> click probability in (0, 1)
train():    learn the weights from logged (features, clicked) examples

Kept dependency-free on purpose — the math is the lesson.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from .features import NUM_FEATURES

# One training example: a feature vector and whether it was clicked (1/0).
Example = Tuple[List[float], int]


def sigmoid(x: float) -> float:
    """Map any real number to (0, 1). Clamped to avoid math.exp overflow."""
    if x < -35.0:
        return 0.0
    if x > 35.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class CTRModel:
    """Logistic regression over hashed features, trained with SGD."""

    def __init__(self, num_features: int = NUM_FEATURES, lr: float = 0.1) -> None:
        self.weights: List[float] = [0.0] * num_features
        self.lr = lr

    def score(self, features: List[float]) -> float:
        """Raw linear score: the dot product of weights and features."""
        return sum(w * f for w, f in zip(self.weights, features))

    def predict(self, features: List[float]) -> float:
        """Predicted click-through probability for one opportunity."""
        return sigmoid(self.score(features))

    def update(self, features: List[float], clicked: int) -> float:
        """One SGD step on a single example. Returns the example's log loss."""
        p = self.predict(features)
        error = p - clicked                      # gradient of log loss wrt score
        for i, f in enumerate(features):
            if f != 0.0:                         # skip the zeros — they don't move
                self.weights[i] -= self.lr * error * f
        eps = 1e-12
        return -(clicked * math.log(p + eps) + (1 - clicked) * math.log(1 - p + eps))

    def train(self, data: Iterable[Example], epochs: int = 1) -> float:
        """Train over the dataset for `epochs` passes. Returns final-epoch loss."""
        last_loss = 0.0
        for _ in range(epochs):
            total, n = 0.0, 0
            for features, clicked in data:
                total += self.update(features, clicked)
                n += 1
            last_loss = total / max(n, 1)
        return last_loss
