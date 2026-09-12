"""A perceptron and an ADALINE, from scratch, and the boundary they cannot
cross.

Both are single-layer linear classifiers, and the contrast between them is
instructive. The perceptron learns from the sign of its output with the
perceptron rule, and converges in finite time if and only if the classes are
linearly separable. ADALINE learns from the *continuous* output with the delta
(LMS) rule, minimising mean-squared error, so it converges smoothly and is less
jittery near the boundary - but it is still a single linear unit.

That shared limitation is the honest finding: neither can solve XOR, because no
straight line separates it. It is the 1969 result that stalled neural networks
for a decade, reproduced here rather than recounted.
"""

from __future__ import annotations

import numpy as np


class Perceptron:
    def __init__(self, n_features: int, lr: float = 0.1, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.lr = lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(X @ self.w + self.b >= 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> dict:
        """The perceptron rule: on a misclassified point, nudge the weights
        toward it. Track the epoch at which errors first hit zero - which
        happens for separable data and never for the rest."""
        converged_epoch = None
        history = []
        for epoch in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = 1 if xi @ self.w + self.b >= 0 else 0
                update = self.lr * (target - pred)
                self.w += update * xi
                self.b += update
                errors += int(update != 0.0)
            history.append(errors)
            if errors == 0 and converged_epoch is None:
                converged_epoch = epoch
                break
        acc = float(np.mean(self.predict(X) == y))
        return {"converged_epoch": converged_epoch, "final_accuracy": round(acc, 3),
                "error_history": history}


class Adaline:
    def __init__(self, n_features: int, lr: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.lr = lr

    def activation(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b        # linear, no threshold during training

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.activation(X) >= 0.5, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200) -> dict:
        """The delta rule: descend the mean-squared error of the *continuous*
        output. Because it minimises a smooth cost, it settles rather than
        oscillating - but it still draws only a straight boundary."""
        mse_history = []
        for _ in range(epochs):
            output = self.activation(X)
            error = y - output
            self.w += self.lr * (X.T @ error) / len(X)
            self.b += self.lr * error.mean()
            mse_history.append(float((error ** 2).mean()))
        acc = float(np.mean(self.predict(X) == y))
        return {"final_mse": round(mse_history[-1], 4),
                "final_accuracy": round(acc, 3),
                "mse_history": [round(m, 4) for m in mse_history[::max(1, epochs // 10)]]}


def linearly_separable(seed: int, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """A separable traffic-state dataset: two features (say normalised queue and
    occupancy), class 1 = congested when their sum is high, with a clear gap so
    the classes do not touch."""
    rng = np.random.default_rng(seed)
    free = rng.normal([0.3, 0.3], 0.1, (n // 2, 2))
    congested = rng.normal([0.7, 0.7], 0.1, (n // 2, 2))
    X = np.vstack([free, congested])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def xor_problem() -> tuple[np.ndarray, np.ndarray]:
    """The canonical non-separable set: XOR. No straight line splits it, so no
    single linear unit can classify it correctly."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    return X, y
