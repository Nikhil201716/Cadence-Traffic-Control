"""A Hopfield network that recovers corrupted detector readings.

The detectors across the network report a pattern - which approaches are
congested, which are clear - and sensors are noisy: bits flip. A Hopfield
network stores the handful of typical traffic-state patterns as attractors in
an associative memory, and relaxes a noisy reading back to the nearest stored
one. Weights are Hebbian (the outer-product rule), states are bipolar, and
recall is asynchronous until the network settles - all written out, no library.

Its famous limit is capacity: a network of N neurons reliably stores only about
0.138 N patterns before the attractors interfere and recall breaks down. The
experiments measure both the recovery curve and that capacity cliff rather than
quoting them.
"""

from __future__ import annotations

import numpy as np


class HopfieldNetwork:
    def __init__(self, size: int):
        self.size = size
        self.W = np.zeros((size, size))

    def train(self, patterns: np.ndarray) -> None:
        """Hebbian storage: W = (1/N) sum_p x_p x_p^T, with a zero diagonal so a
        neuron does not reinforce itself. Patterns are bipolar (-1/+1) rows."""
        self.W = np.zeros((self.size, self.size))
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= self.size
        np.fill_diagonal(self.W, 0.0)

    def recall(self, x: np.ndarray, max_steps: int = 20,
               rng: np.random.Generator | None = None) -> np.ndarray:
        """Relax a state to a fixed point by asynchronous updates: repeatedly
        set a random neuron to the sign of its weighted input, until a full
        sweep changes nothing. Asynchronous (one neuron at a time) is what
        guarantees convergence for a symmetric zero-diagonal weight matrix."""
        rng = rng or np.random.default_rng(0)
        s = x.copy().astype(float)
        for _ in range(max_steps):
            changed = False
            for i in rng.permutation(self.size):
                net = self.W[i] @ s
                new = 1.0 if net >= 0 else -1.0
                if new != s[i]:
                    s[i] = new
                    changed = True
            if not changed:
                break
        return s


def corrupt(pattern: np.ndarray, flips: int,
            rng: np.random.Generator) -> np.ndarray:
    """Flip `flips` random bits of a bipolar pattern - modelling sensor noise."""
    out = pattern.copy()
    idx = rng.choice(len(pattern), size=flips, replace=False)
    out[idx] *= -1
    return out


def recovery_curve(size: int, n_patterns: int, seed: int,
                   trials: int = 200) -> list[dict]:
    """Store `n_patterns` random patterns, then measure exact-recovery rate as
    the number of flipped bits grows - the associative-memory recovery curve."""
    rng = np.random.default_rng(seed)
    patterns = rng.choice([-1, 1], size=(n_patterns, size))
    net = HopfieldNetwork(size)
    net.train(patterns)

    out = []
    for flips in range(0, size // 2 + 1, max(1, size // 12)):
        recovered = 0
        for _ in range(trials):
            p = patterns[rng.integers(n_patterns)]
            noisy = corrupt(p, flips, rng)
            settled = net.recall(noisy, rng=rng)
            if np.array_equal(settled, p):
                recovered += 1
        out.append({"flipped_bits": int(flips),
                    "fraction_of_bits": round(flips / size, 3),
                    "exact_recovery_rate": round(recovered / trials, 3)})
    return out


def capacity_curve(size: int, seed: int, trials: int = 200) -> list[dict]:
    """Store an increasing number of patterns and measure how reliably a
    lightly-corrupted pattern is recovered - the capacity cliff around
    0.138 N."""
    rng = np.random.default_rng(seed)
    flips = max(1, size // 20)                # a light, fixed corruption
    out = []
    for m in range(1, int(0.30 * size) + 1, max(1, size // 20)):
        patterns = rng.choice([-1, 1], size=(m, size))
        net = HopfieldNetwork(size)
        net.train(patterns)
        recovered = 0
        for _ in range(trials):
            p = patterns[rng.integers(m)]
            settled = net.recall(corrupt(p, flips, rng), rng=rng)
            if np.array_equal(settled, p):
                recovered += 1
        out.append({"stored_patterns": m,
                    "load_ratio": round(m / size, 3),
                    "recovery_rate": round(recovered / trials, 3)})
    return out
