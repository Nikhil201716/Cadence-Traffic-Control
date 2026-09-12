"""Hopfield and the single-layer neurons: associative recall and its capacity
limit, perceptron convergence, and the XOR wall neither can cross.
"""

import numpy as np

from _harness import check, summary

from cadence.hopfield import HopfieldNetwork, corrupt, recovery_curve, capacity_curve
from cadence.neurons import (
    Perceptron, Adaline, linearly_separable, xor_problem,
)


def test_hopfield_recall():
    print("-- Hopfield recovers stored and lightly-corrupted patterns")
    rng = np.random.default_rng(0)
    patterns = rng.choice([-1, 1], size=(3, 64))
    net = HopfieldNetwork(64)
    net.train(patterns)
    # an exact pattern is a fixed point
    exact = all(np.array_equal(net.recall(p, rng=rng), p) for p in patterns)
    check("stored patterns are fixed points", exact)
    # a lightly corrupted pattern relaxes back
    p = patterns[0]
    recovered = sum(np.array_equal(net.recall(corrupt(p, 5, rng), rng=rng), p)
                    for _ in range(50))
    check("light corruption is recovered most of the time", recovered >= 45,
          f"{recovered}/50")


def test_hopfield_capacity():
    print("-- recovery degrades as the network is overloaded")
    cap = capacity_curve(64, seed=1, trials=100)
    low = cap[0]["recovery_rate"]
    high = cap[-1]["recovery_rate"]
    check("recovery is near-perfect at low load", low >= 0.95, str(low))
    check("recovery collapses past capacity", high < 0.5, str(high))


def test_perceptron_separable():
    print("-- perceptron converges on linearly separable data")
    X, y = linearly_separable(1)
    r = Perceptron(2, seed=1).fit(X, y)
    check("perceptron converges", r["converged_epoch"] is not None)
    check("perceptron is perfectly accurate when separable",
          r["final_accuracy"] == 1.0, str(r["final_accuracy"]))


def test_adaline_separable():
    print("-- ADALINE minimises MSE and classifies separable data")
    X, y = linearly_separable(1)
    r = Adaline(2, seed=1).fit(X, y)
    check("ADALINE reaches full accuracy when separable",
          r["final_accuracy"] == 1.0, str(r["final_accuracy"]))
    check("ADALINE's MSE decreases", r["mse_history"][0] > r["mse_history"][-1])


def test_xor_wall():
    print("-- neither single-layer unit can solve XOR")
    X, y = xor_problem()
    p = Perceptron(2, seed=1).fit(X, y, epochs=200)
    a = Adaline(2, seed=1).fit(X, y, epochs=500)
    check("perceptron never converges on XOR", p["converged_epoch"] is None)
    check("perceptron is below perfect on XOR", p["final_accuracy"] < 1.0,
          str(p["final_accuracy"]))
    check("ADALINE is below perfect on XOR", a["final_accuracy"] < 1.0,
          str(a["final_accuracy"]))


def main() -> int:
    for t in (test_hopfield_recall, test_hopfield_capacity,
              test_perceptron_separable, test_adaline_separable, test_xor_wall):
        t()
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
