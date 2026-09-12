# Running Cadence

Python 3.11 or newer and NumPy — one dependency, one `pip install`. NumPy is the
only thing the plan allows, and the only thing used: the fuzzy system, the
genetic algorithm, the Hopfield network and the neurons are all written from
scratch.

```bash
python --version              # 3.11+
pip install -r requirements.txt
```

---

## The tests

```bash
python tests/test_trafficsim.py       #  5 checks
python tests/test_fuzzy.py            # 10 checks
python tests/test_optimisers.py       #  5 checks
python tests/test_softcomputing.py    # 11 checks
```

`test_optimisers` includes the headline check: that the genetic algorithm
reaches grid search's exact optimum in fewer than half its evaluations, and that
both beat fixed offsets. `test_trafficsim` includes the fuzzy-vs-best-fixed
comparison. `test_softcomputing` pins Hopfield's recovery and capacity and the
XOR wall.

## The experiments

```bash
python run.py
```

Writes `reports/*.json` — `fuzzy`, `offsets`, `hopfield`, `neurons`. Every figure
in the README and the notebook is read from these; none is typed by hand. The
run takes about half a minute, most of it the exhaustive grid search over
offsets.

---

## What each report contains

- **fuzzy.json** — the swept fixed-green baseline (so the comparison is against
  the *best* fixed split, not an arbitrary one), the fuzzy result, and the
  delay reduction.
- **offsets.json** — fixed, grid-search and genetic-algorithm results on the
  arterial, the GA's evaluation saving, and the scaling figure showing the grid
  exploding while the GA stays cheap.
- **hopfield.json** — the recovery curve versus corruption and the capacity
  curve versus load, with the measured cliff against the 0.138 N theory.
- **neurons.json** — perceptron and ADALINE on separable data, and both failing
  on XOR.

---

## Reproducibility

Everything is seeded (`SEED = 20260911`), and the traffic model is a
deterministic queue simulation, so every number reproduces exactly on any
machine. The genetic algorithm is seeded too, so even the heuristic's path is
identical run to run. The directions of the findings are robust; the exact
percentages depend on the seeded demand profiles, which the reports record in
full.
