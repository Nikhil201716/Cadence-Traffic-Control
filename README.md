# Cadence — soft-computing traffic signal control

**The honest test the plan demanded: does a genetic algorithm beat exhaustive
grid search at coordinating traffic signals? On a four-intersection arterial it
finds the *same* optimum — a green wave, delay 24,480 — in 177 simulations
against grid search's 1,728, a 90% saving. It matches quality and wins on time,
and the win widens with the network: at six intersections an exhaustive grid
would be 248,832 simulations; the GA used 825.**

Every input here is genuinely fuzzy — a queue is "long" by degree, demand is
"heavy" by degree — so the controllers are soft-computing methods written from
scratch on NumPy alone: a Mamdani fuzzy inference system, a genetic algorithm, a
Hopfield associative memory, and a perceptron and ADALINE. Each is measured
against an honest baseline on a deterministic, seeded traffic simulator. Every
number below is read from `reports/*.json`.

---

## Finding 1 — the genetic algorithm versus the exhaustive baseline

Only relative signal offsets matter, so grid search can enumerate the whole
discretised space and find the true optimum — the yardstick a heuristic must
justify itself against.

| Method | Best delay (veh·s) | Simulations | Notes |
|---|---:|---:|---|
| Fixed (zero offsets) | 44,290 | — | no coordination |
| **Grid search** (exhaustive) | **24,480** | 1,728 | the true optimum |
| **Genetic algorithm** | **24,480** | **177** | same optimum, 90% fewer sims |

The GA reaches grid search's exact optimum — offsets `[0, 15, 30, 45]`, a green
wave matching the 15-second travel time — in a tenth of the simulations. On a
space this small grid search is still perfectly usable (a few seconds), so the
honest reading is that the GA's advantage is real but marginal here. It becomes
decisive as the network grows: the grid is `(cycle/step)^(n−1)` points, so at
six intersections it would be **248,832** simulations, while the GA found a
strong solution in **825**. The heuristic earns its place by scaling, not by
beating the baseline on quality — and where it did not beat it (quality was a
tie), the table says so.

---

## Finding 2 — fuzzy green time versus the best-tuned fixed split

The baseline is not an arbitrary fixed timing but the *best* one: the constant
green was swept and the lowest-delay choice kept, so the comparison is fair
rather than a strawman.

On demand that is heavy on the main street and then heavy on the cross street,
against the best fixed split (green = 25 s, delay 9,392):

| Controller | Total delay (veh·s) |
|---|---:|
| Best-tuned fixed split | 9,392 |
| **Mamdani fuzzy** | **3,917** |

A **58% delay reduction** against the best fixed timing, because the fuzzy
controller lengthens the green for a long queue and yields early for a short
one — something no single fixed split can do when demand shifts. The full
Mamdani pipeline is here: overlapping triangular and shoulder memberships,
a nine-rule base with min-implication, and centroid defuzzification, all from
scratch.

---

## Finding 3 — a Hopfield network recovers corrupted detector readings

Detectors are noisy; bits flip. A Hopfield network stores the typical
traffic-state patterns as attractors and relaxes a noisy reading to the nearest
one.

- **Recovery** is exact up to about a quarter of the bits flipped, then falls
  away — the associative-memory recovery curve, measured over 200 trials per
  point.
- **Capacity** shows the classic cliff: a 64-neuron network recovers reliably
  until the stored patterns exceed a load ratio of about **0.156**, close to the
  theoretical **0.138 N** — after which the attractors interfere and recall
  breaks down.

---

## Finding 4 — the perceptron, ADALINE, and the wall neither can cross

A perceptron and an ADALINE, written from scratch, classify a traffic state
(congested versus free-flowing) from two features. On linearly separable data
the perceptron converges in **4 epochs** to perfect accuracy and ADALINE
descends its mean-squared error to the same. On **XOR**, neither exceeds chance —
no straight line separates it, and a single linear unit can only draw a straight
line. It is the 1969 result that stalled neural networks for a decade,
reproduced rather than recounted, and the honest reason multi-layer networks
exist.

---

## What this does not establish

- **The traffic model is a store-and-forward queue model**, not a
  car-following microsimulation. It captures delay, progression and
  fragmentation of green faithfully; it does not model lane changes,
  acceleration, or turning movements.
- **The GA's win is on time-to-solution, not solution quality** — it matched
  grid search, it did not beat it, and on a tiny network grid search is fine.
  The value is the scaling.
- **The soft-computing components are single, from-scratch implementations**,
  chosen for legibility over the tuned, vectorised versions a production system
  would use.
- **Results are for the seeded demand profiles shown.** The direction of each
  finding is robust; the exact percentages depend on how asymmetric the demand
  is and on the seed.

Every number is read from `reports/*.json`, produced by a real run.

---

## Running it

```bash
pip install -r requirements.txt      # NumPy, nothing else
python tests/test_trafficsim.py      # and test_fuzzy, test_optimisers, test_softcomputing
python run.py                        # every experiment -> reports/*.json
```

Python 3.11+ and NumPy. See [RUN.md](RUN.md) for the full reference and the
[notebook](docs/Cadence-Notebook.pdf) for the derivations — the Mamdani
pipeline, the GA operators, Hopfield's Hebbian storage and capacity bound, and
why XOR defeats a single layer.
