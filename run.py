"""Run every experiment and write reports/*.json - the evidence behind the
README and the notebook.

    python run.py

Everything is seeded, so the results are identical run to run. NumPy is the
only dependency.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cadence.trafficsim import simulate_arterial, simulate_isolated       # noqa: E402
from cadence.fuzzy import FuzzyGreenController, FixedGreenController       # noqa: E402
from cadence import gridsearch, genetic                                   # noqa: E402
from cadence.hopfield import recovery_curve, capacity_curve               # noqa: E402
from cadence.neurons import (                                             # noqa: E402
    Perceptron, Adaline, linearly_separable, xor_problem,
)

SEED = 20260911
REPORTS = ROOT / "reports"

ARTERIAL = dict(cycle=60, arterial_green=30, travel_time=15, inflow=0.5,
                cross_inflow=0.15, sat_flow=1.0, horizon=900)


def write(name: str, payload: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote reports/{name}")


# ------------------------------------------------------------------- fuzzy
def run_fuzzy() -> None:
    print("[fuzzy] adaptive green time versus the best-tuned fixed split")
    # Asymmetric, time-varying demand: the main street heavy first, then the
    # cross street - exactly where a fixed split wastes green on an empty road.
    demand = [(0.45, 0.08)] * 6 + [(0.08, 0.45)] * 6

    # The baseline must be the BEST fixed green, not an arbitrary one, or the
    # comparison is a strawman. Sweep the constant green and keep the best.
    fixed_sweep = {}
    best_fixed_delay = float("inf")
    best_fixed_green = None
    for g in range(10, 51, 5):
        r = simulate_isolated(FixedGreenController(g), demand, sat_flow=1.0, seed=SEED)
        fixed_sweep[g] = round(r.total_delay, 1)
        if r.total_delay < best_fixed_delay:
            best_fixed_delay = r.total_delay
            best_fixed_green = g

    fuzzy = simulate_isolated(FuzzyGreenController(), demand, sat_flow=1.0, seed=SEED)
    reduction = (best_fixed_delay - fuzzy.total_delay) / best_fixed_delay
    write("fuzzy.json", {
        "demand_profile": "main-street-heavy then cross-street-heavy",
        "fixed_green_sweep": fixed_sweep,
        "best_fixed_green": best_fixed_green,
        "best_fixed_delay": round(best_fixed_delay, 1),
        "fuzzy": fuzzy.as_dict(),
        "delay_reduction_vs_best_fixed": round(reduction, 4),
        "sample_rules": {
            "my_long_other_short": round(FuzzyGreenController().infer(16, 4), 1),
            "my_short_other_long": round(FuzzyGreenController().infer(2, 14), 1),
            "both_medium": round(FuzzyGreenController().infer(10, 10), 1),
        },
    })


# ---------------------------------------------------------------- offsets
def run_offsets() -> None:
    print("[offsets] fixed vs exhaustive grid search vs genetic algorithm")

    def objective(offsets):
        return simulate_arterial(offsets, **ARTERIAL).total_delay

    n, cycle, step = 4, 60, 5
    fixed_delay = objective([0] * n)

    t = time.perf_counter()
    grid = gridsearch.search(objective, n, cycle, step)
    grid["seconds"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    ga = genetic.optimise(objective, n, cycle, step, seed=SEED)
    ga["seconds"] = round(time.perf_counter() - t, 2)

    matched = ga["best_delay"] <= grid["best_delay"] * 1.001

    # Scaling: at n intersections the grid is (cycle/step)^(n-1) points. Run the
    # GA at a larger n where an exhaustive grid would be prohibitive, to show it
    # still finds a strong solution at roughly constant cost.
    big_n = 6
    grid_points_big = (cycle // step) ** (big_n - 1)

    def objective_big(offsets):
        return simulate_arterial(offsets, **{**ARTERIAL, "horizon": 700}).total_delay

    ga_big = genetic.optimise(objective_big, big_n, cycle, step, seed=SEED,
                              population=30, generations=40)
    fixed_big = objective_big([0] * big_n)

    write("offsets.json", {
        "network": {"intersections": n, "cycle": cycle, "offset_step": step,
                    **{k: ARTERIAL[k] for k in ("arterial_green", "travel_time",
                                                "inflow", "cross_inflow")}},
        "fixed_zero_offsets_delay": round(fixed_delay, 1),
        "grid_search": grid,
        "genetic_algorithm": ga,
        "ga_matched_grid_optimum": matched,
        "ga_evaluation_saving": round(1 - ga["evaluations"] / grid["evaluations"], 3),
        "scaling": {
            "intersections": big_n,
            "grid_points_that_would_be_needed": grid_points_big,
            "ga_evaluations_used": ga_big["evaluations"],
            "ga_delay": ga_big["best_delay"],
            "fixed_delay": round(fixed_big, 1),
            "note": "An exhaustive grid at this size is "
                    f"{grid_points_big:,} simulations; the GA used "
                    f"{ga_big['evaluations']}.",
        },
    })


# --------------------------------------------------------------- hopfield
def run_hopfield() -> None:
    print("[hopfield] recovering corrupted detector readings")
    size = 64
    recovery = recovery_curve(size, n_patterns=3, seed=SEED, trials=200)
    capacity = capacity_curve(size, seed=SEED, trials=200)
    # the theoretical capacity is about 0.138 N; find where recovery drops below
    # 0.9 in the measured curve
    cliff = next((row["load_ratio"] for row in capacity
                  if row["recovery_rate"] < 0.9), None)
    write("hopfield.json", {
        "network_size": size,
        "theoretical_capacity_ratio": 0.138,
        "recovery_curve": recovery,
        "capacity_curve": capacity,
        "measured_capacity_cliff_ratio": cliff,
    })


# ---------------------------------------------------------------- neurons
def run_neurons() -> None:
    print("[neurons] perceptron and ADALINE, and the XOR wall")
    Xsep, ysep = linearly_separable(SEED)
    perc = Perceptron(2, seed=SEED).fit(Xsep, ysep)
    ada = Adaline(2, seed=SEED).fit(Xsep, ysep)

    Xxor, yxor = xor_problem()
    perc_xor = Perceptron(2, seed=SEED).fit(Xxor, yxor, epochs=200)
    ada_xor = Adaline(2, seed=SEED).fit(Xxor, yxor, epochs=500)

    write("neurons.json", {
        "linearly_separable": {
            "perceptron": {k: perc[k] for k in ("converged_epoch", "final_accuracy")},
            "adaline": {k: ada[k] for k in ("final_mse", "final_accuracy")},
        },
        "xor": {
            "perceptron_accuracy": perc_xor["final_accuracy"],
            "perceptron_converged": perc_xor["converged_epoch"] is not None,
            "adaline_accuracy": ada_xor["final_accuracy"],
            "note": "Neither single-layer unit can solve XOR - no line separates it.",
        },
    })


def main() -> int:
    run_fuzzy()
    run_offsets()
    run_hopfield()
    run_neurons()
    print("all reports written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
