"""Simulator correctness: progression ordering, and the fuzzy controller
beating the best-tuned fixed split on asymmetric demand.
"""

from _harness import check, summary

from cadence.trafficsim import simulate_arterial, simulate_isolated
from cadence.fuzzy import FuzzyGreenController, FixedGreenController

ARTERIAL = dict(cycle=60, arterial_green=30, travel_time=15, inflow=0.5,
                cross_inflow=0.15, sat_flow=1.0, horizon=900)


def test_progression_ordering():
    print("-- a green wave beats no coordination beats an anti-wave")
    wave = simulate_arterial([0, 15, 30, 45], **ARTERIAL).total_delay
    zero = simulate_arterial([0, 0, 0, 0], **ARTERIAL).total_delay
    anti = simulate_arterial([0, 45, 30, 15], **ARTERIAL).total_delay
    check("green wave has lowest delay", wave < zero, f"{wave} vs {zero}")
    check("anti-wave has highest delay", anti > zero, f"{anti} vs {zero}")


def test_determinism():
    print("-- the simulator is deterministic")
    a = simulate_arterial([0, 10, 20, 30], **ARTERIAL).total_delay
    b = simulate_arterial([0, 10, 20, 30], **ARTERIAL).total_delay
    check("same offsets give the same delay", a == b)


def test_fuzzy_beats_best_fixed():
    print("-- fuzzy control beats the best-tuned fixed split")
    demand = [(0.45, 0.08)] * 6 + [(0.08, 0.45)] * 6
    best_fixed = min(
        simulate_isolated(FixedGreenController(g), demand, sat_flow=1.0, seed=7).total_delay
        for g in range(10, 51, 5)
    )
    fuzzy = simulate_isolated(FuzzyGreenController(), demand, sat_flow=1.0, seed=7).total_delay
    check("fuzzy delay is below the best fixed split", fuzzy < best_fixed,
          f"{fuzzy} vs {best_fixed}")


def test_symmetric_demand_no_worse():
    print("-- on symmetric demand fuzzy should not be worse than fixed")
    demand = [(0.3, 0.3)] * 10
    fixed = simulate_isolated(FixedGreenController(25), demand, sat_flow=1.0, seed=3).total_delay
    fuzzy = simulate_isolated(FuzzyGreenController(), demand, sat_flow=1.0, seed=3).total_delay
    # allow a small margin; the point is fuzzy does not badly regress when
    # there is no asymmetry to exploit
    check("fuzzy within 20% of fixed on symmetric demand",
          fuzzy <= fixed * 1.2, f"{fuzzy} vs {fixed}")


def main() -> int:
    for t in (test_progression_ordering, test_determinism,
              test_fuzzy_beats_best_fixed, test_symmetric_demand_no_worse):
        t()
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
