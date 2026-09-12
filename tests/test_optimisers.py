"""The headline comparison: the genetic algorithm must reach grid search's
optimum, in fewer evaluations, and both must beat fixed offsets.
"""

from _harness import check, summary

from cadence.trafficsim import simulate_arterial
from cadence import gridsearch, genetic

ARTERIAL = dict(cycle=60, arterial_green=30, travel_time=15, inflow=0.5,
                cross_inflow=0.15, sat_flow=1.0, horizon=700)


def objective(offsets):
    return simulate_arterial(offsets, **ARTERIAL).total_delay


def test_grid_finds_progression():
    print("-- grid search finds a coordinated offset better than zero")
    fixed = objective([0, 0, 0, 0])
    grid = gridsearch.search(objective, 4, 60, 5)
    check("grid optimum beats fixed offsets", grid["best_delay"] < fixed,
          f"{grid['best_delay']} vs {fixed}")
    check("grid enumerated the whole space",
          grid["evaluations"] == (60 // 5) ** 3, str(grid["evaluations"]))


def test_ga_matches_grid_cheaper():
    print("-- the GA reaches the grid optimum in fewer evaluations")
    grid = gridsearch.search(objective, 4, 60, 5)
    ga = genetic.optimise(objective, 4, 60, 5, seed=1)
    check("GA reaches the grid optimum",
          ga["best_delay"] <= grid["best_delay"] * 1.001,
          f"GA {ga['best_delay']} vs grid {grid['best_delay']}")
    check("GA used far fewer evaluations than grid",
          ga["evaluations"] < grid["evaluations"] / 2,
          f"{ga['evaluations']} vs {grid['evaluations']}")


def test_ga_caches():
    print("-- the GA does not re-simulate an offset it has already seen")
    calls = {"n": 0}

    def counting_obj(offsets):
        calls["n"] += 1
        return objective(offsets)

    ga = genetic.optimise(counting_obj, 4, 60, 5, seed=2)
    check("distinct simulations equal the reported evaluation count",
          calls["n"] == ga["evaluations"], f"{calls['n']} vs {ga['evaluations']}")


def main() -> int:
    for t in (test_grid_finds_progression, test_ga_matches_grid_cheaper,
              test_ga_caches):
        t()
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
