"""Exhaustive grid search over signal offsets - the baseline the genetic
algorithm has to justify itself against.

Only relative offsets matter, so the first intersection is pinned at zero and
the rest are searched over a discrete grid. On a small arterial that grid is
small enough to enumerate completely, which makes grid search the honest
yardstick: it finds the true optimum for the discretisation, at a known cost of
exactly one simulation per grid point. If the genetic algorithm cannot reach
that optimum in fewer simulations, grid search wins, and the report says so.
"""

from __future__ import annotations

import itertools


def offset_grid(n: int, cycle: int, step: int) -> list[list[int]]:
    """All offset vectors with intersection 0 pinned at 0 and the others on a
    grid of `step`-second increments across the cycle."""
    values = list(range(0, cycle, step))
    return [[0, *rest] for rest in itertools.product(values, repeat=n - 1)]


def search(objective, n: int, cycle: int, step: int) -> dict:
    """Evaluate `objective(offsets) -> delay` at every grid point and keep the
    best. Returns the optimum, its delay, and the number of evaluations - which
    for grid search is simply the grid size."""
    best_offsets = None
    best_delay = float("inf")
    evaluations = 0
    for offsets in offset_grid(n, cycle, step):
        delay = objective(offsets)
        evaluations += 1
        if delay < best_delay:
            best_delay = delay
            best_offsets = offsets
    return {"method": "grid_search", "best_offsets": best_offsets,
            "best_delay": round(best_delay, 1), "evaluations": evaluations}
