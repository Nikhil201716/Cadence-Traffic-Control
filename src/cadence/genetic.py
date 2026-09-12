"""A genetic algorithm for signal offsets, and an honest accounting of what it
costs.

A chromosome is an offset vector (intersection 0 pinned at 0). Fitness is the
negative of total delay from the traffic simulator. Selection is by tournament,
crossover is uniform, and mutation nudges an offset to a nearby grid value. The
algorithm caches every offset vector it has already simulated, so the
`evaluations` it reports is the number of *distinct* simulations run - the fair
quantity to compare against grid search's exhaustive count.

The point of the comparison is not to make the GA win. It is to find out
whether, on a search space small enough for grid search to enumerate, the GA
reaches the same optimum in meaningfully fewer simulations - and to report the
answer either way.
"""

from __future__ import annotations

import random


def optimise(objective, n: int, cycle: int, step: int, *,
             population: int = 20, generations: int = 25,
             tournament: int = 3, mutation_rate: float = 0.2,
             seed: int = 0) -> dict:
    """Minimise `objective(offsets) -> delay` over offset vectors.

    Returns the best offsets found, their delay, the number of distinct
    simulations run (cache hits do not count, since they cost nothing), and the
    generation at which the best was first reached - so a reader can see how
    quickly it converged.
    """
    rng = random.Random(seed)
    values = list(range(0, cycle, step))
    cache: dict[tuple, float] = {}

    def evaluate(chrom: list[int]) -> float:
        key = tuple(chrom)
        if key not in cache:
            cache[key] = objective(chrom)
        return cache[key]

    def random_chrom() -> list[int]:
        return [0, *[rng.choice(values) for _ in range(n - 1)]]

    def mutate(chrom: list[int]) -> list[int]:
        child = list(chrom)
        for i in range(1, n):          # never mutate the pinned first offset
            if rng.random() < mutation_rate:
                child[i] = rng.choice(values)
        return child

    def crossover(a: list[int], b: list[int]) -> list[int]:
        return [0, *[a[i] if rng.random() < 0.5 else b[i] for i in range(1, n)]]

    def select(pop: list[list[int]]) -> list[int]:
        contenders = rng.sample(pop, tournament)
        return min(contenders, key=evaluate)

    pop = [random_chrom() for _ in range(population)]
    best = min(pop, key=evaluate)
    best_delay = evaluate(best)
    best_gen = 0

    for gen in range(1, generations + 1):
        new_pop = [best]               # elitism: carry the best forward
        while len(new_pop) < population:
            child = mutate(crossover(select(pop), select(pop)))
            new_pop.append(child)
        pop = new_pop
        gen_best = min(pop, key=evaluate)
        if evaluate(gen_best) < best_delay:
            best = gen_best
            best_delay = evaluate(gen_best)
            best_gen = gen

    return {"method": "genetic_algorithm", "best_offsets": best,
            "best_delay": round(best_delay, 1),
            "evaluations": len(cache),
            "converged_generation": best_gen,
            "population": population, "generations": generations}
