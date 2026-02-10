# brain/strategy_evolver.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple

from brain.strategy_genome import random_genome, mutate
from brain.strategy_fitness import compute_fitness


EvaluatorFn = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class EvolutionResult:
    best_genome: Dict[str, Any]
    best_fitness: float
    history_best: List[float]          # best fitness per generation
    history_avg: List[float]           # avg fitness per generation


def evolve(
    evaluator: EvaluatorFn,
    generations: int = 10,
    population_size: int = 30,
    elite_k: int = 5,
    mutation_rate: float = 0.10,
    rng: random.Random | None = None,
) -> EvolutionResult:
    """
    Simple genetic evolution loop (v1):
      - init random population
      - evaluate fitness for each genome
      - keep top elite_k
      - refill population by mutating elites
      - repeat for N generations
    """
    rng = rng or random.Random()

    if generations <= 0:
        raise ValueError("generations must be > 0")
    if population_size <= 1:
        raise ValueError("population_size must be > 1")
    if elite_k <= 0 or elite_k >= population_size:
        raise ValueError("elite_k must be in [1, population_size-1]")
    if not (0.0 <= mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0, 1]")

    population: List[Dict[str, Any]] = [random_genome(rng=rng) for _ in range(population_size)]

    best_genome: Dict[str, Any] | None = None
    best_fitness: float = float("-inf")

    history_best: List[float] = []
    history_avg: List[float] = []

    for _gen in range(generations):
        scored: List[Tuple[float, Dict[str, Any]]] = []

        for g in population:
            stats = evaluator(g)
            fit = compute_fitness(stats)
            scored.append((fit, g))

            if fit > best_fitness:
                best_fitness = fit
                best_genome = g

        scored.sort(key=lambda x: x[0], reverse=True)
        fits = [x[0] for x in scored]
        history_best.append(fits[0])
        history_avg.append(sum(fits) / len(fits))

        elites = [deepcopy_genome(x[1]) for x in scored[:elite_k]]

        # refill: keep elites, then mutate randomly chosen elite parents
        new_population: List[Dict[str, Any]] = elites[:]
        while len(new_population) < population_size:
            parent = rng.choice(elites)
            child = mutate(parent, rate=mutation_rate, rng=rng)
            new_population.append(child)

        population = new_population

    assert best_genome is not None
    return EvolutionResult(
        best_genome=best_genome,
        best_fitness=best_fitness,
        history_best=history_best,
        history_avg=history_avg,
    )


def deepcopy_genome(g: Dict[str, Any]) -> Dict[str, Any]:
    # local small deepcopy to avoid importing copy everywhere
    return {k: (v.copy() if isinstance(v, dict) else v) for k, v in g.items()}
