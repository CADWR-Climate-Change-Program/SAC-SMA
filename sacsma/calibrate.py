"""GA calibration scaffold (Wang 1991 real-coded genetic algorithm).

This provides a reusable bounded real-coded GA engine and a KGE objective
helper.  What is intentionally left as TODO for a later milestone:

  * the cluster -> HRU parameter expansion (PET-by-veg, SMA-by-soil,
    snow/routing per basin) that reduces the ~6000-HRU field to the 203
    calibrated free parameters, and
  * the *pooled* multi-basin objective (mean KGE across gauges) used by
    Wi & Steinschneider.

The :class:`GAConfig` / :func:`run_ga` pieces are functional now so a
single-vector calibration can be driven immediately; wiring the SAC-SMA
forward model as the fitness function is the remaining work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .metrics import kge


@dataclass
class GAConfig:
    pop_size: int = 250          # matches the archived 15-CDEC run
    n_generations: int = 100
    tournament_k: int = 3
    crossover_rate: float = 0.9
    blend_alpha: float = 0.5     # BLX-alpha crossover
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1  # fraction of each parameter's range
    elitism: int = 1
    seed: int = 0


def kge_objective(sim: np.ndarray, obs: np.ndarray) -> float:
    """Maximise KGE -> GA fitness (already 'higher is better')."""
    value = kge(sim, obs)
    return -1e9 if not np.isfinite(value) else value


def run_ga(
    fitness: Callable[[np.ndarray], float],
    lower: np.ndarray,
    upper: np.ndarray,
    config: GAConfig | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Maximise ``fitness`` over a box ``[lower, upper]``.

    Returns ``(best_params, best_fitness, history)`` where ``history`` is the
    best fitness per generation.  ``fitness`` should return higher-is-better
    (use :func:`kge_objective`).
    """
    cfg = config or GAConfig()
    rng = np.random.default_rng(cfg.seed)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    span = upper - lower
    d = lower.size

    pop = lower + rng.random((cfg.pop_size, d)) * span
    fit = np.array([fitness(ind) for ind in pop])
    history: list[float] = []

    def tournament() -> np.ndarray:
        idx = rng.integers(0, cfg.pop_size, cfg.tournament_k)
        return pop[idx[np.argmax(fit[idx])]]

    for _gen in range(cfg.n_generations):
        elite_idx = np.argsort(fit)[::-1][: cfg.elitism]
        new_pop = [pop[i].copy() for i in elite_idx]
        while len(new_pop) < cfg.pop_size:
            p1, p2 = tournament(), tournament()
            if rng.random() < cfg.crossover_rate:
                lo = np.minimum(p1, p2)
                hi = np.maximum(p1, p2)
                ext = cfg.blend_alpha * (hi - lo)
                child = rng.uniform(lo - ext, hi + ext)
            else:
                child = p1.copy()
            mut = rng.random(d) < cfg.mutation_rate
            child = child + mut * rng.normal(0.0, cfg.mutation_scale * span)
            child = np.clip(child, lower, upper)
            new_pop.append(child)
        pop = np.array(new_pop[: cfg.pop_size])
        fit = np.array([fitness(ind) for ind in pop])
        history.append(float(fit.max()))

    best = int(np.argmax(fit))
    return pop[best], float(fit[best]), history
