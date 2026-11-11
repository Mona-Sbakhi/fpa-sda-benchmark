"""Shared machinery for the gradient-free optimisers.

Every optimiser:
  * minimises f(theta) inside box bounds [lb, ub] (solutions are clipped),
  * starts from a uniform random population of `pop_size`,
  * spends exactly pop_size * (iterations + 1) fitness evaluations
    (initial population + one new candidate per individual per iteration),
  * records the best-so-far fitness after initialisation and after every iteration,
  * draws all randomness from the numpy Generator it is given (reproducible per seed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    best_x: np.ndarray
    best_f: float
    history: list[float]
    n_evals: int
    extra: dict = field(default_factory=dict)


def levy(rng: np.random.Generator, dim: int, beta: float = 1.5) -> np.ndarray:
    """Levy-stable step by Mantegna's algorithm (Mantegna, Phys. Rev. E 49, 4677, 1994)."""
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1 / beta)
    u = rng.normal(0.0, sigma, dim)
    v = rng.normal(0.0, 1.0, dim)
    return u / np.abs(v) ** (1 / beta)


class Optimizer:
    name = "base"

    def __init__(self, pop_size: int = 45, iterations: int = 150, bounds=(-1.0, 1.0), **params):
        self.pop_size, self.iterations = int(pop_size), int(iterations)
        self.lb, self.ub = map(float, bounds)
        self.params = params

    # -- helpers ---------------------------------------------------------------------------
    def clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def init_population(self, f, dim, rng):
        pop = rng.uniform(self.lb, self.ub, (self.pop_size, dim))
        fit = np.array([f(x) for x in pop])
        return pop, fit

    def run(self, f, dim: int, rng: np.random.Generator) -> Result:  # pragma: no cover
        raise NotImplementedError
