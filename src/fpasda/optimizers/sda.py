"""Seed-dispersal search as implemented in the original notebook (Sec. 4.4.2 of the manuscript).

Each seed picks one operator per iteration:
  wind   (p_wind)   x <- x + L * (x_best - x)              (Levy step towards the best)
  gravity(p_grav)   x <- x + U(-step, step)                 (local random walk)
  animal (p_animal) x <- x + eps * (x_r - x), eps~U(0,1)    (move towards a random seed)
Greedy, in-place replacement.

NOTE: the manuscript attributes this algorithm to Sulaiman et al. (2015) and Abbasian &
Moulaei (2020); those sources could not be verified. The operators are documented here
exactly as implemented so they can be described as the authors' own if necessary.
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer, Result, levy


class SDA(Optimizer):
    name = "SDA"

    def step(self, f, pop, fit, best, rng):
        p_wind, p_grav, _ = self.params.get("dispersal_probs", (0.7, 0.15, 0.15))
        s = self.params.get("local_step", 0.1)
        n, dim = pop.shape
        for i in range(n):
            r = rng.random()
            if r < p_wind:
                cand = pop[i] + levy(rng, dim) * (best[0] - pop[i])
            elif r < p_wind + p_grav:
                cand = pop[i] + rng.uniform(-s, s, dim)
            else:
                k = rng.integers(n)
                cand = pop[i] + rng.uniform(0, 1, dim) * (pop[k] - pop[i])
            cand = self.clip(cand)
            fc = f(cand)
            if fc < fit[i]:
                pop[i], fit[i] = cand, fc
                if fc < best[1]:
                    best[0], best[1] = cand.copy(), fc

    def run(self, f, dim, rng, iterations=None, start=None) -> Result:
        iterations = self.iterations if iterations is None else iterations
        if start is None:
            pop, fit = self.init_population(f, dim, rng)
        else:
            pop, fit = start
        b = int(np.argmin(fit))
        best = [pop[b].copy(), float(fit[b])]
        hist = [best[1]]
        for _ in range(iterations):
            self.step(f, pop, fit, best, rng)
            hist.append(best[1])
        return Result(best[0], best[1], hist, getattr(f, "n_evals", -1), {"population": (pop, fit)})
