"""Flower Pollination Algorithm (Yang, UCNC 2012, LNCS 7445:240-249), as in the original notebook.

Global pollination: x_i <- x_i + gamma * L * (g* - x_i)       with probability p
Local pollination:  x_i <- x_i + eps * (x_j - x_k), eps~U(0,1)  otherwise
Greedy, in-place (asynchronous) replacement; the global best is updated immediately.
gamma = 1.0 reproduces the notebook; the manuscript text states gamma = 0.01.
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer, Result, levy


class FPA(Optimizer):
    name = "FPA"

    def step(self, f, pop, fit, best, rng):
        p = self.params.get("switch_probability", 0.8)
        gamma = self.params.get("gamma", 1.0)
        n, dim = pop.shape
        for i in range(n):
            if rng.random() < p:
                cand = pop[i] + gamma * levy(rng, dim) * (best[0] - pop[i])
            else:
                j, k = rng.choice(n, 2, replace=False)
                cand = pop[i] + rng.uniform(0, 1, dim) * (pop[j] - pop[k])
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
