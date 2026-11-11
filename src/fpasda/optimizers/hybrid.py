"""Sequential Hybrid FPA -> SDA.

Phase 1: FPA for `transition` iterations.  Phase 2: SDA for the remaining iterations.

Transition modes:
  "transfer" (original notebook): SDA continues from FPA's final population unchanged.
  "reinit"   (manuscript Sec. 4.4.3): the population is re-sampled around FPA's best solution,
             x = clip(g* + N(0, (sigma*(ub-lb))^2)); the best solution itself is kept. The
             re-sampled individuals are evaluated, so this mode costs pop_size-1 extra
             evaluations; to keep the budget equal, phase 2 is shortened by one iteration.
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer, Result
from .fpa import FPA
from .sda import SDA


class HybridFPASDA(Optimizer):
    name = "Hybrid"

    def run(self, f, dim, rng) -> Result:
        tau = int(self.params.get("transition", self.iterations // 2))
        mode = self.params.get("mode", "transfer")
        if not 0 < tau < self.iterations:
            raise ValueError(f"transition={tau} must lie strictly between 0 and iterations={self.iterations}")
        common = dict(pop_size=self.pop_size, bounds=(self.lb, self.ub))
        fpa = FPA(iterations=tau, **common, **self.params.get("fpa", {}))
        sda = SDA(iterations=self.iterations - tau, **common, **self.params.get("sda", {}))

        r1 = fpa.run(f, dim, rng)
        pop, fit = r1.extra["population"]
        phase2_iters = self.iterations - tau
        if mode == "reinit":
            sigma = self.params.get("reinit_sigma", 0.05) * (self.ub - self.lb)
            g = r1.best_x
            pop = self.clip(g + rng.normal(0.0, sigma, pop.shape))
            pop[0] = g
            fit = np.array([r1.best_f] + [f(x) for x in pop[1:]])
            phase2_iters -= 1
        elif mode != "transfer":
            raise ValueError(f"unknown transition mode {mode!r}")

        r2 = sda.run(f, dim, rng, iterations=phase2_iters, start=(pop, fit))
        hist = r1.history + r2.history[1:]
        if mode == "reinit":
            hist.append(hist[-1])  # keep history length = iterations + 1
        best_x, best_f = (r2.best_x, r2.best_f) if r2.best_f <= r1.best_f else (r1.best_x, r1.best_f)
        return Result(best_x, best_f, hist, getattr(f, "n_evals", -1),
                      {"transition": tau, "mode": mode, "phase1_best": r1.best_f})
