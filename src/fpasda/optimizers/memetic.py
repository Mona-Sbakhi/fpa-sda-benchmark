"""Memetic (Lamarckian) hybrid: a gradient-free optimiser explores, then the best solution is
refined with full-batch Adam on the same objective. `adam_only` runs the same Adam refinement
from the benchmark's random initial population (best of pop_size uniform samples), which isolates
the contribution of the metaheuristic phase.

Each Adam step costs one forward + one backward pass; it is reported separately as
`grad_steps` and is not counted against the evaluation budget of the exploration phase.
"""
from __future__ import annotations

import numpy as np

from ..model import value_and_grad
from .base import Optimizer, Result


def adam(fit, theta, steps, lr, lb, ub, history):
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    best_f, best_x = np.inf, theta.copy()
    for t in range(1, steps + 1):
        f, g = value_and_grad(fit, theta)
        if f < best_f:
            best_f, best_x = f, theta.copy()
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        theta = np.clip(theta - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8), lb, ub)
        history.append(float(min(best_f, history[-1] if history else np.inf)))
    f = fit(theta)
    if f < best_f:
        best_f, best_x = f, theta.copy()
    return best_x, best_f


class Memetic(Optimizer):
    name = "Memetic"

    def run(self, f, dim, rng) -> Result:
        from . import build
        steps = int(self.params.get("adam_steps", 300)); lr = float(self.params.get("lr", 1e-3))
        inner = self.params.get("inner")
        if inner is None:  # Adam-only control: best of the uniform initial population
            pop = rng.uniform(self.lb, self.ub, (self.pop_size, dim))
            fit0 = np.array([f(x) for x in pop]); x0, f0 = pop[int(np.argmin(fit0))], float(fit0.min())
            hist = [f0]
        else:
            frac = float(self.params.get("inner_fraction", 1.0))
            it = max(1, int(round(self.iterations * frac)))
            opt = build(dict(inner), self.pop_size, it, (self.lb, self.ub))
            r = opt.run(f, dim, rng)
            x0, f0, hist = r.best_x, r.best_f, list(r.history)
        hist_adam = [f0]
        bx, bf = adam(f, x0.copy(), steps, lr, self.lb, self.ub, hist_adam)
        return Result(bx, bf, hist + hist_adam[1:], getattr(f, "n_evals", -1),
                      {"phase1_best": f0, "grad_steps": steps})
