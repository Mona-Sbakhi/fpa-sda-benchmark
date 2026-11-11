"""Comparison optimisers under the same budget: pop_size * (iterations + 1) evaluations."""
from __future__ import annotations

import numpy as np

from .base import Optimizer, Result


class RandomSearch(Optimizer):
    """The notebook's 'PCA + Random Search': fresh uniform samples, greedy per-slot replacement."""
    name = "Random"

    def run(self, f, dim, rng) -> Result:
        pop, fit = self.init_population(f, dim, rng)
        hist = [float(fit.min())]
        for _ in range(self.iterations):
            for i in range(self.pop_size):
                c = rng.uniform(self.lb, self.ub, dim)
                fc = f(c)
                if fc < fit[i]:
                    pop[i], fit[i] = c, fc
            hist.append(float(fit.min()))
        b = int(np.argmin(fit))
        return Result(pop[b].copy(), float(fit[b]), hist, getattr(f, "n_evals", -1))


class PSO(Optimizer):
    """Global-best PSO with constriction coefficients (Clerc & Kennedy, IEEE TEC 6(1), 2002)."""
    name = "PSO"

    def run(self, f, dim, rng) -> Result:
        w = self.params.get("w", 0.7298); c1 = self.params.get("c1", 1.49618); c2 = self.params.get("c2", 1.49618)
        vmax = self.params.get("vmax_frac", 0.2) * (self.ub - self.lb)
        x, fit = self.init_population(f, dim, rng)
        v = np.zeros_like(x)
        pb, pbf = x.copy(), fit.copy()
        g = int(np.argmin(pbf)); gb, gbf = pb[g].copy(), float(pbf[g])
        hist = [gbf]
        for _ in range(self.iterations):
            for i in range(self.pop_size):
                v[i] = np.clip(w * v[i] + c1 * rng.random(dim) * (pb[i] - x[i]) + c2 * rng.random(dim) * (gb - x[i]),
                               -vmax, vmax)
                x[i] = self.clip(x[i] + v[i])
                fx = f(x[i])
                if fx < pbf[i]:
                    pb[i], pbf[i] = x[i].copy(), fx
                    if fx < gbf:
                        gb, gbf = x[i].copy(), fx
            hist.append(gbf)
        return Result(gb, gbf, hist, getattr(f, "n_evals", -1))


class DE(Optimizer):
    """DE/rand/1/bin (Storn & Price, J. Global Optim. 11, 1997)."""
    name = "DE"

    def run(self, f, dim, rng) -> Result:
        F = self.params.get("F", 0.5); CR = self.params.get("CR", 0.9)
        pop, fit = self.init_population(f, dim, rng)
        n = self.pop_size
        hist = [float(fit.min())]
        for _ in range(self.iterations):
            for i in range(n):
                a, b, c = rng.choice([k for k in range(n) if k != i], 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mask = rng.random(dim) < CR
                mask[rng.integers(dim)] = True
                trial = self.clip(np.where(mask, mutant, pop[i]))
                ft = f(trial)
                if ft <= fit[i]:
                    pop[i], fit[i] = trial, ft
            hist.append(float(fit.min()))
        b = int(np.argmin(fit))
        return Result(pop[b].copy(), float(fit[b]), hist, getattr(f, "n_evals", -1))


class GA(Optimizer):
    """Real-coded steady-state GA: binary tournament, BLX-alpha crossover, Gaussian mutation,
    offspring replaces the worst individual if better (elitist)."""
    name = "GA"

    def run(self, f, dim, rng) -> Result:
        alpha = self.params.get("blx_alpha", 0.5)
        pm = self.params.get("mutation_rate", 1.0 / dim * 10)
        ms = self.params.get("mutation_sigma", 0.1) * (self.ub - self.lb)
        pop, fit = self.init_population(f, dim, rng)
        n = self.pop_size
        hist = [float(fit.min())]

        def tour():
            i, j = rng.integers(n, size=2)
            return i if fit[i] < fit[j] else j

        for _ in range(self.iterations):
            for _ in range(n):
                p1, p2 = pop[tour()], pop[tour()]
                lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
                span = hi - lo
                child = rng.uniform(lo - alpha * span, hi + alpha * span)
                m = rng.random(dim) < pm
                child[m] += rng.normal(0, ms, int(m.sum()))
                child = self.clip(child)
                fc = f(child)
                w = int(np.argmax(fit))
                if fc < fit[w]:
                    pop[w], fit[w] = child, fc
            hist.append(float(fit.min()))
        b = int(np.argmin(fit))
        return Result(pop[b].copy(), float(fit[b]), hist, getattr(f, "n_evals", -1))
