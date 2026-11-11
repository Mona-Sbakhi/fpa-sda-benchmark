"""Redesigned hybrid and state-of-the-art comparison optimisers (revision track 2).

All methods spend exactly the same number of fitness evaluations as the benchmark optimisers:
budget = pop_size * (iterations + 1). Each records the best-so-far fitness once per
`pop_size` evaluations, so convergence curves share the benchmark's x-axis.

References
----------
SHADE / L-SHADE : Tanabe & Fukunaga, CEC 2013 / CEC 2014 (success-history adaptation of F, CR;
                  linear population size reduction; current-to-pbest/1 with archive, from JADE:
                  Zhang & Sanderson, IEEE TEC 13(5), 2009).
AOS             : adaptive operator selection by probability matching (Thierens, GECCO 2005).
OpenAI-ES       : Salimans et al., arXiv:1703.03864 (2017); antithetic sampling, rank shaping;
                  ES at LLM scale: Qiu et al., arXiv:2509.24372 (2025).
ZO-SGD / MeZO   : Spall (1992) SPSA; Malladi et al., NeurIPS 2023 (arXiv:2305.17333).
"""
from __future__ import annotations

import numpy as np

from .base import Optimizer, Result, levy


class _Budget:
    """Tracks evaluations and records best-so-far once per `stride` evaluations."""

    def __init__(self, f, budget, stride):
        self.f, self.budget, self.stride = f, budget, stride
        self.used, self.best_f, self.best_x, self.history = 0, np.inf, None, []

    def __call__(self, x):
        fx = self.f(x)
        self.used += 1
        if fx < self.best_f:
            self.best_f, self.best_x = fx, x.copy()
        if self.used % self.stride == 0:
            self.history.append(float(self.best_f))
        return fx

    @property
    def left(self):
        return self.budget - self.used


def _lehmer(values, weights):
    w = weights / weights.sum()
    return float((w * values ** 2).sum() / max((w * values).sum(), 1e-12))


class LSHADE(Optimizer):
    """L-SHADE: current-to-pbest/1/bin, success-history F/CR memories, archive, LPSR."""
    name = "L-SHADE"

    def run(self, f, dim, rng) -> Result:
        n0 = self.pop_size; nmin = self.params.get("n_min", 4); H = self.params.get("memory", 6)
        p_best = self.params.get("p_best", 0.11)
        B = _Budget(f, n0 * (self.iterations + 1), n0)
        pop = rng.uniform(self.lb, self.ub, (n0, dim)); fit = np.array([B(x) for x in pop])
        MF, MCR, k, archive = np.full(H, 0.5), np.full(H, 0.5), 0, []
        while B.left > 0:
            n = len(pop); SF, SCR, dF = [], [], []
            order = np.argsort(fit); top = order[:max(2, int(round(p_best * n)))]
            new_pop, new_fit = pop.copy(), fit.copy()
            for i in range(n):
                if B.left <= 0:
                    break
                r = rng.integers(H)
                F = -1.0
                while F <= 0:
                    F = MF[r] + 0.1 * rng.standard_cauchy()
                F = min(F, 1.0)
                CR = float(np.clip(rng.normal(MCR[r], 0.1), 0, 1))
                pb = pop[rng.choice(top)]
                r1 = rng.choice([j for j in range(n) if j != i])
                pool = np.vstack([pop, np.array(archive)]) if archive else pop
                r2 = rng.integers(len(pool))
                v = pop[i] + F * (pb - pop[i]) + F * (pop[r1] - pool[r2])
                mask = rng.random(dim) < CR; mask[rng.integers(dim)] = True
                u = self.clip(np.where(mask, v, pop[i]))
                fu = B(u)
                if fu <= fit[i]:
                    if fu < fit[i]:
                        SF.append(F); SCR.append(CR); dF.append(fit[i] - fu); archive.append(pop[i].copy())
                    new_pop[i], new_fit[i] = u, fu
            pop, fit = new_pop, new_fit
            if SF:
                w = np.array(dF)
                MF[k] = _lehmer(np.array(SF), w); MCR[k] = float((w / w.sum() * np.array(SCR)).sum())
                k = (k + 1) % H
            target = int(round(n0 - (n0 - nmin) * B.used / B.budget))
            if target < len(pop):
                keep = np.argsort(fit)[:max(target, nmin)]
                pop, fit = pop[keep], fit[keep]
            while len(archive) > len(pop):
                archive.pop(rng.integers(len(archive)))
        return Result(B.best_x, B.best_f, B.history, getattr(f, "n_evals", -1))


class AdaptivePlantHybrid(Optimizer):
    """Redesigned FPA-SDA hybrid ("APH").

    Replaces the fixed two-phase schedule (FPA then SDA at tau) with operator-level hybridisation:
      O1 global pollination   v = x + F * clip(L) * (g* - x)              (FPA, Levy)
      O2 local pollination    v = x + F * (x_j - x_k)                     (FPA)
      O3 animal dispersal     v = x + F * (x_pbest - x) + F * (x_r1 - a_r2)  (SDA, as current-to-pbest)
      O4 gravity dispersal    v = x + F * s * N(0, 1), s = per-dim population spread (SDA)
    followed by binomial crossover (rate CR), so only part of the weight vector moves per trial.
    Operator probabilities: adaptive operator selection by probability matching on relative
    improvement. F and CR: SHADE success-history memories. Population: linear size reduction.
    Parameter-free apart from the shared defaults below.
    """
    name = "APH"

    def run(self, f, dim, rng) -> Result:
        n0 = self.pop_size; nmin = self.params.get("n_min", 4); H = self.params.get("memory", 6)
        p_best = self.params.get("p_best", 0.11); p_min = self.params.get("p_min", 0.05)
        alpha = self.params.get("aos_rate", 0.3); levy_clip = self.params.get("levy_clip", 3.0)
        ops = self.params.get("operators", ["global", "local", "animal", "gravity"])
        K = len(ops)
        B = _Budget(f, n0 * (self.iterations + 1), n0)
        pop = rng.uniform(self.lb, self.ub, (n0, dim)); fit = np.array([B(x) for x in pop])
        MF, MCR, km, archive = np.full(H, 0.5), np.full(H, 0.5), 0, []
        q = np.ones(K) / K
        op_use = np.zeros(K)
        while B.left > 0:
            n = len(pop); SF, SCR, dF = [], [], []
            reward, count = np.zeros(K), np.zeros(K)
            probs = p_min + (1 - K * p_min) * q / q.sum()
            order = np.argsort(fit); top = order[:max(2, int(round(p_best * n)))]
            g = pop[order[0]].copy()
            spread = pop.std(axis=0) + 1e-8
            new_pop, new_fit = pop.copy(), fit.copy()
            for i in range(n):
                if B.left <= 0:
                    break
                r = rng.integers(H)
                F = -1.0
                while F <= 0:
                    F = MF[r] + 0.1 * rng.standard_cauchy()
                F = min(F, 1.0)
                CR = float(np.clip(rng.normal(MCR[r], 0.1), 0, 1))
                o = int(rng.choice(K, p=probs)); op = ops[o]; op_use[o] += 1
                x = pop[i]
                if op == "global":
                    v = x + F * np.clip(levy(rng, dim), -levy_clip, levy_clip) * (g - x)
                elif op == "local":
                    j, kk = rng.choice(n, 2, replace=False); v = x + F * (pop[j] - pop[kk])
                elif op == "animal":
                    pb = pop[rng.choice(top)]; r1 = rng.choice([j for j in range(n) if j != i])
                    pool = np.vstack([pop, np.array(archive)]) if archive else pop
                    v = x + F * (pb - x) + F * (pop[r1] - pool[rng.integers(len(pool))])
                else:  # gravity
                    v = x + F * spread * rng.standard_normal(dim)
                if self.params.get("crossover", True):
                    mask = rng.random(dim) < CR; mask[rng.integers(dim)] = True
                    u = self.clip(np.where(mask, v, x))
                else:
                    u = self.clip(v)
                fu = B(u)
                count[o] += 1
                if fu <= fit[i]:
                    if fu < fit[i]:
                        imp = (fit[i] - fu) / (abs(fit[i]) + 1e-12)
                        reward[o] += imp
                        SF.append(F); SCR.append(CR); dF.append(fit[i] - fu); archive.append(x.copy())
                    new_pop[i], new_fit[i] = u, fu
            pop, fit = new_pop, new_fit
            mean_r = np.divide(reward, count, out=np.zeros(K), where=count > 0)
            if mean_r.sum() > 0:
                q = (1 - alpha) * q + alpha * mean_r / mean_r.sum()
            if SF:
                w = np.array(dF)
                MF[km] = _lehmer(np.array(SF), w); MCR[km] = float((w / w.sum() * np.array(SCR)).sum())
                km = (km + 1) % H
            target = int(round(n0 - (n0 - nmin) * B.used / B.budget))
            if target < len(pop):
                keep = np.argsort(fit)[:max(target, nmin)]
                pop, fit = pop[keep], fit[keep]
            while len(archive) > len(pop):
                archive.pop(rng.integers(len(archive)))
        return Result(B.best_x, B.best_f, B.history, getattr(f, "n_evals", -1),
                      {"operator_share": dict(zip(ops, (op_use / max(op_use.sum(), 1)).round(4).tolist())),
                       "final_operator_probs": dict(zip(ops, probs.round(4).tolist()))})


class OpenAIES(Optimizer):
    """OpenAI-ES: antithetic Gaussian perturbations, centred-rank shaping, Adam on the estimate."""
    name = "OpenAI-ES"

    def run(self, f, dim, rng) -> Result:
        sigma = self.params.get("sigma", 0.02); lr = self.params.get("lr", 0.01)
        half = max(1, (self.pop_size - 1) // 2)
        B = _Budget(f, self.pop_size * (self.iterations + 1), self.pop_size)
        theta = rng.uniform(self.lb, self.ub, dim)
        # initial population of the benchmark: spend pop_size evaluations, start from the best
        init = [theta] + [rng.uniform(self.lb, self.ub, dim) for _ in range(self.pop_size - 1)]
        fi = [B(x) for x in init]; theta = init[int(np.argmin(fi))].copy()
        m = np.zeros(dim); v = np.zeros(dim); t = 0
        while B.left >= 2 * half + 1:
            eps = rng.standard_normal((half, dim))
            fp = np.array([B(self.clip(theta + sigma * e)) for e in eps])
            fm = np.array([B(self.clip(theta - sigma * e)) for e in eps])
            allf = np.concatenate([fp, fm])
            ranks = np.empty(len(allf)); ranks[np.argsort(allf)] = np.arange(len(allf))
            shaped = ranks / (len(allf) - 1) - 0.5           # low fitness (good) -> negative
            w = shaped[:half] - shaped[half:]
            grad = (w[:, None] * eps).sum(0) / (2 * half * sigma)  # ascent direction of loss
            t += 1; m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
            theta = self.clip(theta - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8))
            B(theta)
        while B.left > 0:
            B(theta)
        return Result(B.best_x, B.best_f, B.history, getattr(f, "n_evals", -1))


class ZOSGD(Optimizer):
    """Zeroth-order SGD with two-point SPSA estimates (MeZO-style), normalised step."""
    name = "ZO-SGD"

    def run(self, f, dim, rng) -> Result:
        eps = self.params.get("eps", 1e-3); lr = self.params.get("lr", 1e-3)
        B = _Budget(f, self.pop_size * (self.iterations + 1), self.pop_size)
        init = [rng.uniform(self.lb, self.ub, dim) for _ in range(self.pop_size)]
        fi = [B(x) for x in init]; theta = init[int(np.argmin(fi))].copy()
        while B.left >= 2:
            z = rng.standard_normal(dim)
            g = (B(self.clip(theta + eps * z)) - B(self.clip(theta - eps * z))) / (2 * eps)
            theta = self.clip(theta - lr * g * z)
        while B.left > 0:
            B(theta)
        return Result(B.best_x, B.best_f, B.history, getattr(f, "n_evals", -1))
