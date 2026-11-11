"""Fully connected network whose parameters are a flat vector (for gradient-free optimisers).

Layout of the flat vector (same as the original notebook): all weight matrices first
(row-major, shape [in, out]), then all bias vectors.
"""
from __future__ import annotations

import numpy as np


class FlatMLP:
    def __init__(self, input_size: int, hidden: list[int], n_classes: int = 2):
        self.sizes = [input_size, *hidden, n_classes]
        self.w_shapes = [(a, b) for a, b in zip(self.sizes[:-1], self.sizes[1:])]
        self.b_sizes = list(self.sizes[1:])
        self.n_weights = sum(a * b for a, b in self.w_shapes)
        self.dim = self.n_weights + sum(self.b_sizes)

    def unpack(self, theta: np.ndarray):
        Ws, bs, i = [], [], 0
        for a, b in self.w_shapes:
            Ws.append(theta[i:i + a * b].reshape(a, b)); i += a * b
        for b in self.b_sizes:
            bs.append(theta[i:i + b]); i += b
        return Ws, bs

    def logits(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        Ws, bs = self.unpack(theta.astype(np.float32, copy=False))
        A = X
        for k, (W, b) in enumerate(zip(Ws, bs)):
            A = A @ W + b
            if k < len(Ws) - 1:
                np.maximum(A, 0, out=A)
        return A

    def predict_proba(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        Z = self.logits(theta, X).astype(np.float64)
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.logits(theta, X).argmax(axis=1)


class Fitness:
    """Class-weighted cross-entropy + L2 penalty (lambda / 2m) * sum(W^2) -- the notebook's loss.

    Cross-entropy uses a numerically stable log-softmax instead of clipping probabilities at
    1e-12, so saturated wrong predictions are not capped at 27.6 nats.
    Counts evaluations so every optimiser can be checked against the same budget.
    """

    def __init__(self, net: FlatMLP, X: np.ndarray, y: np.ndarray, l2_lambda: float = 1e-3,
                 class_weights: np.ndarray | None = None):
        self.net, self.X, self.y = net, X.astype(np.float32), y.astype(np.int64)
        self.l2 = l2_lambda
        m = len(y)
        w = np.ones(m) if class_weights is None else np.asarray(class_weights)[self.y]
        self.sample_w = w / m
        self.n_evals = 0

    def components(self, theta: np.ndarray) -> tuple[float, float]:
        Z = self.net.logits(theta, self.X).astype(np.float64)
        Z -= Z.max(axis=1, keepdims=True)
        logp = Z[np.arange(len(self.y)), self.y] - np.log(np.exp(Z).sum(axis=1))
        ce = float(-(self.sample_w * logp).sum())
        reg = float(self.l2 / (2 * len(self.y)) * np.dot(theta[:self.net.n_weights], theta[:self.net.n_weights]))
        return ce, reg

    def __call__(self, theta: np.ndarray) -> float:
        self.n_evals += 1
        ce, reg = self.components(theta)
        f = ce + reg
        return f if np.isfinite(f) else np.inf


def balanced_class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=2).astype(float)
    return len(y) / (len(counts) * counts)


def value_and_grad(fit: "Fitness", theta: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact loss and gradient of Fitness (weighted CE + L2) by backpropagation (numpy)."""
    net = fit.net
    Ws, bs = net.unpack(theta.astype(np.float64))
    acts, A = [fit.X.astype(np.float64)], fit.X.astype(np.float64)
    for k, (W, b) in enumerate(zip(Ws, bs)):
        Z = A @ W + b
        A = np.maximum(Z, 0) if k < len(Ws) - 1 else Z
        acts.append(A)
    Z = acts[-1] - acts[-1].max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    m = len(fit.y)
    ce = float(-(fit.sample_w * np.log(P[np.arange(m), fit.y] + 1e-300)).sum())
    reg = fit.l2 / (2 * m) * sum(float((W ** 2).sum()) for W in Ws)
    G = P.copy(); G[np.arange(m), fit.y] -= 1; G *= fit.sample_w[:, None]
    gW, gb = [None] * len(Ws), [None] * len(Ws)
    for k in range(len(Ws) - 1, -1, -1):
        gW[k] = acts[k].T @ G + fit.l2 / m * Ws[k]
        gb[k] = G.sum(axis=0)
        if k > 0:
            G = (G @ Ws[k].T) * (acts[k] > 0)
    grad = np.concatenate([g.ravel() for g in gW] + [g.ravel() for g in gb])
    return ce + reg, grad


def glorot_scale(net: FlatMLP, mult: float = 1.0) -> np.ndarray:
    """Per-coordinate scale s so that theta = s * z with z in [-1, 1]^D spans `mult` times the
    Glorot-uniform range sqrt(6 / (fan_in + fan_out)) of each layer (biases use their layer's scale)."""
    s = np.empty(net.dim); i = 0
    lims = [mult * np.sqrt(6.0 / (a + b)) for a, b in net.w_shapes]
    for (a, b), lim in zip(net.w_shapes, lims):
        s[i:i + a * b] = lim; i += a * b
    for b, lim in zip(net.b_sizes, lims):
        s[i:i + b] = lim; i += b
    return s


class ScaledObjective:
    """Fitness in normalised coordinates z: f(s * z). Evaluations are counted by the wrapped Fitness."""

    def __init__(self, fit: Fitness, scale: np.ndarray):
        self.fit, self.scale = fit, scale

    def __call__(self, z: np.ndarray) -> float:
        return self.fit(z * self.scale)

    @property
    def n_evals(self) -> int:
        return self.fit.n_evals
