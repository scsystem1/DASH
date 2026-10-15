"""Prequential density stacking with leave-one-evidence-out robustness."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import erfcx, logsumexp, ndtr


class RobustNLLSurrogate:
    def __init__(self, factories, weights):
        self.factories = factories
        self.weights = np.asarray(weights, dtype=float)
        self.models = []

    def fit(self, candidates, y):
        self.models = []
        for factory in self.factories:
            model = factory()
            model.fit(candidates, y)
            self.models.append(model)

    def component_predictions(self, candidates, *, observation_noise=False):
        from pools.component_pools import CoCaBOGPSurrogate

        predictions = []
        for model in self.models:
            if observation_noise and isinstance(model, CoCaBOGPSurrogate):
                import torch
                with torch.no_grad():
                    posterior = model.model.posterior(model.encode_candidates(candidates), observation_noise=True)
                    mean = posterior.mean.squeeze(-1).cpu().numpy()
                    std = posterior.variance.squeeze(-1).clamp_min(1e-12).sqrt().cpu().numpy()
            else:
                mean, std = model.predict(candidates)
            predictions.append((mean, std))
        means = np.asarray([p[0] for p in predictions], dtype=float)
        stds = np.maximum(np.asarray([p[1] for p in predictions], dtype=float), 1e-6)
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)):
            raise ValueError("Non-finite robust_nll component predictions")
        return means, stds

    def predict(self, candidates):
        means, stds = self.component_predictions(candidates)
        mean = self.weights @ means
        variance = self.weights @ (stds**2 + (means - mean)**2)
        return mean, np.sqrt(np.maximum(variance, 1e-12))


def update_weights(log_densities, previous, kl_penalty=1.0):
    """Minimize worst leave-one-out mean log ratio + lambda KL(w || old).

    The one-record startup uses its single loss. An epigraph keeps the small
    constrained problem smooth; failed/non-improving solves retain old weights.
    """
    old = np.maximum(np.asarray(previous, dtype=float), 1e-6)
    old /= old.sum()
    logp = np.asarray([row for row in log_densities if row is not None], dtype=float)
    if kl_penalty <= 0 or not np.isfinite(kl_penalty):
        raise ValueError("robust_nll_kl_penalty must be finite and positive")
    if not logp.size:
        return old
    if logp.ndim != 2 or logp.shape[1] != len(old) or not np.all(np.isfinite(logp)):
        raise ValueError("Invalid prequential log densities")
    # Per-observation offsets cancel in likelihood ratios. Remove them before
    # subtraction so very small densities cannot erase meaningful differences.
    logp = logp - logp.max(axis=1, keepdims=True)
    n, k = logp.shape
    q = (np.ones((n, n)) - np.eye(n)) / (n - 1) if n > 1 else np.ones((1, 1))
    baseline = logsumexp(logp + np.log(old), axis=1)

    def risks(w):
        mixture = logsumexp(logp + np.log(w), axis=1)
        return q @ (baseline - mixture), -q @ np.exp(logp - mixture[:, None])

    def objective(z):
        return z[-1] + kl_penalty * np.sum(z[:k] * np.log(z[:k] / old))

    def gradient(z):
        return np.r_[kl_penalty * (np.log(z[:k] / old) + 1), 1.0]

    result = minimize(
        objective, np.r_[old, 0.0], jac=gradient, method="SLSQP",
        bounds=[(1e-6, 1.0)] * k + [(None, None)],
        constraints=[
            {"type": "eq", "fun": lambda z: z[:k].sum() - 1,
             "jac": lambda z: np.r_[np.ones(k), 0.]},
            {"type": "ineq", "fun": lambda z: z[-1] - risks(z[:k])[0],
             "jac": lambda z: np.column_stack((-risks(z[:k])[1], np.ones(n)))},
        ], options={"maxiter": 150, "ftol": 1e-10},
    )
    if result.success and np.all(np.isfinite(result.x)):
        w = np.maximum(result.x[:k], 1e-6)
        w /= w.sum()
        value = risks(w)[0].max() + kl_penalty * np.sum(w * np.log(w / old))
        if value < -1e-10:
            return w
    return old


def consume_predictions(cache, observations, candidate_key, direction):
    """Consume only outcomes arriving after the cached pre-observation forecast."""
    history = list(cache.get("history", []))[-5:]
    pending = cache.get("pending", {})
    for observation in observations[int(cache.get("observed_count", len(observations))):]:
        prediction = pending.get(candidate_key(observation.get("candidate", {})))
        if prediction is None:
            # This round still ages the window; never backfill with a refit.
            history.append(None)
            continue
        mean, std = np.asarray(prediction["mean"]), np.asarray(prediction["std"])
        y = float(observation["result"]) * (-1 if direction == "minimize" else 1)
        logp = -.5 * ((y - mean) / std)**2 - np.log(std) - .5 * np.log(2 * np.pi)
        if not np.all(np.isfinite(logp)):
            raise ValueError("Non-finite robust_nll observation likelihood")
        history.append(logp.tolist())
    return history[-5:]


def gaussian_log_ei(mean, std, best):
    """Log Gaussian EI without underflow/cancellation in the negative tail."""
    std = np.maximum(np.asarray(std, dtype=float), 1e-8)
    z = (np.asarray(mean, dtype=float) - best) / std
    out = np.empty_like(z)
    upper = z >= -1
    out[upper] = np.log(np.exp(-.5*z[upper]**2) / np.sqrt(2*np.pi) + z[upper]*ndtr(z[upper]))
    tail = ~upper
    a = -z[tail]
    correction = np.empty_like(a)
    moderate = a <= 50
    log_ratio = np.log(a[moderate]) + .5*np.log(np.pi/2) + np.log(erfcx(a[moderate]/np.sqrt(2)))
    correction[moderate] = np.log(-np.expm1(log_ratio))
    inv_sq = 1 / a[~moderate]**2
    correction[~moderate] = -2*np.log(a[~moderate]) + np.log1p(
        inv_sq * (-3 + inv_sq * (15 + inv_sq * (-105 + inv_sq * 945))))
    out[tail] = -.5*a**2 - .5*np.log(2*np.pi) + correction
    return np.log(std) + out
