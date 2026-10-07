"""Nonnegative effort cohorts with empirical-Bayes prefix adaptation.

Offline research model. Each historical rank curve supplies a latent vector of
*terminal score mass fractions*, rather than instantaneous-rate coefficients.
Prediction averages anchored curves under a rank/type-conditioned empirical
prior, updated using only the visible distribution of score increments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from behavior_pace_model import (
    MS_PER_HOUR, PaceModelConfig, _PaceIntegrator, _normalize_observation_array,
    reward_pressure,
)


@dataclass(frozen=True)
class CohortConfig:
    basis: str = "cohort6"
    same_type: bool = True
    strength: float = 16.0
    rank_bandwidth: float = float(np.log(2.0))
    reward_transfer: bool = False

    def __post_init__(self):
        if self.basis not in {"pace3", "cohort6"}:
            raise ValueError("unknown cohort basis")
        if not np.isfinite(self.strength) or self.strength < 0:
            raise ValueError("strength must be finite and nonnegative")
        if not np.isfinite(self.rank_bandwidth) or self.rank_bandwidth <= 0:
            raise ValueError("rank_bandwidth must be finite and positive")


@dataclass(frozen=True)
class CohortTemplate:
    event_id: int
    end_at: int
    event_type: str
    tier: int
    mass: tuple[float, ...]
    fit_rmse: float
    pressure: float = 1.0


def mass_basis(hours, *, duration_hours, start_local_hour, tier, reward_tiers,
               basis="cohort6") -> np.ndarray:
    """Cumulative fraction of each cohort's effort, exactly 0/1 at start/end."""
    h = np.asarray(hours, dtype=float)
    if h.ndim != 1 or not np.all(np.isfinite(h)):
        raise ValueError("hours must be a finite vector")
    if np.any(h < 0) or np.any(h > duration_hours):
        raise ValueError("hours outside event")
    integrator = _PaceIntegrator(duration_hours=duration_hours,
                                 event_start_local_hour=start_local_hour,
                                 config=PaceModelConfig())
    if basis == "pace3":
        c = integrator.cumulative_components(np.append(h, duration_hours),
                                             tier=tier, reward_tiers=reward_tiers)
        return c[:-1] / c[-1]
    if basis != "cohort6":
        raise ValueError("unknown cohort basis")
    x = np.array([integrator.h(float(t)) for t in h]) / integrator.h(duration_hours)
    columns = [x]
    # Exponentially distributed dropout and deadline activation in available time.
    for tau in (0.05, 0.20):
        columns.append(-np.expm1(-x / tau) / -np.expm1(-1.0 / tau))
    for tau in (0.03, 0.10, 0.30):
        columns.append((np.exp((x - 1) / tau) - np.exp(-1 / tau))
                       / -np.expm1(-1 / tau))
    return np.column_stack(columns)


def fit_mass(hours, scores, *, duration_hours, start_local_hour, tier,
             reward_tiers, basis) -> tuple[tuple[float, ...], float]:
    """Fit a convex cumulative score model; final truth is required at T."""
    prefix = _normalize_observation_array(np.column_stack((hours, scores)),
        tier=tier, duration_hours=duration_hours, origin_hour=duration_hours)
    if not np.isclose(prefix.hours[-1], duration_hours, atol=1e-8, rtol=0):
        raise ValueError("completed template requires canonical terminal observation")
    # Time-integrated cumulative error keeps dense tracker periods from dominating.
    x = mass_basis(prefix.hours, duration_hours=duration_hours,
        start_local_hour=start_local_hour, tier=tier, reward_tiers=reward_tiers, basis=basis)
    y = prefix.scores / prefix.scores[-1]
    dt = np.diff(prefix.hours)
    weight = np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2] / duration_hours
    n = x.shape[1]
    def objective(m):
        residual = x @ m - y
        return float(np.sum(weight * residual**2)), 2 * x.T @ (weight * residual)
    fitted = minimize(objective, np.full(n, 1 / n), jac=True, method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints={"type": "eq", "fun": lambda m: m.sum() - 1,
                     "jac": lambda m: np.ones(n)},
        options={"ftol": 1e-12, "maxiter": 200})
    if not fitted.success:
        raise RuntimeError(f"cohort fit failed: {fitted.message}")
    mass = np.maximum(fitted.x, 0)
    mass /= mass.sum()
    return tuple(float(v) for v in mass), float(np.sqrt(objective(mass)[0]))


def predict_cohort_curve(observations, *, tier: int, start_at: int, end_at: int,
                         origin_at: int, event_type: str, reward_tiers: Sequence[int],
                         templates: Sequence[CohortTemplate], config: CohortConfig,
                         forecast_times=None):
    """Return anchored nondecreasing score curves and prefix diagnostics.

    `observations` is an n x 2 array of absolute millisecond times and scores.
    Scores after origin are masked before parsing. No T10 scale or other target
    rank is required. Historical templates must end strictly before target start.
    """
    if not (start_at < origin_at < end_at):
        raise ValueError("origin must be inside the target event")
    raw = np.asarray(observations)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("observations must be n x 2")
    times = np.asarray(raw[:, 0], dtype=float)
    if not np.all(np.isfinite(times)) or np.any(times != np.floor(times)):
        raise ValueError("timestamps must be finite integer milliseconds")
    visible = times <= origin_at
    duration = (end_at - start_at) / MS_PER_HOUR
    prefix = _normalize_observation_array(
        np.column_stack(((times[visible] - start_at) / MS_PER_HOUR, raw[visible, 1])),
        tier=tier, duration_hours=duration,
        origin_hour=(origin_at - start_at) / MS_PER_HOUR)
    if not templates:
        raise ValueError("empty cohort prior")
    if any(t.end_at >= start_at for t in templates):
        raise ValueError("training templates must end strictly before target start")
    chosen = [t for t in templates if not config.same_type or t.event_type == event_type]
    if not chosen:
        raise ValueError(f"no historical template for type {event_type}")
    masses = np.asarray([t.mass for t in chosen], dtype=float)
    expected_dim = 3 if config.basis == "pace3" else 6
    if (masses.shape != (len(chosen), expected_dim) or np.any(~np.isfinite(masses))
            or np.any(masses < 0) or not np.allclose(masses.sum(axis=1), 1, atol=1e-9)):
        raise ValueError("invalid template masses")
    if config.reward_transfer:
        historical_pressure = np.array([t.pressure for t in chosen], dtype=float)
        if np.any(~np.isfinite(historical_pressure)) or np.any(historical_pressure <= 0):
            raise ValueError("invalid historical reward pressure")
        current_pressure = reward_pressure(tier, reward_tiers)
        # Offline masses already contain historical reward-driven deadline
        # effort. Transfer its intensity to the target reward setting exactly
        # once, before normalizing to new terminal mass fractions.
        deadline_start = 2 if config.basis == "pace3" else 3
        masses[:, deadline_start:] *= (current_pressure / historical_pressure)[:, None]
        masses /= masses.sum(axis=1, keepdims=True)
    # Equal event mass, and a smooth rank kernel within each event. A target's
    # observed score is never synthesized from adjacent ranks.
    log_prior = -0.5 * (np.log(np.array([t.tier for t in chosen]) / tier)
                         / config.rank_bandwidth)**2
    event_ids = np.array([t.event_id for t in chosen])
    for e in np.unique(event_ids):
        take = event_ids == e
        log_prior[take] -= logsumexp(log_prior[take])
    log_prior -= logsumexp(log_prior)
    anchor = prefix.last_observed_hour
    knots = np.unique(np.r_[np.arange(0, anchor, 6.0), anchor])
    observed = np.interp(knots, prefix.hours, prefix.scores)
    p = np.diff(observed) / prefix.anchor_score
    start_local = (start_at / MS_PER_HOUR + 8) % 24
    b = mass_basis(knots, duration_hours=duration, start_local_hour=start_local,
                   tier=tier, reward_tiers=reward_tiers, basis=config.basis)
    curves = b @ masses.T
    f_anchor = curves[-1]
    if np.any(f_anchor <= 0):
        raise ValueError("template has zero visible score mass")
    q = np.diff(curves, axis=0) / f_anchor
    positive = p > 0
    divergence = np.sum(p[positive, None]
        * (np.log(p[positive, None]) - np.log(np.maximum(q[positive], 1e-300))), axis=0)
    log_posterior = log_prior - config.strength * (anchor / 24) * divergence
    posterior = np.exp(log_posterior - logsumexp(log_posterior))
    if forecast_times is None:
        forecast_times = np.array([origin_at, end_at], dtype=np.int64)
    forecast_times = np.asarray(forecast_times)
    if (forecast_times.ndim != 1 or not len(forecast_times)
        or np.any(~np.isfinite(forecast_times)) or np.any(forecast_times != np.floor(forecast_times))
        or np.any(np.diff(forecast_times) < 0) or np.any(forecast_times < times[visible].max())
        or np.any(forecast_times > end_at)):
        raise ValueError("forecast timestamps must be ordered between anchor and event end")
    future = mass_basis((forecast_times - start_at) / MS_PER_HOUR,
        duration_hours=duration, start_local_hour=start_local,
        tier=tier, reward_tiers=reward_tiers, basis=config.basis)
    predicted = prefix.anchor_score * ((future @ masses.T) / f_anchor) @ posterior
    return predicted, {
        "anchor_at": int(round(start_at + anchor * MS_PER_HOUR)),
        "anchor_score": prefix.anchor_score,
        "template_count": len(chosen), "history_event_count": len(np.unique(event_ids)),
        "max_training_end_at": max(t.end_at for t in chosen),
        "effective_template_count": float(1 / np.sum(posterior**2)),
        "posterior_mean_mass": (posterior @ masses).tolist(),
        "prefix_max_time": int(times[visible].max()),
    }
