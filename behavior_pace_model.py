"""Compact causal pace model for reward-tier score curves.

The model separates an event's cumulative pace into three non-negative
components:

``sustain``
    ordinary play following the configured circadian availability ``A(t)``;
``launch``
    players who are still pursuing a routine target, represented by
    ``D(t) = 1 - H(t) / H(T)`` where ``H(t) = integral A``;
``deadline``
    terminal response ``P_E(r) U(t)`` at tier ``r``.

For every tier, the score rate is

    v_r(t) = amplitude_r * A(t) * (
        w_sustain + w_launch * D(t) + w_deadline * P_E(r) * U(t)
    ).

The three weights are a non-negative simplex shared within an event.  Training
profiles a separate non-negative amplitude for every tier and minimises a
duration-weighted interval-rate loss.  Tier timestamps never need to align.

Prediction does not refit the weights.  It uses an event-equal prior weight and
anchors each requested tier continuously at its last visible score:

    S_r(t) = S_r(anchor) * C_r(t) / C_r(anchor).

There is deliberately no T1000 dependency, provider fallback, probability
interval, cross-tier speed constraint, or hidden score repair in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize


MS_PER_HOUR = 3_600_000
PACE_COMPONENT_NAMES = ("sustain", "launch", "deadline")
_SIMPLEX_TOLERANCE = 1e-10
_OPTIMIZER_TOLERANCE = 1e-11


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a positive integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _integer_milliseconds(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer millisecond timestamp")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer millisecond timestamp")
    return int(value)


def canonicalize_reward_tiers(
    values: Iterable[int],
    *,
    label: str = "reward_tiers",
) -> tuple[int, ...]:
    """Return a sorted, duplicate-free tuple of explicit positive ranks."""

    if values is None or isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be an explicit integer sequence")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be an explicit integer sequence") from exc
    result = tuple(_positive_int(value, label) for value in raw)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(result))


@dataclass(frozen=True)
class PaceModelConfig:
    """Fixed structural choices shared by training and prediction."""

    circadian_amplitude: float = 0.30
    circadian_peak_hour: float = 21.0
    availability_floor: float = 0.05
    urgency_power: float = 1.6
    urgency_epsilon: float = 0.08
    pressure_floor: float = 0.15
    rank_min: int = 1
    rank_max: int = 100_000
    utc_offset_hours: float = 8.0

    def __post_init__(self) -> None:
        continuous = np.asarray(
            [
                self.circadian_amplitude,
                self.circadian_peak_hour,
                self.availability_floor,
                self.urgency_power,
                self.urgency_epsilon,
                self.pressure_floor,
                self.utc_offset_hours,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(continuous)):
            raise ValueError("pace model config must not contain NaN/Infinity")
        if not 0.0 <= self.circadian_amplitude < 0.95:
            raise ValueError("circadian_amplitude must be in [0, 0.95)")
        if not 0.0 <= self.circadian_peak_hour < 24.0:
            raise ValueError("circadian_peak_hour must be in [0, 24)")
        if not 0.0 < self.availability_floor <= 1.0:
            raise ValueError("availability_floor must be in (0, 1]")
        if not 0.25 <= self.urgency_power <= 8.0:
            raise ValueError("urgency_power must be in [0.25, 8]")
        if not 0.01 <= self.urgency_epsilon <= 0.5:
            raise ValueError("urgency_epsilon must be in [0.01, 0.5]")
        if not 0.0 <= self.pressure_floor <= 1.0:
            raise ValueError("pressure_floor must be in [0, 1]")
        _positive_int(self.rank_min, "rank_min")
        _positive_int(self.rank_max, "rank_max")
        if self.rank_max <= self.rank_min:
            raise ValueError("rank_max must be greater than rank_min")


@dataclass(frozen=True)
class PaceWeights:
    """Non-negative simplex weights for sustain, launch and deadline pace."""

    sustain: float
    launch: float
    deadline: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                _finite_float(self.sustain, "sustain"),
                _finite_float(self.launch, "launch"),
                _finite_float(self.deadline, "deadline"),
            ],
            dtype=float,
        )
        if np.any(values < 0.0):
            raise ValueError("pace weights must be non-negative")
        if not np.isclose(
            float(values.sum()),
            1.0,
            rtol=0.0,
            atol=_SIMPLEX_TOLERANCE,
        ):
            raise ValueError("pace weights must sum to one")

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.sustain, self.launch, self.deadline],
            dtype=float,
        )

    @classmethod
    def from_sequence(cls, values: Sequence[float] | np.ndarray) -> "PaceWeights":
        array = np.asarray(values, dtype=float)
        if array.shape != (3,):
            raise ValueError("pace weight vector must have exactly three values")
        return cls(*array.tolist())


@dataclass(frozen=True)
class PaceTrainingEvent:
    """One completed event with asynchronous per-tier observations.

    ``tier_observations[tier]`` is an ``n x 2`` array whose columns are
    ``hours_since_start`` and cumulative ``score``.  Different tiers may use
    completely different timestamp grids.
    """

    event_id: int
    duration_hours: float
    reward_tiers: Iterable[int]
    tier_observations: Mapping[int, np.ndarray]
    event_start_local_hour: float = 0.0

    def __post_init__(self) -> None:
        _positive_int(self.event_id, "event_id")
        duration = _finite_float(self.duration_hours, "duration_hours")
        if duration <= 0.0:
            raise ValueError("duration_hours must be positive")
        start_hour = _finite_float(
            self.event_start_local_hour,
            "event_start_local_hour",
        )
        if not 0.0 <= start_hour < 24.0:
            raise ValueError("event_start_local_hour must be in [0, 24)")
        rewards = canonicalize_reward_tiers(self.reward_tiers)
        # Materialise one-shot iterables once so validation cannot consume a
        # generator and silently change its meaning during fitting.
        object.__setattr__(self, "reward_tiers", rewards)
        if not isinstance(self.tier_observations, Mapping) or not self.tier_observations:
            raise ValueError("tier_observations must be a non-empty mapping")

    @classmethod
    def from_surface(
        cls,
        *,
        event_id: int,
        start_at: int,
        end_at: int,
        surface: pd.DataFrame,
        reward_tiers: Iterable[int],
        config: PaceModelConfig = PaceModelConfig(),
    ) -> "PaceTrainingEvent":
        """Build an event from a wide, absolute-millisecond tier surface."""

        start = _integer_milliseconds(start_at, "start_at")
        end = _integer_milliseconds(end_at, "end_at")
        if end <= start:
            raise ValueError("end_at must be greater than start_at")
        if not isinstance(surface, pd.DataFrame) or surface.empty:
            raise ValueError("surface must be a non-empty DataFrame")
        raw = surface.copy(deep=True)
        try:
            numeric_index = pd.to_numeric(raw.index, errors="raise").to_numpy(
                dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("surface index must contain integer milliseconds") from exc
        if (
            np.any(~np.isfinite(numeric_index))
            or np.any(numeric_index != np.floor(numeric_index))
        ):
            raise ValueError("surface index must contain integer milliseconds")
        timestamps = numeric_index.astype(np.int64)
        visible = (timestamps >= start) & (timestamps <= end)
        if not np.any(visible):
            raise ValueError("surface has no observations inside the event")
        raw = raw.iloc[np.flatnonzero(visible)].copy()
        timestamps = timestamps[visible]
        if len(timestamps) != len(np.unique(timestamps)):
            raise ValueError("surface has duplicate in-event timestamps")
        order = np.argsort(timestamps, kind="stable")
        raw = raw.iloc[order]
        timestamps = timestamps[order]
        observations: dict[int, np.ndarray] = {}
        seen: set[int] = set()
        for column in raw.columns:
            tier = _positive_int(column, "surface tier")
            if tier in seen:
                raise ValueError("surface contains duplicate tiers")
            seen.add(tier)
            try:
                scores = pd.to_numeric(raw[column], errors="raise").to_numpy(
                    dtype=float
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"surface tier {tier} contains a non-numeric score") from exc
            finite = np.isfinite(scores)
            if not np.any(finite):
                continue
            hours = (timestamps[finite] - start) / MS_PER_HOUR
            observations[tier] = np.column_stack((hours, scores[finite]))
        if not observations:
            raise ValueError("surface has no finite in-event tier observations")
        local_start = (start / MS_PER_HOUR + config.utc_offset_hours) % 24.0
        return cls(
            event_id=event_id,
            duration_hours=(end - start) / MS_PER_HOUR,
            reward_tiers=tuple(reward_tiers),
            tier_observations=observations,
            event_start_local_hour=local_start,
        )


@dataclass(frozen=True)
class NormalizedTierPrefix:
    tier: int
    hours: np.ndarray
    scores: np.ndarray
    last_observed_hour: float
    anchor_score: float


@dataclass(frozen=True)
class NormalizedPaceEvent:
    event_id: int
    duration_hours: float
    event_start_local_hour: float
    reward_tiers: tuple[int, ...]
    tiers: tuple[int, ...]
    tier_prefixes: Mapping[int, NormalizedTierPrefix]


@dataclass(frozen=True)
class PaceEventFit:
    event_id: int
    weights: PaceWeights
    tier_amplitudes: Mapping[int, float]
    tier_relative_losses: Mapping[int, float]
    objective: float
    raw_duration_weighted_sse: float
    interval_count: int
    tier_count: int


@dataclass(frozen=True)
class TierCurvePrediction:
    tier: int
    origin_at: int
    anchor_at: int
    anchor_score: float
    forecast_times: np.ndarray
    scores: np.ndarray
    speeds_per_hour: np.ndarray
    weights: PaceWeights
    diagnostics: Mapping[str, Any]


def _normalize_observation_array(
    values: Any,
    *,
    tier: int,
    duration_hours: float,
    origin_hour: float,
) -> NormalizedTierPrefix:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tier {tier} observations must be an n x 2 array") from exc
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise ValueError(f"tier {tier} observations must be an n x 2 array")
    try:
        hours = np.asarray(array[:, 0], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tier {tier} observation hours must be numeric") from exc
    if np.any(~np.isfinite(hours)):
        raise ValueError(f"tier {tier} observation hours must be finite")
    # The score suffix is intentionally not parsed before the causal cut.
    causal = hours <= origin_hour
    if not np.any(causal):
        raise ValueError(f"tier {tier} has no observation at or before origin")
    hours = hours[causal]
    try:
        scores = np.asarray(array[causal, 1], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tier {tier} causal scores must be numeric") from exc
    if np.any(~np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError(f"tier {tier} causal scores must be finite and non-negative")
    if np.any(hours < 0.0) or np.any(hours > duration_hours):
        raise ValueError(f"tier {tier} causal hours must lie inside the event")
    order = np.argsort(hours, kind="stable")
    hours = hours[order]
    scores = scores[order]
    if np.any(np.diff(hours) <= 0.0):
        raise ValueError(f"tier {tier} has duplicate causal hours")
    if np.any(np.diff(scores) < 0.0):
        raise ValueError(f"tier {tier} cumulative scores must not decrease")
    if hours[0] == 0.0:
        if scores[0] != 0.0:
            raise ValueError(f"tier {tier} score at event start must be zero")
    else:
        hours = np.insert(hours, 0, 0.0)
        scores = np.insert(scores, 0, 0.0)
    if len(hours) < 2 or scores[-1] <= 0.0:
        raise ValueError(f"tier {tier} needs positive observed score mass")
    return NormalizedTierPrefix(
        tier=tier,
        hours=hours,
        scores=scores,
        last_observed_hour=float(hours[-1]),
        anchor_score=float(scores[-1]),
    )


def normalize_training_event(
    event: PaceTrainingEvent,
    config: PaceModelConfig = PaceModelConfig(),
) -> NormalizedPaceEvent:
    """Validate and causally normalize all asynchronous tiers in an event."""

    if not isinstance(event, PaceTrainingEvent):
        raise ValueError("event must be a PaceTrainingEvent")
    if not isinstance(config, PaceModelConfig):
        raise ValueError("config must be a PaceModelConfig")
    rewards = canonicalize_reward_tiers(event.reward_tiers)
    if any(tier < config.rank_min or tier > config.rank_max for tier in rewards):
        raise ValueError("reward_tiers must lie inside config rank bounds")
    normalized: dict[int, NormalizedTierPrefix] = {}
    for raw_tier, observations in event.tier_observations.items():
        tier = _positive_int(raw_tier, "tier_observations key")
        if tier < config.rank_min or tier > config.rank_max:
            raise ValueError(f"tier {tier} lies outside config rank bounds")
        if tier in normalized:
            raise ValueError("tier_observations contains duplicate canonical tiers")
        normalized[tier] = _normalize_observation_array(
            observations,
            tier=tier,
            duration_hours=float(event.duration_hours),
            origin_hour=float(event.duration_hours),
        )
    tiers = tuple(sorted(normalized))
    if not tiers:
        raise ValueError("event has no valid tier observations")
    return NormalizedPaceEvent(
        event_id=int(event.event_id),
        duration_hours=float(event.duration_hours),
        event_start_local_hour=float(event.event_start_local_hour),
        reward_tiers=rewards,
        tiers=tiers,
        tier_prefixes={tier: normalized[tier] for tier in tiers},
    )


def availability_curve(
    hours: Sequence[float] | np.ndarray,
    *,
    event_start_local_hour: float,
    config: PaceModelConfig = PaceModelConfig(),
) -> np.ndarray:
    """Existing positive circadian availability semantics on UTC+8 local time."""

    values = np.asarray(hours, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("hours must be finite")
    start_hour = _finite_float(event_start_local_hour, "event_start_local_hour")
    if not 0.0 <= start_hour < 24.0:
        raise ValueError("event_start_local_hour must be in [0, 24)")
    phase = (
        2.0
        * np.pi
        * (values + start_hour - config.circadian_peak_hour)
        / 24.0
    )
    raw = 1.0 + config.circadian_amplitude * np.cos(phase)
    return np.maximum(raw, config.availability_floor)


def urgency_curve(
    hours: Sequence[float] | np.ndarray,
    *,
    duration_hours: float,
    config: PaceModelConfig = PaceModelConfig(),
) -> np.ndarray:
    """Existing bounded deadline urgency semantics, equal to one at the end."""

    duration = _finite_float(duration_hours, "duration_hours")
    if duration <= 0.0:
        raise ValueError("duration_hours must be positive")
    values = np.asarray(hours, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("hours must be finite")
    remaining = np.clip((duration - values) / duration, 0.0, 1.0)
    eps = config.urgency_epsilon
    return (eps / (remaining + eps)) ** config.urgency_power


def reward_pressure(
    tier: int,
    reward_tiers: Iterable[int],
    config: PaceModelConfig = PaceModelConfig(),
) -> float:
    """Current rank/reward pressure scalar ``P_E(r)`` without reward classes."""

    rank = _positive_int(tier, "tier")
    if rank < config.rank_min or rank > config.rank_max:
        raise ValueError("tier lies outside config rank bounds")
    rewards = canonicalize_reward_tiers(reward_tiers)
    if any(value < config.rank_min or value > config.rank_max for value in rewards):
        raise ValueError("reward_tiers must lie inside config rank bounds")
    competitiveness = np.log(config.rank_max / rank) / np.log(
        config.rank_max / config.rank_min
    )
    competitiveness = float(np.clip(competitiveness, 0.0, 1.0))
    base = config.pressure_floor + (1.0 - config.pressure_floor) * competitiveness
    utility = (
        float(np.mean(rank <= np.asarray(rewards, dtype=int)))
        if rewards
        else 0.0
    )
    return float(1.0 - (1.0 - base) * (1.0 - utility))


class _PaceIntegrator:
    def __init__(
        self,
        *,
        duration_hours: float,
        event_start_local_hour: float,
        config: PaceModelConfig,
    ):
        self.duration_hours = _finite_float(duration_hours, "duration_hours")
        if self.duration_hours <= 0.0:
            raise ValueError("duration_hours must be positive")
        self.event_start_local_hour = _finite_float(
            event_start_local_hour,
            "event_start_local_hour",
        )
        if not 0.0 <= self.event_start_local_hour < 24.0:
            raise ValueError("event_start_local_hour must be in [0, 24)")
        self.config = config
        self._h_cache: dict[float, float] = {0.0: 0.0}
        self._g_cache: dict[float, float] = {0.0: 0.0}
        self._h_total = self.h(self.duration_hours)
        if not np.isfinite(self._h_total) or self._h_total <= 0.0:
            raise RuntimeError("availability integral over the event is not positive")

    def _availability_scalar(self, hour: float) -> float:
        return float(
            availability_curve(
                np.asarray([hour]),
                event_start_local_hour=self.event_start_local_hour,
                config=self.config,
            )[0]
        )

    def h(self, hour: float) -> float:
        value = _finite_float(hour, "hour")
        if not 0.0 <= value <= self.duration_hours:
            raise ValueError("hour must lie inside the event")
        if value in self._h_cache:
            return self._h_cache[value]
        # In the usual configuration the floor is inactive, so use the exact
        # cosine antiderivative.  The clipped case is integrated directly.
        if self.config.availability_floor <= 1.0 - self.config.circadian_amplitude:
            omega = 2.0 * np.pi / 24.0
            phase_zero = omega * (
                self.event_start_local_hour - self.config.circadian_peak_hour
            )
            result = value + self.config.circadian_amplitude / omega * (
                np.sin(phase_zero + omega * value) - np.sin(phase_zero)
            )
        else:
            result = quad(
                self._availability_scalar,
                0.0,
                value,
                epsabs=1e-10,
                epsrel=1e-11,
                limit=256,
            )[0]
        result = float(result)
        if not np.isfinite(result):
            raise RuntimeError("availability integral is non-finite")
        self._h_cache[value] = result
        return result

    def g(self, hour: float) -> float:
        value = _finite_float(hour, "hour")
        if not 0.0 <= value <= self.duration_hours:
            raise ValueError("hour must lie inside the event")
        if value in self._g_cache:
            return self._g_cache[value]

        def integrand(point: float) -> float:
            return self._availability_scalar(point) * float(
                urgency_curve(
                    np.asarray([point]),
                    duration_hours=self.duration_hours,
                    config=self.config,
                )[0]
            )

        result = float(
            quad(
                integrand,
                0.0,
                value,
                epsabs=1e-10,
                epsrel=1e-11,
                limit=256,
            )[0]
        )
        if not np.isfinite(result):
            raise RuntimeError("deadline pace integral is non-finite")
        self._g_cache[value] = result
        return result

    def cumulative_components(
        self,
        hours: Sequence[float] | np.ndarray,
        *,
        tier: int,
        reward_tiers: Iterable[int],
    ) -> np.ndarray:
        values = np.asarray(hours, dtype=float)
        if values.ndim != 1 or np.any(~np.isfinite(values)):
            raise ValueError("hours must be a one-dimensional finite array")
        if np.any(values < 0.0) or np.any(values > self.duration_hours):
            raise ValueError("hours must lie inside the event")
        h_values = np.asarray([self.h(float(value)) for value in values])
        g_values = np.asarray([self.g(float(value)) for value in values])
        launch = h_values - h_values**2 / (2.0 * self._h_total)
        pressure = reward_pressure(tier, reward_tiers, self.config)
        return np.column_stack((h_values, launch, pressure * g_values))

    def instantaneous_components(
        self,
        hours: Sequence[float] | np.ndarray,
        *,
        tier: int,
        reward_tiers: Iterable[int],
    ) -> np.ndarray:
        values = np.asarray(hours, dtype=float)
        if values.ndim != 1 or np.any(~np.isfinite(values)):
            raise ValueError("hours must be a one-dimensional finite array")
        if np.any(values < 0.0) or np.any(values > self.duration_hours):
            raise ValueError("hours must lie inside the event")
        availability = availability_curve(
            values,
            event_start_local_hour=self.event_start_local_hour,
            config=self.config,
        )
        h_values = np.asarray([self.h(float(value)) for value in values])
        depletion = 1.0 - h_values / self._h_total
        pressure = reward_pressure(tier, reward_tiers, self.config)
        urgency = urgency_curve(
            values,
            duration_hours=self.duration_hours,
            config=self.config,
        )
        return availability[:, None] * np.column_stack(
            (
                np.ones(len(values), dtype=float),
                depletion,
                pressure * urgency,
            )
        )


def cumulative_pace_components(
    hours: Sequence[float] | np.ndarray,
    *,
    tier: int,
    duration_hours: float,
    event_start_local_hour: float,
    reward_tiers: Iterable[int],
    config: PaceModelConfig = PaceModelConfig(),
) -> np.ndarray:
    """Public cumulative component matrix used for auditing and synthesis."""

    return _PaceIntegrator(
        duration_hours=duration_hours,
        event_start_local_hour=event_start_local_hour,
        config=config,
    ).cumulative_components(hours, tier=tier, reward_tiers=reward_tiers)


@dataclass(frozen=True)
class _TierFitData:
    tier: int
    design: np.ndarray
    rates: np.ndarray
    durations: np.ndarray
    energy: float


def _profile_tier(
    data: _TierFitData,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    shape = data.design @ weights
    denominator = float(np.sum(data.durations * shape**2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("profiled tier pace has zero design energy")
    numerator = float(np.sum(data.durations * shape * data.rates))
    amplitude = max(numerator / denominator, 0.0)
    residual = data.rates - amplitude * shape
    sse = float(np.sum(data.durations * residual**2))
    relative = sse / data.energy
    return amplitude, sse, relative


def fit_event_weights(
    event: PaceTrainingEvent,
    config: PaceModelConfig = PaceModelConfig(),
) -> PaceEventFit:
    """Fit one event's simplex weights with per-tier profiled amplitudes."""

    normalized = normalize_training_event(event, config)
    integrator = _PaceIntegrator(
        duration_hours=normalized.duration_hours,
        event_start_local_hour=normalized.event_start_local_hour,
        config=config,
    )
    tier_data: list[_TierFitData] = []
    interval_count = 0
    for tier in normalized.tiers:
        prefix = normalized.tier_prefixes[tier]
        starts = prefix.hours[:-1]
        ends = prefix.hours[1:]
        durations = ends - starts
        if np.any(durations <= 0.0):
            raise ValueError(f"tier {tier} has non-positive intervals")
        rates = np.diff(prefix.scores) / durations
        if np.any(~np.isfinite(rates)) or np.any(rates < 0.0):
            raise ValueError(f"tier {tier} has invalid interval rates")
        endpoints = np.unique(np.concatenate((starts, ends)))
        cumulative = integrator.cumulative_components(
            endpoints,
            tier=tier,
            reward_tiers=normalized.reward_tiers,
        )
        lookup = {float(hour): cumulative[index] for index, hour in enumerate(endpoints)}
        start_components = np.vstack([lookup[float(hour)] for hour in starts])
        end_components = np.vstack([lookup[float(hour)] for hour in ends])
        design = (end_components - start_components) / durations[:, None]
        energy = float(np.sum(durations * rates**2))
        if not np.isfinite(energy) or energy <= 0.0:
            raise ValueError(f"tier {tier} has no positive rate energy")
        tier_data.append(
            _TierFitData(
                tier=tier,
                design=design,
                rates=rates,
                durations=durations,
                energy=energy,
            )
        )
        interval_count += len(rates)
    if interval_count < 3:
        raise ValueError("event needs at least three observed intervals")

    def objective(free_values: np.ndarray) -> float:
        """Evaluate the simplex through two free coordinates.

        SLSQP probes points a few ulps outside active constraints while
        estimating derivatives.  Returning infinity for those probes makes
        the finite-difference gradient unusable, especially at a simplex
        vertex.  The inequality remains the authority for feasibility; the
        objective itself stays finite in its small numerical neighbourhood.
        """

        free = np.asarray(free_values, dtype=float)
        if free.shape != (2,) or np.any(~np.isfinite(free)):
            return float(np.finfo(float).max)
        weights = np.asarray([free[0], free[1], 1.0 - free.sum()], dtype=float)
        return float(
            np.mean([_profile_tier(data, weights)[2] for data in tier_data])
        )

    starts = (
        np.asarray([1.0 / 3.0, 1.0 / 3.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.0]),
        np.asarray([0.5, 0.5]),
        np.asarray([0.5, 0.0]),
        np.asarray([0.0, 0.5]),
    )
    candidates: list[tuple[float, np.ndarray]] = []
    for initial in starts:
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=((0.0, 1.0),) * 2,
            constraints=(
                {
                    "type": "ineq",
                    "fun": lambda values: float(1.0 - np.sum(values)),
                },
            ),
            options={
                "ftol": _OPTIMIZER_TOLERANCE,
                "maxiter": 1_000,
            },
        )
        if result.success and np.all(np.isfinite(result.x)):
            free = np.asarray(result.x, dtype=float)
            weights = np.asarray([free[0], free[1], 1.0 - free.sum()])
            # Canonicalise only the optimiser's final floating-point residue.
            weights[np.abs(weights) <= 1e-9] = 0.0
            weights = np.maximum(weights, 0.0)
            total = float(weights.sum())
            if total > 0.0:
                weights /= total
                candidates.append((objective(weights[:2]), weights))
    if not candidates:
        raise RuntimeError("simplex pace-weight optimisation failed")
    best_objective, best_vector = min(candidates, key=lambda item: item[0])
    if not np.isfinite(best_objective):
        raise RuntimeError("simplex pace-weight optimisation returned non-finite loss")
    weights = PaceWeights.from_sequence(best_vector)
    amplitudes: dict[int, float] = {}
    losses: dict[int, float] = {}
    raw_sse = 0.0
    for data in tier_data:
        amplitude, sse, relative = _profile_tier(data, weights.as_array())
        amplitudes[data.tier] = float(amplitude)
        losses[data.tier] = float(relative)
        raw_sse += sse
    return PaceEventFit(
        event_id=normalized.event_id,
        weights=weights,
        tier_amplitudes=amplitudes,
        tier_relative_losses=losses,
        objective=float(np.mean(list(losses.values()))),
        raw_duration_weighted_sse=float(raw_sse),
        interval_count=int(interval_count),
        tier_count=len(tier_data),
    )


def aggregate_event_weights(
    fits: Sequence[PaceEventFit | PaceWeights | Sequence[float] | np.ndarray],
) -> PaceWeights:
    """Return the arithmetic event-equal mean of explicit simplex vectors."""

    if isinstance(fits, (str, bytes)) or not isinstance(fits, Sequence) or not fits:
        raise ValueError("fits must be a non-empty sequence")
    vectors: list[np.ndarray] = []
    event_ids: list[int] = []
    for item in fits:
        if isinstance(item, PaceEventFit):
            vectors.append(item.weights.as_array())
            event_ids.append(item.event_id)
        elif isinstance(item, PaceWeights):
            vectors.append(item.as_array())
        else:
            vectors.append(PaceWeights.from_sequence(item).as_array())
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("fits must not repeat an event_id")
    mean = np.mean(np.vstack(vectors), axis=0)
    mean = np.maximum(mean, 0.0)
    total = float(mean.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("event-equal mean pace weight is invalid")
    return PaceWeights.from_sequence(mean / total)


def _prefix_pairs(prefix: Any) -> np.ndarray:
    if isinstance(prefix, pd.Series):
        return np.column_stack((prefix.index.to_numpy(), prefix.to_numpy()))
    if isinstance(prefix, pd.DataFrame):
        if not {"time", "score"}.issubset(prefix.columns):
            raise ValueError("prefix DataFrame must contain time and score columns")
        return prefix.loc[:, ["time", "score"]].to_numpy()
    try:
        array = np.asarray(prefix)
    except (TypeError, ValueError) as exc:
        raise ValueError("prefix must be an n x 2 array, Series, or DataFrame") from exc
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("prefix must be an n x 2 array, Series, or DataFrame")
    return array


def predict_tier_curve(
    prefix: Any,
    tier: int,
    start_at: int,
    end_at: int,
    forecast_times: Sequence[int] | np.ndarray,
    reward_tiers: Iterable[int],
    weights: PaceWeights | Sequence[float] | np.ndarray,
    config: PaceModelConfig = PaceModelConfig(),
) -> TierCurvePrediction:
    """Predict one tier from its last visible score before the forecast origin.

    ``prefix`` uses absolute millisecond timestamps.  ``forecast_times`` must
    include the prediction origin as its first entry; rows in ``prefix`` after
    that origin are discarded before their scores are parsed or validated.
    If the last visible observation is at ``a <= origin``, every requested
    score follows ``S(t) = S(a) C(t) / C(a)``.  A freshness gap therefore
    accumulates modelled growth instead of being silently treated as zero.
    """

    rank = _positive_int(tier, "tier")
    start = _integer_milliseconds(start_at, "start_at")
    end = _integer_milliseconds(end_at, "end_at")
    if end <= start:
        raise ValueError("end_at must be greater than start_at")
    raw_forecast = np.asarray(forecast_times)
    if raw_forecast.ndim != 1 or raw_forecast.size == 0:
        raise ValueError("forecast_times must be a non-empty one-dimensional array")
    if not np.issubdtype(raw_forecast.dtype, np.integer):
        raise ValueError("forecast_times must contain integer milliseconds")
    forecast = raw_forecast.astype(np.int64)
    if np.any(np.diff(forecast) < 0):
        raise ValueError("forecast_times must be non-decreasing")
    origin = int(forecast[0])
    if origin <= start or forecast[-1] > end:
        raise ValueError("forecast_times must stay after start_at and through end_at")
    canonical_weights = (
        weights if isinstance(weights, PaceWeights) else PaceWeights.from_sequence(weights)
    )
    rewards = canonicalize_reward_tiers(reward_tiers)
    if rank < config.rank_min or rank > config.rank_max:
        raise ValueError("tier lies outside config rank bounds")
    if any(value < config.rank_min or value > config.rank_max for value in rewards):
        raise ValueError("reward_tiers must lie inside config rank bounds")

    pairs = _prefix_pairs(prefix)
    if len(pairs) == 0:
        raise ValueError("prefix must not be empty")
    try:
        raw_times = np.asarray(pairs[:, 0], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prefix timestamps must be integer milliseconds") from exc
    if (
        np.any(~np.isfinite(raw_times))
        or np.any(raw_times != np.floor(raw_times))
    ):
        raise ValueError("prefix timestamps must be integer milliseconds")
    timestamps = raw_times.astype(np.int64)
    causal = timestamps <= origin
    if not np.any(causal):
        raise ValueError("prefix has no observation at or before origin")
    causal_times = timestamps[causal]
    try:
        causal_scores = np.asarray(pairs[causal, 1], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("causal prefix scores must be numeric") from exc
    if np.any(~np.isfinite(causal_scores)) or np.any(causal_scores < 0.0):
        raise ValueError("causal prefix scores must be finite and non-negative")
    if np.any(causal_times < start) or np.any(causal_times > end):
        raise ValueError("causal prefix timestamps must lie inside the event")
    order = np.argsort(causal_times, kind="stable")
    causal_times = causal_times[order]
    causal_scores = causal_scores[order]
    if np.any(np.diff(causal_times) <= 0):
        raise ValueError("causal prefix has duplicate timestamps")
    if np.any(np.diff(causal_scores) < 0.0):
        raise ValueError("causal prefix cumulative scores must not decrease")
    anchor_score = float(causal_scores[-1])
    if anchor_score <= 0.0:
        raise ValueError("last visible score must be positive")
    anchor_at = int(causal_times[-1])

    duration = (end - start) / MS_PER_HOUR
    origin_hour = (origin - start) / MS_PER_HOUR
    anchor_hour = (anchor_at - start) / MS_PER_HOUR
    hours = (forecast.astype(float) - start) / MS_PER_HOUR
    local_start = (start / MS_PER_HOUR + config.utc_offset_hours) % 24.0
    integrator = _PaceIntegrator(
        duration_hours=duration,
        event_start_local_hour=local_start,
        config=config,
    )
    cumulative = integrator.cumulative_components(
        hours,
        tier=rank,
        reward_tiers=rewards,
    )
    anchor_components = integrator.cumulative_components(
        np.asarray([anchor_hour]),
        tier=rank,
        reward_tiers=rewards,
    )[0]
    weight_vector = canonical_weights.as_array()
    cumulative_anchor = float(anchor_components @ weight_vector)
    if not np.isfinite(cumulative_anchor) or cumulative_anchor <= 0.0:
        raise RuntimeError("anchor cumulative pace is not positive")
    cumulative_shape = cumulative @ weight_vector
    amplitude = anchor_score / cumulative_anchor
    scores = amplitude * cumulative_shape
    end_components = integrator.cumulative_components(
        np.asarray([duration]),
        tier=rank,
        reward_tiers=rewards,
    )[0]
    cumulative_end = float(end_components @ weight_vector)
    instantaneous = integrator.instantaneous_components(
        hours,
        tier=rank,
        reward_tiers=rewards,
    )
    speeds = amplitude * (instantaneous @ weight_vector)
    if np.any(~np.isfinite(scores)) or np.any(np.diff(scores) < -1e-8):
        raise RuntimeError("predicted cumulative score curve is invalid")
    if np.any(~np.isfinite(speeds)) or np.any(speeds < 0.0):
        raise RuntimeError("predicted speed curve is invalid")
    return TierCurvePrediction(
        tier=rank,
        origin_at=origin,
        anchor_at=anchor_at,
        anchor_score=anchor_score,
        forecast_times=forecast.copy(),
        scores=scores,
        speeds_per_hour=speeds,
        weights=canonical_weights,
        diagnostics={
            "duration_hours": float(duration),
            "origin_hour": float(origin_hour),
            "event_start_local_hour": float(local_start),
            "anchor_hour": float(anchor_hour),
            "cumulative_pace_at_anchor": cumulative_anchor,
            "cumulative_pace_at_end": cumulative_end,
            "profiled_amplitude": float(amplitude),
            "reward_pressure": reward_pressure(rank, rewards, config),
            "growth_factor_to_end": float(cumulative_end / cumulative_anchor),
            "predicted_end_score": float(amplitude * cumulative_end),
            "weight_source": "event_equal_prior",
            "fallback_used": False,
        },
    )


__all__ = [
    "MS_PER_HOUR",
    "PACE_COMPONENT_NAMES",
    "NormalizedPaceEvent",
    "NormalizedTierPrefix",
    "PaceEventFit",
    "PaceModelConfig",
    "PaceTrainingEvent",
    "PaceWeights",
    "TierCurvePrediction",
    "aggregate_event_weights",
    "availability_curve",
    "canonicalize_reward_tiers",
    "cumulative_pace_components",
    "fit_event_weights",
    "normalize_training_event",
    "predict_tier_curve",
    "reward_pressure",
    "urgency_curve",
]
