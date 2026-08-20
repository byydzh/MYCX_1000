import os
import sys

import numpy as np
import pytest


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behavior_pace_model import (
    MS_PER_HOUR,
    PaceModelConfig,
    PaceTrainingEvent,
    PaceWeights,
    aggregate_event_weights,
    availability_curve,
    cumulative_pace_components,
    fit_event_weights,
    normalize_training_event,
    predict_tier_curve,
)


DURATION = 120.0
EVENT_START_LOCAL_HOUR = 13.0
REWARD_TIERS = (500, 1_500, 2_000)
TIERS_WITHOUT_T1000 = (500, 1_500, 2_000)
TRUE_WEIGHTS = PaceWeights(sustain=0.34, launch=0.41, deadline=0.25)
CONFIG = PaceModelConfig(
    circadian_amplitude=0.25,
    circadian_peak_hour=21.0,
    availability_floor=0.05,
)


def _grid(step: float, *, offset: float = 0.0) -> np.ndarray:
    middle = np.arange(max(step + offset, step), DURATION, step)
    return np.unique(np.concatenate(([0.0], middle, [DURATION])))


def _scores(
    hours: np.ndarray,
    tier: int,
    weights: PaceWeights,
    amplitude: float,
) -> np.ndarray:
    components = cumulative_pace_components(
        hours,
        tier=tier,
        duration_hours=DURATION,
        event_start_local_hour=EVENT_START_LOCAL_HOUR,
        reward_tiers=REWARD_TIERS,
        config=CONFIG,
    )
    return amplitude * (components @ weights.as_array())


def _event(
    *,
    event_id: int = 250,
    weights: PaceWeights = TRUE_WEIGHTS,
    grids: dict[int, np.ndarray] | None = None,
) -> PaceTrainingEvent:
    selected_grids = grids or {
        500: _grid(4.0),
        1_500: _grid(5.0, offset=0.7),
        2_000: _grid(7.0, offset=1.3),
    }
    amplitudes = {500: 80_000.0, 1_500: 47_000.0, 2_000: 31_000.0}
    observations = {
        tier: np.column_stack(
            (
                hours,
                _scores(hours, tier, weights, amplitudes[tier]),
            )
        )
        for tier, hours in selected_grids.items()
    }
    return PaceTrainingEvent(
        event_id=event_id,
        duration_hours=DURATION,
        reward_tiers=REWARD_TIERS,
        tier_observations=observations,
        event_start_local_hour=EVENT_START_LOCAL_HOUR,
    )


def _absolute_prefix(
    *,
    tier: int,
    weights: PaceWeights,
    amplitude: float,
    hours: np.ndarray,
    start_at: int,
) -> np.ndarray:
    local_start = (start_at / MS_PER_HOUR + CONFIG.utc_offset_hours) % 24.0
    components = cumulative_pace_components(
        hours,
        tier=tier,
        duration_hours=DURATION,
        event_start_local_hour=local_start,
        reward_tiers=REWARD_TIERS,
        config=CONFIG,
    )
    scores = amplitude * (components @ weights.as_array())
    timestamps = start_at + np.rint(hours * MS_PER_HOUR).astype(np.int64)
    return np.column_stack((timestamps, scores))


def test_no_t1000_dependency_and_event_equal_aggregation() -> None:
    fit = fit_event_weights(_event(), CONFIG)

    assert set(fit.tier_amplitudes) == set(TIERS_WITHOUT_T1000)
    np.testing.assert_allclose(
        fit.weights.as_array(),
        TRUE_WEIGHTS.as_array(),
        atol=2e-5,
        rtol=0.0,
    )

    mean = aggregate_event_weights(
        [
            PaceWeights(0.6, 0.3, 0.1),
            PaceWeights(0.2, 0.1, 0.7),
        ]
    )
    np.testing.assert_allclose(mean.as_array(), [0.4, 0.2, 0.4])


def test_asynchronous_tier_timestamps_are_fitted_independently() -> None:
    event = _event(
        grids={
            500: np.asarray([0.0, 3.0, 11.0, 28.0, 57.0, 91.0, 120.0]),
            1_500: np.asarray([0.0, 5.5, 19.0, 44.0, 78.0, 120.0]),
            2_000: np.asarray([0.0, 8.0, 23.0, 39.0, 66.0, 99.0, 120.0]),
        }
    )
    normalized = normalize_training_event(event, CONFIG)
    fit = fit_event_weights(event, CONFIG)

    assert not np.array_equal(
        normalized.tier_prefixes[500].hours,
        normalized.tier_prefixes[1_500].hours,
    )
    assert fit.interval_count == (6 + 5 + 6)
    np.testing.assert_allclose(
        fit.weights.as_array(),
        TRUE_WEIGHTS.as_array(),
        atol=3e-5,
        rtol=0.0,
    )


def test_hidden_suffix_mutation_is_bitwise_invariant() -> None:
    start_at = 1_800_000_000_000
    end_at = start_at + int(DURATION * MS_PER_HOUR)
    hours = np.asarray([0.0, 12.0, 24.0, 36.0, 48.0, 72.0, 120.0])
    clean = _absolute_prefix(
        tier=1_500,
        weights=TRUE_WEIGHTS,
        amplitude=42_000.0,
        hours=hours,
        start_at=start_at,
    )
    poisoned = clean.astype(object)
    poisoned[hours > 48.0, 1] = "not-a-causal-score"
    forecast = start_at + np.asarray([48, 72, 96, 120], dtype=np.int64) * MS_PER_HOUR

    left = predict_tier_curve(
        clean,
        1_500,
        start_at,
        end_at,
        forecast,
        REWARD_TIERS,
        TRUE_WEIGHTS,
        CONFIG,
    )
    right = predict_tier_curve(
        poisoned,
        1_500,
        start_at,
        end_at,
        forecast,
        REWARD_TIERS,
        TRUE_WEIGHTS,
        CONFIG,
    )

    np.testing.assert_array_equal(left.scores, right.scores)
    np.testing.assert_array_equal(left.speeds_per_hour, right.speeds_per_hour)
    assert left.diagnostics == right.diagnostics


def test_last_visible_observation_is_the_continuous_anchor() -> None:
    start_at = 1_800_000_000_000
    end_at = start_at + int(DURATION * MS_PER_HOUR)
    prefix_hours = np.asarray([0.0, 12.0, 31.0, 47.0])
    prefix = _absolute_prefix(
        tier=500,
        weights=TRUE_WEIGHTS,
        amplitude=75_000.0,
        hours=prefix_hours,
        start_at=start_at,
    )
    forecast = start_at + np.asarray([48, 72, 120], dtype=np.int64) * MS_PER_HOUR

    prediction = predict_tier_curve(
        prefix,
        500,
        start_at,
        end_at,
        forecast,
        REWARD_TIERS,
        TRUE_WEIGHTS,
        CONFIG,
    )

    assert prediction.anchor_at == start_at + 47 * MS_PER_HOUR
    assert prediction.anchor_score == prefix[-1, 1]
    assert prediction.scores[0] > prediction.anchor_score
    assert prediction.diagnostics["fallback_used"] is False

    at_anchor = predict_tier_curve(
        prefix,
        500,
        start_at,
        end_at,
        start_at + np.asarray([47, 72, 120], dtype=np.int64) * MS_PER_HOUR,
        REWARD_TIERS,
        TRUE_WEIGHTS,
        CONFIG,
    )
    assert at_anchor.scores[0] == pytest.approx(prefix[-1, 1])


def test_launch_only_deseasonalized_speed_strictly_decreases() -> None:
    start_at = 1_800_000_000_000
    end_at = start_at + int(DURATION * MS_PER_HOUR)
    weights = PaceWeights(0.0, 1.0, 0.0)
    prefix_hours = np.asarray([0.0, 12.0, 24.0])
    prefix = _absolute_prefix(
        tier=500,
        weights=weights,
        amplitude=50_000.0,
        hours=prefix_hours,
        start_at=start_at,
    )
    forecast_hours = np.asarray([24.0, 48.0, 72.0, 96.0, 120.0])
    forecast = start_at + (forecast_hours * MS_PER_HOUR).astype(np.int64)
    prediction = predict_tier_curve(
        prefix,
        500,
        start_at,
        end_at,
        forecast,
        REWARD_TIERS,
        weights,
        CONFIG,
    )
    local_start = prediction.diagnostics["event_start_local_hour"]
    availability = availability_curve(
        forecast_hours,
        event_start_local_hour=local_start,
        config=CONFIG,
    )
    deseasonalized = prediction.speeds_per_hour / availability

    assert np.all(np.diff(deseasonalized) < 0.0)
    assert deseasonalized[-1] == pytest.approx(0.0, abs=1e-10)


def test_deadline_only_deseasonalized_speed_strictly_increases() -> None:
    start_at = 1_800_000_000_000
    end_at = start_at + int(DURATION * MS_PER_HOUR)
    weights = PaceWeights(0.0, 0.0, 1.0)
    prefix_hours = np.asarray([0.0, 12.0, 24.0])
    prefix = _absolute_prefix(
        tier=2_000,
        weights=weights,
        amplitude=50_000.0,
        hours=prefix_hours,
        start_at=start_at,
    )
    forecast_hours = np.asarray([24.0, 48.0, 72.0, 96.0, 120.0])
    forecast = start_at + (forecast_hours * MS_PER_HOUR).astype(np.int64)
    prediction = predict_tier_curve(
        prefix,
        2_000,
        start_at,
        end_at,
        forecast,
        REWARD_TIERS,
        weights,
        CONFIG,
    )
    local_start = prediction.diagnostics["event_start_local_hour"]
    availability = availability_curve(
        forecast_hours,
        event_start_local_hour=local_start,
        config=CONFIG,
    )
    deseasonalized = prediction.speeds_per_hour / availability

    assert np.all(np.diff(deseasonalized) > 0.0)


def test_mechanical_interval_split_does_not_change_fitted_weights() -> None:
    coarse = _event(
        event_id=251,
        grids={tier: _grid(12.0) for tier in TIERS_WITHOUT_T1000},
    )
    fine = _event(
        event_id=252,
        grids={tier: _grid(6.0) for tier in TIERS_WITHOUT_T1000},
    )

    coarse_fit = fit_event_weights(coarse, CONFIG)
    fine_fit = fit_event_weights(fine, CONFIG)

    np.testing.assert_allclose(
        coarse_fit.weights.as_array(),
        fine_fit.weights.as_array(),
        atol=2e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        coarse_fit.weights.as_array(),
        TRUE_WEIGHTS.as_array(),
        atol=2e-5,
        rtol=0.0,
    )


def test_synthetic_simplex_weights_are_recovered() -> None:
    fit = fit_event_weights(_event(event_id=253), CONFIG)

    assert fit.objective < 1e-12
    assert fit.raw_duration_weighted_sse >= 0.0
    np.testing.assert_allclose(
        fit.weights.as_array(),
        TRUE_WEIGHTS.as_array(),
        atol=2e-5,
        rtol=0.0,
    )
    assert all(value > 0.0 for value in fit.tier_amplitudes.values())


def test_predicted_cumulative_score_is_non_decreasing_without_rank_projection() -> None:
    start_at = 1_800_000_000_000
    end_at = start_at + int(DURATION * MS_PER_HOUR)
    prefix_hours = np.asarray([0.0, 8.0, 19.0, 36.0])
    prefix = _absolute_prefix(
        tier=2_000,
        weights=TRUE_WEIGHTS,
        amplitude=29_000.0,
        hours=prefix_hours,
        start_at=start_at,
    )
    forecast_hours = np.arange(36.0, DURATION + 0.1, 3.0)
    forecast = start_at + (forecast_hours * MS_PER_HOUR).astype(np.int64)

    prediction = predict_tier_curve(
        prefix,
        2_000,
        start_at,
        end_at,
        forecast,
        REWARD_TIERS,
        TRUE_WEIGHTS,
        CONFIG,
    )

    assert np.all(np.diff(prediction.scores) >= 0.0)
    assert np.all(prediction.speeds_per_hour >= 0.0)


def test_bad_causal_input_is_rejected_instead_of_repaired() -> None:
    bad = PaceTrainingEvent(
        event_id=254,
        duration_hours=24.0,
        reward_tiers=(),
        tier_observations={
            500: np.asarray(
                [
                    [0.0, 0.0],
                    [4.0, 100.0],
                    [8.0, 90.0],
                    [24.0, 200.0],
                ]
            )
        },
    )

    with pytest.raises(ValueError, match="must not decrease"):
        fit_event_weights(bad, PaceModelConfig(circadian_amplitude=0.0))
