from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from domain_models import EventData, EventMeta
from scripts import evaluate_reward_tiers_318_319 as evaluator


MS_PER_HOUR = evaluator.MS_PER_HOUR


def _event(
    event_id: int,
    *,
    reward_tiers=evaluator.DEFAULT_REWARD_TIERS,
    end_hours: int = 49,
) -> evaluator.EvaluationEvent:
    rows = []
    for tier in reward_tiers:
        for hour, score in ((1, 10.0), (12, 30.0), (30, 60.0), (50, 9999.0)):
            rows.append(
                {"time": hour * MS_PER_HOUR, "score": score + tier, "tier": tier}
            )
    return evaluator.EvaluationEvent(
        event_id=event_id,
        start_at=0,
        end_at=end_hours * MS_PER_HOUR,
        frame=pd.DataFrame(rows),
        reward_tiers=reward_tiers,
        availability_status="hhwx_timestamp_masked_debug_replay",
    )


def _loaded(
    event_id: int,
    raw_tier_records,
    *,
    reward_tiers=evaluator.DEFAULT_REWARD_TIERS,
    end_hours: int = 49,
) -> evaluator.LoadedEvaluationEvent:
    return evaluator.LoadedEvaluationEvent(
        event=_event(
            event_id,
            reward_tiers=reward_tiers,
            end_hours=end_hours,
        ),
        raw_tier_records=raw_tier_records,
        cache_path=Path(f"{event_id}.json"),
        cache_sha256="a" * 64,
        reward_provenance={"source": "hhwx", "server": 3},
        collection_metadata={
            "source": "hhwx",
            "server": 3,
            "fetched_at": "2026-08-20T00:00:00+00:00",
            "availability_status": "explicit_row_level_first_seen_at",
        },
    )


def _excluded(event_id: int) -> evaluator.ExcludedEvaluationEvent:
    return evaluator.ExcludedEvaluationEvent(
        event_id=event_id,
        status="not_applicable",
        reason="empty_canonical_reward_tiers",
        reward_tiers=(),
        event_start_at=0,
        event_end_at=10 * MS_PER_HOUR,
        cache_path=Path(f"{event_id}.json"),
        cache_sha256="b" * 64,
        reward_provenance={"source": "hhwx", "server": 3},
        collection_metadata={"source": "hhwx", "server": 3},
    )


def test_cache_loader_does_not_read_or_expose_final_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"truth_unlocked": False}

    class GuardedFinalRecord(dict):
        def __getitem__(self, key):
            if key == "ep" and not state["truth_unlocked"]:
                raise AssertionError("held-out final score was read during cache load")
            return super().__getitem__(key)

    tier_records = {}
    for tier in evaluator.DEFAULT_REWARD_TIERS:
        tier_records[str(tier)] = [
            {"time": MS_PER_HOUR, "ep": 10.0 + tier},
            {"time": 2 * MS_PER_HOUR, "ep": 20.0 + tier},
            GuardedFinalRecord(
                {
                    "time": 11 * MS_PER_HOUR,
                    "ep": 1000.0 + tier,
                    "isFinal": True,
                }
            ),
        ]
    tier_records["10"] = [
        {"time": MS_PER_HOUR, "ep": 100.0},
        {"time": 2 * MS_PER_HOUR, "ep": 200.0},
    ]
    payload = {
        "event_id": 318,
        "meta": {
            "start_at": 0,
            "end_at": 10 * MS_PER_HOUR,
            "event_type": "mission_live",
        },
        "reward_tiers": list(evaluator.DEFAULT_REWARD_TIERS),
        "reward_tier_provenance": {"source": "hhwx", "server": 3},
        "collection_metadata": {
            "source": "hhwx",
            "server": 3,
            "requested_tiers": list(evaluator.DEFAULT_REWARD_TIERS),
            "failed_tiers": [],
        },
        "tier_records": tier_records,
    }
    monkeypatch.setattr(Path, "is_file", lambda _self: True)
    monkeypatch.setattr(Path, "read_bytes", lambda _self: b"{}")
    monkeypatch.setattr(evaluator.json, "loads", lambda _text: payload)

    loaded = evaluator.load_hhwx_event(
        Path("unused"),
        318,
        completed_as_of_ms=20 * MS_PER_HOUR,
    )

    assert isinstance(loaded, evaluator.LoadedEvaluationEvent)
    assert loaded.event.frame["time"].max() == 2 * MS_PER_HOUR
    assert 10 in loaded.event.tiers
    state["truth_unlocked"] = True
    assert evaluator._actual_final_score(loaded, 500) == 1500.0


def test_cache_loader_records_explicit_empty_reward_tiers_as_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "event_id": 250,
        "meta": {
            "start_at": 0,
            "end_at": 10 * MS_PER_HOUR,
            "event_type": "mission_live",
        },
        "reward_tiers": [],
        "reward_tier_provenance": {"source": "hhwx", "server": 3},
        "collection_metadata": {
            "source": "hhwx",
            "server": 3,
            "requested_tiers": [],
            "failed_tiers": [],
        },
        # An empty canonical target is authoritative; no fixed tier is guessed.
        "tier_records": {},
    }
    monkeypatch.setattr(Path, "is_file", lambda _self: True)
    monkeypatch.setattr(Path, "read_bytes", lambda _self: b"{}")
    monkeypatch.setattr(evaluator.json, "loads", lambda _text: payload)

    result = evaluator.load_hhwx_event(
        Path("unused"),
        250,
        completed_as_of_ms=20 * MS_PER_HOUR,
    )

    assert isinstance(result, evaluator.ExcludedEvaluationEvent)
    assert result.status == "not_applicable"
    assert result.reason == "empty_canonical_reward_tiers"
    assert result.reward_tiers == ()


def test_frozen_cache_parser_only_normalizes_legacy_isfinal_nan() -> None:
    parsed = evaluator._parse_frozen_cache_json(
        b'{"tier_records":{"500":[{"time":1,"ep":2,"isFinal":NaN}]}}',
        Path("legacy.json"),
    )
    assert parsed["tier_records"]["500"][0]["isFinal"] is None
    with pytest.raises(ValueError, match="forbidden"):
        evaluator._parse_frozen_cache_json(
            b'{"tier_records":{"500":[{"time":1,"ep":NaN}]}}',
            Path("invalid.json"),
        )


def test_truth_is_first_post_end_bucket_not_latest_or_is_final() -> None:
    loaded = _loaded(
        318,
        {
            500: (
                {"time": 49 * MS_PER_HOUR + 1, "ep": 1500.0},
                {
                    "time": 50 * MS_PER_HOUR,
                    "ep": 9999.0,
                    "isFinal": True,
                },
            )
        },
        reward_tiers=(500,),
        end_hours=49,
    )

    score, timestamp = evaluator._actual_final_bucket(loaded, 500)

    assert score == 1500.0
    assert timestamp == 49 * MS_PER_HOUR + 1


def test_event_scope_supports_explicit_ids_or_an_inclusive_interval() -> None:
    assert evaluator.resolve_event_ids(None, None, None) == (
        evaluator.DEFAULT_EVENT_IDS
    )
    assert evaluator.resolve_event_ids([319, 317, 318], None, None) == (
        317,
        318,
        319,
    )
    assert evaluator.resolve_event_ids(None, 317, 319) == (317, 318, 319)
    parsed = evaluator.parse_args(["--event-id-range", "317", "319"])
    assert parsed.event_id_range == [317, 319]
    with pytest.raises(ValueError, match="cannot be combined"):
        evaluator.resolve_event_ids([318], 317, 319)
    with pytest.raises(ValueError, match="duplicate"):
        evaluator.resolve_event_ids([318, 318], None, None)
    with pytest.raises(ValueError, match="must be <="):
        evaluator.resolve_event_ids(None, 319, 317)


def test_default_318_319_plan_has_exactly_51_common_scoring_rows() -> None:
    events = (
        _event(318, end_hours=226),
        _event(319, end_hours=202),
    )

    planned_rows = sum(
        len(evaluator.origin_hours(event)) * len(event.reward_tiers)
        for event in events
    )

    assert evaluator.origin_hours(events[0]) == tuple(range(24, 217, 24))
    assert evaluator.origin_hours(events[1]) == tuple(range(24, 193, 24))
    assert planned_rows == 51


def test_evaluate_event_masks_every_predictor_and_opens_truth_after_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reward_tiers = (500, 2000)
    event = _event(318, reward_tiers=reward_tiers)
    state = {
        "truth_reads": 0,
        "baseline_prefix": None,
        "skeleton_variants": [],
        "pace_calls": 0,
    }
    expected_prediction_calls = len(evaluator.origin_hours(event)) * len(reward_tiers)

    class GuardedTruthRows(dict):
        def __getitem__(self, tier):
            assert state["pace_calls"] == expected_prediction_calls
            assert len(state["skeleton_variants"]) == 4 * expected_prediction_calls
            state["truth_reads"] += 1
            return super().__getitem__(tier)

    truth = GuardedTruthRows(
        {
            tier: ({"time": 50 * MS_PER_HOUR, "ep": 1000.0 + tier, "isFinal": True},)
            for tier in reward_tiers
        }
    )
    loaded = _loaded(318, truth, reward_tiers=reward_tiers)
    real_baselines = evaluator._baseline_predictions

    def guarded_baselines(event, prefix, tier):
        state["baseline_prefix"] = prefix
        return real_baselines(event, prefix, tier)

    monkeypatch.setattr(evaluator, "_baseline_predictions", guarded_baselines)

    def guarded_pace(event, prefix, *, origin_at, tier, pace_prior):
        assert prefix is state["baseline_prefix"]
        assert bool((prefix["time"] <= origin_at).all())
        state["pace_calls"] += 1
        return 2500.0 + tier, {
            "fallback_used": False,
            "tier_interpolation_used": False,
        }

    monkeypatch.setattr(evaluator, "_predict_behavior_pace", guarded_pace)

    class GuardedSkeletonReplay:
        def predict(self, event, prefix, *, origin_at, tier, variant):
            assert prefix is state["baseline_prefix"]
            assert bool((prefix["time"] <= origin_at).all())
            assert variant in {
                "same_rank",
                "reward_behavior",
                "same_rank_paired_intersection",
                "reward_behavior_paired_intersection",
            }
            state["skeleton_variants"].append(variant)
            return 3000.0 + tier, {
                "variant": variant,
                "fallback_used": False,
                "tier_interpolation_used": False,
                "history_event_ids": (
                    [10] if "paired_intersection" in variant else [10, 9]
                ),
            }

    rows = evaluator.evaluate_event(
        loaded,
        pace_prior=object(),
        skeleton_replay=GuardedSkeletonReplay(),
    )

    assert len(rows) == 2 * len(reward_tiers)
    assert state["pace_calls"] == len(rows)
    assert state["truth_reads"] == len(reward_tiers)
    assert state["baseline_prefix"] is not None
    assert all(row["prefix_max_time"] <= row["origin_at"] for row in rows)
    assert all(row["actual_truth_at"] == 50 * MS_PER_HOUR for row in rows)
    assert all(row["actual_truth_rule"] == "first_post_end_bucket" for row in rows)
    assert all(set(row["predictions"]) == set(evaluator.METHODS) for row in rows)
    assert all(set(row["metrics"]) == set(evaluator.METHODS) for row in rows)
    assert {
        variant: state["skeleton_variants"].count(variant)
        for variant in set(state["skeleton_variants"])
    } == {
        "same_rank": len(rows),
        "reward_behavior": len(rows),
        "same_rank_paired_intersection": len(rows),
        "reward_behavior_paired_intersection": len(rows),
    }
    assert {row["tier"] for row in rows} == set(reward_tiers)
    assert 1000 not in {row["tier"] for row in rows}


def test_evaluate_event_records_common_input_and_skeleton_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = evaluator.EvaluationEvent(
        event_id=291,
        start_at=0,
        end_at=49 * MS_PER_HOUR,
        frame=pd.DataFrame(
            [
                {"time": 30 * MS_PER_HOUR, "score": 100.0, "tier": 1000},
                {"time": 36 * MS_PER_HOUR, "score": 200.0, "tier": 1000},
            ]
        ),
        reward_tiers=(1000,),
        event_type="mission_live",
    )
    loaded = evaluator.LoadedEvaluationEvent(
        event=event,
        raw_tier_records={
            1000: ({"time": 50 * MS_PER_HOUR, "ep": 300.0},)
        },
        cache_path=Path("291.json"),
        cache_sha256="a" * 64,
        reward_provenance={"source": "hhwx", "server": 3},
        collection_metadata={"source": "hhwx", "server": 3},
    )
    calls = {"pace": 0, "skeleton": 0}

    def fake_pace(event, prefix, *, origin_at, tier, pace_prior):
        calls["pace"] += 1
        return 250.0, {"fallback_used": False}

    monkeypatch.setattr(evaluator, "_predict_behavior_pace", fake_pace)

    class MissingSkeleton:
        def predict(self, event, prefix, *, origin_at, tier, variant):
            calls["skeleton"] += 1
            raise evaluator.SkeletonUnavailableError("no_legal_exact_tier_history")

    rows = evaluator.evaluate_event(
        loaded,
        pace_prior=object(),
        skeleton_replay=MissingSkeleton(),
    )

    assert len(rows) == 2
    unavailable, evaluable = rows
    assert unavailable["origin_hours"] == 24
    assert unavailable["scoring_status"] == "input_unavailable"
    assert all(value is None for value in unavailable["predictions"].values())
    assert all(value is None for value in unavailable["metrics"].values())
    assert all(
        status["attempted"] is False
        for status in unavailable["method_status"].values()
    )
    assert evaluable["origin_hours"] == 48
    assert evaluable["scoring_status"] == "evaluable"
    assert all(
        evaluable["predictions"][method] is not None
        for method in evaluator.FULL_COVERAGE_METHODS
    )
    assert all(
        evaluable["predictions"][method] is None
        for method in evaluator.SKELETON_METHODS
    )
    assert calls == {"pace": 1, "skeleton": 4}


def test_skeleton_replay_uses_masked_t10_and_absolute_origin_debug_hours() -> None:
    rows = []
    for tier, scores in (
        (10, ((1, 100.0), (2, 200.0), (25, 10_000.0))),
        (500, ((1, 50.0), (2, 80.0), (25, 9_000.0))),
    ):
        rows.extend(
            {"time": hour * MS_PER_HOUR, "ep": score, "tier": tier}
            for hour, score in scores
        )
    event = evaluator.EvaluationEvent(
        event_id=318,
        start_at=0,
        end_at=30 * MS_PER_HOUR,
        frame=pd.DataFrame(rows).rename(columns={"ep": "score"}),
        reward_tiers=evaluator.DEFAULT_REWARD_TIERS,
        availability_status="hhwx_timestamp_masked_debug_replay",
    )
    history_data = EventData(
        meta=EventMeta(1, "mission_live", 0, 10 * MS_PER_HOUR, 10 * MS_PER_HOUR),
        df=pd.DataFrame({"time": [0, MS_PER_HOUR], "value": [0.0, 1.0]}),
        scale=1.0,
        tier=500,
    )
    prepared = evaluator.PreparedSkeletonHistory(
        event_data=history_data,
        mode="exact",
        source_tiers=(500,),
        provenance={},
    )
    replay = object.__new__(evaluator.SkeletonKFReplay)
    replay.cache_headers = {
        318: SimpleNamespace(
            tracker_tiers=(10,),
            path=Path("318.json"),
            sha256="a" * 64,
        )
    }
    replay._scale_cache = {}
    replay.selections = {
        (318, 500): evaluator.SkeletonHistorySelection(
            same_rank=(prepared,),
            reward_behavior=(prepared,),
            paired_same_rank=(prepared,),
            paired_reward_behavior=(prepared,),
            audit={},
        )
    }
    captured = {}

    class FakeEngine:
        def predict(self, target, history, *, debug_hours):
            captured["target"] = target
            captured["debug_hours"] = debug_hours
            return SimpleNamespace(
                final_score=1234.0,
                ratio=1.0,
                scale_factor=1.0,
                used_params=[1, 2, 3, 4, 5],
            )

    replay.engine = FakeEngine()
    origin_at = 24 * MS_PER_HOUR
    prefix = evaluator._visible_prefix(event, origin_at)
    prediction, diagnostics = replay.predict(
        event,
        prefix,
        origin_at=origin_at,
        tier=500,
        variant="same_rank",
    )

    assert prediction == 1234.0
    assert captured["target"].df["time"].max() == 2 * MS_PER_HOUR
    assert captured["target"].scale == pytest.approx(100.0 / 60.0)
    assert captured["debug_hours"] == pytest.approx(23.0)
    assert diagnostics["absolute_origin_at"] == origin_at
    assert diagnostics["target_scale_provenance"]["prefix_max_time"] == (
        2 * MS_PER_HOUR
    )
    assert diagnostics["target_scale_provenance"]["fallback_used"] is False
    assert diagnostics["engine_corrected_start_at"] == MS_PER_HOUR


def test_skeleton_history_must_end_strictly_before_target_start() -> None:
    replay = object.__new__(evaluator.SkeletonKFReplay)
    replay._fetch_meta = lambda _event_id: {
        "start_at": 0,
        "end_at": 10 * MS_PER_HOUR,
        "event_type": "mission_live",
    }
    tracker = pd.DataFrame(
        {"time": [MS_PER_HOUR, 2 * MS_PER_HOUR], "ep": [10.0, 20.0]}
    )

    with pytest.raises(ValueError, match="not complete before target start"):
        replay._prepare_history(
            event_id=300,
            tier=500,
            frame=tracker,
            scale=1.0,
            scale_provenance={"source": "hhwx"},
            mode="same_rank",
            source_tiers=(500,),
            target_start_at=10 * MS_PER_HOUR,
            tracker_provenance={},
            source_provenance=(),
        )


def test_skeleton_tracker_prefers_frozen_exact_tier_without_provider_call() -> None:
    replay = object.__new__(evaluator.SkeletonKFReplay)
    replay._tracker_cache = {}
    header = SimpleNamespace(
        event_id=300,
        end_at=10 * MS_PER_HOUR,
        tracker_tiers=(500,),
        path=Path("300.json"),
        sha256="c" * 64,
    )
    replay.cache_headers = {300: header}
    replay._read_frozen_payload = lambda _header: {
        "tier_records": {
            "500": [
                {"time": MS_PER_HOUR, "ep": 10.0},
                {"time": 11 * MS_PER_HOUR, "ep": 20.0},
            ]
        }
    }

    class ForbiddenProvider:
        def fetch_tier_data(self, *args, **kwargs):
            raise AssertionError("provider must not run for a frozen exact tier")

    replay.data_source = ForbiddenProvider()
    frame, provenance = replay._fetch_tracker(300, 500)

    assert frame is not None
    assert frame["time"].max() == 10 * MS_PER_HOUR
    assert provenance["source"] == "hhwx"
    assert provenance["fallback_used"] is False
    assert provenance["tier_interpolation_used"] is False
    assert provenance["cache_sha256"] == "c" * 64


def test_skeleton_tracker_routes_missing_exact_tier_without_interpolation() -> None:
    replay = object.__new__(evaluator.SkeletonKFReplay)
    replay._tracker_cache = {}
    replay.cache_headers = {
        309: SimpleNamespace(
            event_id=309,
            end_at=10 * MS_PER_HOUR,
            tracker_tiers=(1000,),
            path=Path("309.json"),
            sha256="d" * 64,
        )
    }
    calls = []

    class RoutedProvider:
        def fetch_tier_data(self, event_id, *, tier, allow_fallback):
            calls.append((event_id, tier, allow_fallback))
            frame = pd.DataFrame(
                {
                    "time": [MS_PER_HOUR, 11 * MS_PER_HOUR],
                    "ep": [10.0, 20.0],
                }
            )
            frame.attrs.update(
                {
                    "source": "bestdori",
                    "requested_source": "hhwx",
                    "fallback_used": True,
                    "primary_error": "HHWX exact tier absent",
                }
            )
            return frame

    replay.data_source = RoutedProvider()
    frame, provenance = replay._fetch_tracker(309, 1500)

    assert frame is not None
    assert calls == [(309, 1500, True)]
    assert provenance["source"] == "bestdori"
    assert provenance["fallback_used"] is True
    assert provenance["tier"] == 1500
    assert provenance["tier_interpolation_used"] is False


def test_paired_history_intersection_aligns_ids_before_count_truncation() -> None:
    def prepared(event_id: int, *, mode: str, source_tier: int):
        return evaluator.PreparedSkeletonHistory(
            event_data=EventData(
                meta=EventMeta(
                    event_id,
                    "mission_live",
                    0,
                    2 * MS_PER_HOUR,
                    2 * MS_PER_HOUR,
                ),
                df=pd.DataFrame(
                    {"time": [0, MS_PER_HOUR], "value": [0.0, 1.0]}
                ),
                scale=1.0,
                tier=500,
            ),
            mode=mode,
            source_tiers=(source_tier,),
            provenance={"event_id": event_id, "source_tiers": [source_tier]},
        )

    same_rank = (
        prepared(11, mode="same_rank", source_tier=500),
        prepared(13, mode="same_rank", source_tier=500),
        prepared(12, mode="same_rank", source_tier=500),
    )
    reward_behavior = (
        prepared(14, mode="reward_behavior_class", source_tier=1000),
        prepared(13, mode="reward_behavior_class", source_tier=1500),
        prepared(11, mode="reward_behavior_class", source_tier=1000),
    )

    paired_same, paired_reward, candidate_ids, selected_ids = (
        evaluator._paired_history_intersection(
            same_rank,
            reward_behavior,
            count=1,
        )
    )

    assert candidate_ids == (13, 11)
    assert selected_ids == (13,)
    assert [item.event_data.meta.event_id for item in paired_same] == [13]
    assert [item.event_data.meta.event_id for item in paired_reward] == [13]
    assert paired_same[0].source_tiers == (500,)
    assert paired_reward[0].source_tiers == (1500,)


def test_reward_behavior_taxonomy_maps_each_class_to_its_own_rank() -> None:
    mapped = evaluator._reward_class_rank_map(
        {
            "voice_stamp:10136": 500,
            "voice_stamp:545": 1500,
            "deco_pins:103320": 2000,
        },
        event_id=318,
    )
    assert mapped == {
        "deco_pins": 2000,
        "voice_stamp_premium": 500,
        "voice_stamp_standard": 1500,
    }
    with pytest.raises(ValueError, match="ambiguous reward-class ranks"):
        evaluator._reward_class_rank_map(
            {"voice_stamp:1": 500, "voice_stamp:2": 1500},
            event_id=1,
        )


def _metric_row(
    event_id: int,
    tier: int,
    value: float,
    *,
    origin_hours: int = 24,
    missing_methods: tuple[str, ...] = (),
    input_available: bool = True,
) -> dict:
    missing = set(missing_methods)
    predictions = {}
    metrics = {}
    method_status = {}
    for method in evaluator.METHODS:
        success = input_available and method not in missing
        predictions[method] = 1000.0 + value if success else None
        metrics[method] = (
            {
                "error": value,
                "absolute_error": abs(value),
                "signed_percent_error": value,
                "absolute_percent_error": abs(value),
                "smape_percent": abs(value),
            }
            if success
            else None
        )
        method_status[method] = {
            "attempted": bool(input_available),
            "success": bool(success),
            "status": (
                "success"
                if success
                else "unavailable"
                if input_available
                else "not_attempted_common_input_unavailable"
            ),
            "failure_reason": (
                None
                if success
                else "no_legal_history"
                if input_available
                else "target_tier_has_fewer_than_two_visible_rows"
            ),
        }
    return {
        "event_id": event_id,
        "tier": tier,
        "origin_hours": origin_hours,
        "scoring_status": "evaluable" if input_available else "input_unavailable",
        "input_status": {
            "scheduled": True,
            "evaluable": input_available,
            "reason": (
                None
                if input_available
                else "target_tier_has_fewer_than_two_visible_rows"
            ),
        },
        "predictions": predictions,
        "metrics": metrics,
        "method_status": method_status,
    }


def _unequal_origin_rows() -> list[dict]:
    rows = [
        _metric_row(event_id, tier, 0.0)
        for event_id in evaluator.DEFAULT_EVENT_IDS
        for tier in evaluator.DEFAULT_REWARD_TIERS
    ]
    rows.extend(
        _metric_row(
            318,
            tier,
            100.0 if tier == 500 else 0.0,
            origin_hours=48,
        )
        for tier in evaluator.DEFAULT_REWARD_TIERS
    )
    return rows


def test_metrics_use_percent_units_and_hierarchical_event_weighting() -> None:
    metrics = evaluator._metrics(110.0, 100.0)
    assert metrics["signed_percent_error"] == pytest.approx(10.0)
    assert metrics["absolute_percent_error"] == pytest.approx(10.0)
    assert metrics["smape_percent"] == pytest.approx(2000.0 / 210.0)

    event_tiers = {
        event_id: evaluator.DEFAULT_REWARD_TIERS
        for event_id in evaluator.DEFAULT_EVENT_IDS
    }
    aggregates = evaluator.aggregate_results(_unequal_origin_rows(), event_tiers)
    overall = aggregates["overall_event_equal"]["methods"]["behavior_pace_model"]
    assert overall["supported_event_count"] == 2
    assert overall["mean_error"] == pytest.approx((50.0 + 5 * 0.0) / 6.0)
    first_cell = aggregates["per_event_tier"][0]["methods"]["behavior_pace_model"]
    assert first_cell["origin_count"] == 2
    assert first_cell["mean_error"] == pytest.approx(50.0)
    with pytest.raises(ValueError, match="canonical reward targets"):
        evaluator.aggregate_results(
            _unequal_origin_rows() + [_metric_row(318, 1000, 0.0)],
            event_tiers,
        )


def test_aggregate_equalizes_tiers_within_event_then_equalizes_events() -> None:
    event_tiers = {1: (500,), 2: (500, 1500)}
    rows = [
        _metric_row(1, 500, 0.0),
        _metric_row(2, 500, 100.0),
        _metric_row(2, 1500, 100.0),
    ]

    aggregates = evaluator.aggregate_results(rows, event_tiers)

    by_event = {
        item["event_id"]: item["methods"]["behavior_pace_model"]
        for item in aggregates["by_event_reward_tier_equal"]
    }
    assert by_event[1]["mean_error"] == pytest.approx(0.0)
    assert by_event[2]["mean_error"] == pytest.approx(100.0)
    overall = aggregates["overall_event_equal"]["methods"]["behavior_pace_model"]
    assert overall["mean_error"] == pytest.approx(50.0)


def test_aggregate_reports_common_input_and_paired_skeleton_support() -> None:
    skeleton_method = "skeleton_kf_same_rank_history"
    first = _metric_row(1, 500, 1.0, origin_hours=24)
    first["metrics"][skeleton_method] = {
        "error": 3.0,
        "absolute_error": 3.0,
        "signed_percent_error": 3.0,
        "absolute_percent_error": 3.0,
        "smape_percent": 3.0,
    }
    second = _metric_row(
        1,
        500,
        100.0,
        origin_hours=48,
        missing_methods=(skeleton_method,),
    )
    third = _metric_row(
        1,
        500,
        999.0,
        origin_hours=72,
        input_available=False,
    )

    aggregates = evaluator.aggregate_results(
        [first, second, third],
        {1: (500,)},
    )

    assert aggregates["input_coverage"] == {
        "scheduled_row_count": 3,
        "evaluable_row_count": 2,
        "input_unavailable_row_count": 1,
        "evaluable_fraction": pytest.approx(2 / 3),
        "scheduled_cell_count": 1,
        "cells_with_evaluable_origins": 1,
        "fully_evaluable_cell_count": 0,
    }
    coverage = aggregates["coverage_by_method"][skeleton_method]
    assert coverage["attempted_row_count"] == 2
    assert coverage["success_row_count"] == 1
    assert coverage["failure_row_count"] == 1
    assert coverage["evaluable_row_coverage_fraction"] == pytest.approx(0.5)
    paired = aggregates["paired_behavior_pace_vs_skeleton"][skeleton_method]
    assert paired["paired_success_row_count"] == 1
    assert paired["behavior_pace_model"][
        "mean_absolute_percent_error"
    ] == pytest.approx(1.0)
    assert paired[skeleton_method]["mean_absolute_percent_error"] == pytest.approx(
        3.0
    )
    assert paired["skeleton_minus_behavior_pace"][
        "mean_absolute_percent_error"
    ] == pytest.approx(2.0)


def test_aggregate_rejects_a_method_missing_from_one_scoring_row() -> None:
    row = _metric_row(1, 500, 0.0)
    del row["predictions"][evaluator.METHODS[-1]]

    with pytest.raises(ValueError, match="exactly METHODS"):
        evaluator.aggregate_results([row], {1: (500,)})


def test_planned_duration_average_baseline_is_named_for_its_raw_schedule() -> None:
    event = _event(318)
    prefix = evaluator._visible_prefix(event, 24 * MS_PER_HOUR)

    predictions = evaluator._baseline_predictions(event, prefix, 500)

    assert "full_elapsed_average_speed" not in predictions
    assert "full_elapsed_average_speed" not in evaluator.METHODS
    assert "planned_duration_average_speed" in evaluator.METHODS
    assert predictions["planned_duration_average_speed"] == pytest.approx(
        (30.0 + 500) * 49.0 / 12.0
    )


def test_behavior_pace_model_uses_frozen_prior_and_masked_target_prefix() -> None:
    event = _event(318, reward_tiers=(500,))
    origin_at = 24 * MS_PER_HOUR
    prefix = evaluator._visible_prefix(event, origin_at)
    pace_prior = evaluator.load_pace_prior(evaluator.DEFAULT_PACE_PRIOR_PATH)

    prediction, diagnostics = evaluator._predict_behavior_pace(
        event,
        prefix,
        origin_at=origin_at,
        tier=500,
        pace_prior=pace_prior,
    )

    assert prediction > 530.0
    assert diagnostics["anchor_at"] == 12 * MS_PER_HOUR
    assert diagnostics["forecast_end_at"] == event.end_at
    assert diagnostics["prior_sha256"] == pace_prior.sha256
    assert pace_prior.source_builder_sha256 == hashlib.sha256(
        (evaluator.PROJECT_ROOT / evaluator.PACE_BUILDER_SOURCE_NAME).read_bytes()
    ).hexdigest()
    assert diagnostics["fallback_used"] is False
    assert diagnostics["tier_interpolation_used"] is False


def test_plot_renders_paired_methods_and_planned_duration_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for event_id in evaluator.DEFAULT_EVENT_IDS:
        for tier in evaluator.DEFAULT_REWARD_TIERS:
            actual = float(1_000_000 + tier)
            row = _metric_row(event_id, tier, 0.0)
            row["actual_final_score"] = actual
            for index, method in enumerate(evaluator.METHODS):
                prediction = actual + float(index * 10_000)
                row["predictions"][method] = prediction
                row["metrics"][method] = evaluator._metrics(prediction, actual)
            rows.append(row)

    saved = {}

    def fake_savefig(_figure, destination, **kwargs):
        saved["destination"] = destination
        saved["kwargs"] = kwargs

    monkeypatch.setattr(evaluator.plt.Figure, "savefig", fake_savefig)
    output = Path("paired-evaluation.png")
    evaluator.render_plot(
        rows,
        output,
        event_tiers={
            event_id: evaluator.DEFAULT_REWARD_TIERS
            for event_id in evaluator.DEFAULT_EVENT_IDS
        },
    )

    assert saved["destination"] == output
    assert saved["kwargs"]["facecolor"] == "white"


def test_broad_plot_uses_summary_axis_and_accepts_partial_skeleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    event_tiers = {}
    for event_id in range(284, 291):
        event_tiers[event_id] = (1000,)
        missing = (
            ("skeleton_kf_same_rank_history",)
            if event_id == 284
            else ()
        )
        row = _metric_row(
            event_id,
            1000,
            float(event_id - 283),
            missing_methods=missing,
        )
        row["actual_final_score"] = 1_000_000.0
        rows.append(row)

    saved = {}

    def fake_savefig(figure, destination, **kwargs):
        saved["axis_count"] = len(figure.axes)
        saved["destination"] = destination

    monkeypatch.setattr(evaluator.plt.Figure, "savefig", fake_savefig)
    output = Path("broad-summary.png")
    evaluator.render_plot(rows, output, event_tiers=event_tiers)

    assert saved == {"axis_count": 1, "destination": output}


def test_document_records_source_contract_execution_and_exact_implementation_hashes() -> None:
    raw_truth = {
        tier: ({"ep": 1000.0 + tier, "isFinal": True},)
        for tier in evaluator.DEFAULT_REWARD_TIERS
    }
    loaded = [
        _loaded(event_id, raw_truth)
        for event_id in evaluator.DEFAULT_EVENT_IDS
    ]
    evaluator_path = Path(evaluator.__file__).resolve()
    assert evaluator._evaluator_source_sha256() == hashlib.sha256(
        evaluator_path.read_bytes()
    ).hexdigest()
    execution = {
        "started_at": "2026-08-20T00:00:00+00:00",
        "completed_at": "2026-08-20T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "timestamp_timezone": "UTC",
    }
    evaluator_sha256 = "e" * 64
    history_provenance = [
        {
            "target_event_id": 318,
            "target_tier": 500,
            "paired_intersection": {
                "candidate_event_ids": [315, 307],
                "selected_event_ids": [315],
            },
        }
    ]

    document = evaluator.build_document(
        loaded,
        [],
        [_metric_row(event_id, tier, 0.0, origin_hours=hour)
         for event_id in evaluator.DEFAULT_EVENT_IDS
         for tier in evaluator.DEFAULT_REWARD_TIERS
        for hour in (24, 48)],
        requested_event_ids=evaluator.DEFAULT_EVENT_IDS,
        pace_prior=evaluator.load_pace_prior(evaluator.DEFAULT_PACE_PRIOR_PATH),
        skeleton_preset=evaluator.load_skeleton_preset(),
        skeleton_source_hashes={
            name: "f" * 64 for name in evaluator.SKELETON_SOURCE_FILES
        },
        skeleton_history_provenance=history_provenance,
        skeleton_failure_count=0,
        execution=execution,
        evaluator_source_sha256=evaluator_sha256,
    )

    assert document["schema_version"] == "reward-tier-evaluation-v6"
    assert document["execution"] == execution
    assert document["evaluation_pipeline"] == {
        "implementation": {
            "hash_algorithm": "sha256",
            "files": {
                evaluator.EVALUATOR_SOURCE_NAME: evaluator_sha256,
            },
        },
        "scheduled_grid": (
            "every canonical event/tier has every 24-hour scheduled origin"
        ),
        "common_input_gate": (
            "all methods are unscored when the target tier has fewer than two "
            "rows visible at the origin"
        ),
        "method_support": (
            "pace and three naive baselines cover every common-evaluable row; "
            "Skeleton methods report explicit available-case coverage"
        ),
    }
    assert "model" not in document
    assert document["scope"]["source_contract"] == {
        "target_provider": "hhwx_frozen_cache",
        "history_and_t10_provider_route": "hhwx_then_bestdori",
        "server": 3,
        "event_start_field": "meta.start_at",
        "event_end_field": "meta.end_at",
        "tracker_time_field": "tier_records[].time",
        "tracker_score_field": "tier_records[].ep",
        "final_truth_flag_field": "not_used_for_truth_selection",
        "timestamp_unit": "unix_epoch_milliseconds",
        "event_display_timezone": "Asia/Shanghai",
    }
    assert document["scope"]["provider_routing"]["bestdori_authorized"] is True
    assert document["scope"]["provider_routing"]["rank_interpolation_used"] is False
    assert document["scope"]["t1000_evaluated"] is False
    assert document["scope"]["requested_event_ids"] == [318, 319]
    assert document["scope"]["evaluated_event_ids"] == [318, 319]
    assert document["scope"]["excluded_event_ids"] == []
    assert document["scope"]["canonical_reward_tiers_by_event"] == {
        "318": [500, 1500, 2000],
        "319": [500, 1500, 2000],
    }
    assert document["behavior_pace_model"]["target_runs_by_event"] == {
        "318": ["T500", "T1500", "T2000"],
        "319": ["T500", "T1500", "T2000"],
    }
    pace = document["behavior_pace_model"]
    assert pace["implementation"]["files"] == {
        evaluator.PACE_MODEL_SOURCE_NAME: hashlib.sha256(
            (evaluator.PROJECT_ROOT / evaluator.PACE_MODEL_SOURCE_NAME).read_bytes()
        ).hexdigest(),
        evaluator.PACE_BUILDER_SOURCE_NAME: hashlib.sha256(
            (evaluator.PROJECT_ROOT / evaluator.PACE_BUILDER_SOURCE_NAME).read_bytes()
        ).hexdigest(),
    }
    assert pace["prior"]["sha256"] == hashlib.sha256(
        evaluator.DEFAULT_PACE_PRIOR_PATH.read_bytes()
    ).hexdigest()
    assert pace["diagnostics_location"] == "per_origin[].behavior_pace_model"
    assert pace["fallback_used"] is False
    assert document["production_baseline_model"]["prediction_failure_count"] == 0
    assert document["production_baseline_model"]["preset"]["params"]
    assert document["production_baseline_model"]["provider_routing"] == (
        "hhwx_then_bestdori_same_exact_tier_or_t10"
    )
    assert document["production_baseline_model"]["history_selection"] == (
        history_provenance
    )
    assert {
        "skeleton_kf_same_rank_paired_intersection",
        "skeleton_kf_reward_behavior_paired_intersection",
    } <= set(document["production_baseline_model"]["methods"])
    assert "planned_duration_average_speed" in document["baseline_definitions"]
    assert "full_elapsed_average_speed" not in document["baseline_definitions"]


def test_document_records_empty_reward_event_as_not_applicable() -> None:
    excluded = _excluded(317)
    document = evaluator.build_document(
        [],
        [excluded],
        [],
        requested_event_ids=(317,),
        pace_prior=evaluator.load_pace_prior(evaluator.DEFAULT_PACE_PRIOR_PATH),
        skeleton_preset=evaluator.load_skeleton_preset(),
        skeleton_source_hashes={
            name: "f" * 64 for name in evaluator.SKELETON_SOURCE_FILES
        },
        skeleton_history_provenance=[],
        skeleton_failure_count=0,
        execution={"started_at": "2026-08-21T00:00:00+00:00"},
        evaluator_source_sha256="e" * 64,
    )

    assert document["scope"]["status"] == "not_applicable"
    assert document["scope"]["evaluated_event_ids"] == []
    assert document["scope"]["excluded_event_ids"] == [317]
    assert document["inputs"]["excluded"][0]["exclusion_reason"] == (
        "empty_canonical_reward_tiers"
    )
    assert document["aggregates"]["status"] == "not_applicable"
    assert document["aggregates"]["overall_event_equal"] is None
