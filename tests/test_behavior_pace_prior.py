from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import behavior_pace_model
import behavior_pace_prior
from behavior_pace_prior import (
    CacheFormatError,
    CoverageGateError,
    DEFAULT_TRAINING_EVENT_IDS,
    SCHEMA_VERSION,
    TRAINING_TIERS,
    TrainingBoundaryError,
    build_behavior_pace_prior,
    decode_training_event,
    dumps_behavior_pace_prior,
    estimate_behavior_pace_prior,
    exact_event_files,
    write_behavior_pace_prior_atomic,
)


START_AT = 1_700_000_000_000
END_AT = START_AT + 4 * 3_600_000


def _records(*rows: tuple[int, float]) -> list[dict[str, int | float]]:
    return [{"time": timestamp, "ep": score} for timestamp, score in rows]


def _payload(event_id: int) -> dict:
    return {
        "event_id": event_id,
        "meta": {
            "event_id": event_id,
            "start_at": START_AT,
            "end_at": END_AT,
            "event_type": "mission_live",
        },
        "reward_tiers": [500, 1500],
        "reward_tier_provenance": {
            "source": "hhwx",
            "server": 3,
            "observed_at": END_AT + 30 * 60_000,
            "last_appearance": {
                "deco_pins:200": 1500,
                "voice_stamp:100": 500,
            },
        },
        "tier_records": {
            "50": _records(
                (START_AT + 15 * 60_000, 70.0),
                (START_AT + 135 * 60_000, 150.0),
                (END_AT + 1, 210.0),
            ),
            "100": _records(
                (START_AT + 20 * 60_000, 55.0),
                (START_AT + 140 * 60_000, 125.0),
                (END_AT + 2, 180.0),
            ),
            "300": _records(
                (START_AT + 40 * 60_000, 35.0),
                (START_AT + 160 * 60_000, 85.0),
                (END_AT + 3, 130.0),
            ),
            "500": _records(
                (START_AT + 1 * 3_600_000, 10.0),
                (START_AT + 2 * 3_600_000, 20.0),
                (END_AT + 60_000, 31.0),
            ),
            "1000": _records(
                (START_AT + 70 * 60_000, 8.0),
                (START_AT + 190 * 60_000, 18.0),
                (END_AT + 5, 27.0),
            ),
            "2000": _records(
                (START_AT + 80 * 60_000, 6.0),
                (START_AT + 200 * 60_000, 14.0),
                (END_AT + 6, 21.0),
            ),
            # Extra post-225 schema tier: parsed for quality, never fitted.
            "1500": _records(
                (START_AT + 30 * 60_000, 5.0),
                (START_AT + 150 * 60_000, 16.0),
                (END_AT + 120_000, 23.0),
            ),
        },
    }


def _raw(payload: dict) -> bytes:
    return json.dumps(payload, allow_nan=False).encode("utf-8")


def _write_cache(path: Path, payload: dict) -> None:
    path.write_bytes(_raw(payload))


def _backend(call_log: dict) -> SimpleNamespace:
    @dataclass(frozen=True)
    class PaceModelConfig:
        ridge: float = 0.0
        utc_offset_hours: float = 8.0

    @dataclass(frozen=True)
    class PaceTrainingEvent:
        event_id: int
        start_at: int
        end_at: int
        duration_hours: float
        reward_tiers: tuple[int, ...]
        tier_observations: object
        event_start_local_hour: float

    def fit_event_weights(event, config):
        call_log.setdefault("fit", []).append((event, config))
        weights = {
            192: np.array([0.2, 0.3, 0.5]),
            193: np.array([0.4, 0.1, 0.5]),
        }.get(event.event_id, np.array([0.1, 0.2, 0.7]))
        return {
            "weights": weights,
            "diagnostics": {
                "tier_count": len(event.tier_observations),
                "synchronized": False,
            },
        }

    def aggregate_event_weights(vectors):
        call_log["aggregate"] = tuple(np.asarray(item).copy() for item in vectors)
        return {
            "weights": np.mean(np.vstack(vectors), axis=0),
            "diagnostics": {"event_count": len(vectors)},
        }

    return SimpleNamespace(
        __name__="fake_behavior_pace_model",
        PACE_COMPONENT_NAMES=("recent", "circadian", "deadline"),
        PaceModelConfig=PaceModelConfig,
        PaceTrainingEvent=PaceTrainingEvent,
        fit_event_weights=fit_event_weights,
        aggregate_event_weights=aggregate_event_weights,
    )


def test_decode_uses_fixed_asynchronous_training_support_and_maps_terminal():
    event = decode_training_event(_raw(_payload(192)), expected_event_id=192)

    assert event.event_id == 192
    assert event.reward_tiers == (500, 1500)
    assert tuple(event.tier_observations) == TRAINING_TIERS
    np.testing.assert_allclose(
        event.tier_observations[500],
        np.array(
            [
                [0.0, 0.0],
                [1.0, 10.0],
                [2.0, 20.0],
                [4.0, 31.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        event.tier_observations[100],
        np.array(
            [
                [0.0, 0.0],
                [1 / 3, 55.0],
                [7 / 3, 125.0],
                [4.0, 180.0],
            ]
        ),
    )
    assert not event.tier_observations[500].flags.writeable
    assert event.data_quality["rank_reference_tier"] is None
    assert event.data_quality["asynchronous_tier_timestamps_allowed"] is True
    assert event.data_quality["reward_tier_provenance"] == {
        "source": "hhwx",
        "server": 3,
        "observed_at": END_AT + 30 * 60_000,
        "last_appearance": {
            "deco_pins:200": 1500,
            "voice_stamp:100": 500,
        },
    }
    assert event.data_quality["tiers"]["500"]["post_end_rows_mapped"] == 1
    assert (
        event.data_quality["tiers"]["100"]["terminal_mapping_delay_ms"]
        == 2
    )
    assert event.data_quality["ignored_extra_tiers"] == [1500]
    assert (
        event.data_quality["tiers"]["1500"]["terminal_mapping_delay_ms"]
        == 120_000
    )


def test_terminal_uses_first_post_end_bucket_and_only_diagnoses_later_regression():
    payload = _payload(192)
    payload["tier_records"]["500"] = _records(
        (START_AT + 1 * 3_600_000, 10.0),
        (START_AT + 2 * 3_600_000, 20.0),
        (END_AT + 1, 31.0),
        # A later tracker revision is not allowed to redefine the terminal.
        (END_AT + 15 * 60_000, 7.0),
    )

    event = decode_training_event(_raw(payload), expected_event_id=192)

    assert event.tier_observations[500][-1].tolist() == [4.0, 31.0]
    quality = event.data_quality["tiers"]["500"]
    assert quality["post_end_rows_observed"] == 2
    assert quality["post_end_rows_mapped"] == 1
    assert quality["post_end_followup_count"] == 1
    assert quality["post_end_disagreement_count"] == 1
    assert quality["post_end_disagreement_values"] == [
        {
            "time": END_AT + 15 * 60_000,
            "delay_ms": 15 * 60_000,
            "score": 7.0,
        }
    ]


def test_decode_repairs_tracker_revisions_per_tier_and_reports_them():
    payload = _payload(192)
    payload["tier_records"]["500"] = _records(
        (START_AT + 1 * 3_600_000, 10.0),
        (START_AT + 2 * 3_600_000, 8.0),
        (END_AT + 1, 12.0),
    )

    event = decode_training_event(_raw(payload), expected_event_id=192)

    assert event.tier_observations[500][:, 1].tolist() == [0.0, 10.0, 10.0, 12.0]
    assert (
        event.data_quality["tiers"]["500"]["non_monotonic_steps_repaired"]
        == 1
    )


def test_extra_schema_tiers_do_not_change_model_training_input():
    with_extra = _payload(192)
    fixed_only = _payload(192)
    del fixed_only["tier_records"]["1500"]

    decoded_extra = decode_training_event(
        _raw(with_extra), expected_event_id=192
    )
    decoded_fixed = decode_training_event(
        _raw(fixed_only), expected_event_id=192
    )

    assert tuple(decoded_extra.tier_observations) == TRAINING_TIERS
    assert tuple(decoded_fixed.tier_observations) == TRAINING_TIERS
    for tier in TRAINING_TIERS:
        np.testing.assert_array_equal(
            decoded_extra.tier_observations[tier],
            decoded_fixed.tier_observations[tier],
        )

@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda payload: payload.update(reward_tiers=[1500, 500]),
            "sorted and unique",
        ),
        (
            lambda payload: payload["tier_records"].update(
                {"999": payload["tier_records"].pop("500")}
            ),
            "unsupported tracker tier",
        ),
        (
            lambda payload: payload["tier_records"].update(
                {
                    "500": _records(
                        (START_AT + 3_600_000, 10.0),
                        (END_AT + 1, 20.0),
                    )
                }
            ),
            "at least two observations inside",
        ),
        (
            lambda payload: payload["meta"].update(end_at=START_AT),
            "later than",
        ),
    ],
)
def test_decode_rejects_malformed_event_contract(mutate, message):
    payload = _payload(192)
    mutate(payload)

    with pytest.raises(CacheFormatError, match=message):
        decode_training_event(_raw(payload), expected_event_id=192)


def test_decode_rejects_nonfinite_json_and_event_id_mismatch():
    payload = _payload(192)
    payload["tier_records"]["500"][0]["ep"] = float("nan")
    nonfinite = json.dumps(payload).encode("utf-8")

    with pytest.raises(CacheFormatError, match="strict JSON"):
        decode_training_event(nonfinite, expected_event_id=192)
    with pytest.raises(CacheFormatError, match="does not match"):
        decode_training_event(_raw(_payload(193)), expected_event_id=192)


def test_decode_requires_reward_tier_provenance():
    payload = _payload(192)
    del payload["reward_tier_provenance"]

    with pytest.raises(CacheFormatError, match="reward_tier_provenance must be an object"):
        decode_training_event(_raw(payload), expected_event_id=192)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda provenance: provenance.update(source="bestdori"),
            "source must be 'hhwx'",
        ),
        (
            lambda provenance: provenance.update(server=2),
            "server must be 3",
        ),
        (
            lambda provenance: provenance.update(observed_at=0),
            "observed_at must be a positive integer",
        ),
        (
            lambda provenance: provenance.update(last_appearance=[]),
            "last_appearance must be an object",
        ),
        (
            lambda provenance: provenance.update(
                last_appearance={"voice_stamp:100": 500}
            ),
            "do not match reward_tiers",
        ),
    ],
)
def test_decode_rejects_conflicting_reward_tier_provenance(mutate, message):
    payload = _payload(192)
    mutate(payload["reward_tier_provenance"])

    with pytest.raises(CacheFormatError, match=message):
        decode_training_event(_raw(payload), expected_event_id=192)


def test_builder_fits_each_event_once_aggregates_event_equally_and_records_hashes(
    tmp_path: Path,
):
    event_192 = tmp_path / "arbitrary-exact-name-192.json"
    event_193 = tmp_path / "193.json"
    event_194 = tmp_path / "event_194.json"
    _write_cache(event_192, _payload(192))
    _write_cache(event_193, _payload(193))
    bad = _payload(194)
    bad["tier_records"]["500"][0]["ep"] = "not-a-score"
    event_194.write_text(json.dumps(bad), encoding="utf-8")
    model_source = tmp_path / "behavior_pace_model.py"
    model_source.write_text("# exact model source snapshot\n", encoding="utf-8")

    calls: dict = {}
    backend = _backend(calls)
    document = build_behavior_pace_prior(
        {192: event_192, 193: event_193, 194: event_194},
        event_ids=(192, 193, 194),
        backend=backend,
        model_source_path=model_source,
        minimum_included_events=2,
        minimum_included_fraction=0.6,
        generated_at="2026-08-21T00:00:00+00:00",
    )

    assert document["schema_version"] == SCHEMA_VERSION
    assert document["included_event_ids"] == [192, 193]
    assert len(calls["fit"]) == 2
    assert all(
        tuple(event.tier_observations) == TRAINING_TIERS
        for event, _ in calls["fit"]
    )
    assert all(
        event.event_start_local_hour
        == pytest.approx((START_AT / 3_600_000 + 8.0) % 24.0)
        for event, _ in calls["fit"]
    )
    assert len(calls["aggregate"]) == 2
    np.testing.assert_allclose(calls["aggregate"][0], [0.2, 0.3, 0.5])
    np.testing.assert_allclose(calls["aggregate"][1], [0.4, 0.1, 0.5])
    assert document["aggregate_weights"] == {
        "recent": pytest.approx(0.3),
        "circadian": pytest.approx(0.2),
        "deadline": pytest.approx(0.5),
    }
    assert document["aggregate_diagnostics"] == {"event_count": 2}
    assert document["event_fits"][0]["diagnostics"]["model"] == {
        "tier_count": 6,
        "synchronized": False,
    }
    assert document["excluded_events"][0]["event_id"] == 194
    assert document["excluded_events"][0]["reason"] == "invalid_cache"
    assert document["coverage_gate"] == {
        "minimum_included_events": 2,
        "minimum_included_fraction": 0.6,
        "requested_event_count": 3,
        "required_included_event_count": 2,
        "actual_included_event_count": 2,
        "actual_included_fraction": pytest.approx(2 / 3),
        "passed": True,
    }
    inputs = {item["event_id"]: item for item in document["input_files"]}
    assert inputs[192]["file_name"] == event_192.name
    assert inputs[192]["sha256"] == hashlib.sha256(event_192.read_bytes()).hexdigest()
    assert document["source_model"]["sha256"] == hashlib.sha256(
        model_source.read_bytes()
    ).hexdigest()
    builder_source = Path(behavior_pace_prior.__file__)
    builder_bytes = builder_source.read_bytes()
    assert document["source_builder"] == {
        "module": "behavior_pace_prior",
        "file_name": "behavior_pace_prior.py",
        "sha256": hashlib.sha256(builder_bytes).hexdigest(),
        "size_bytes": len(builder_bytes),
    }
    assert document["algorithm_config"]["rank_reference_tier"] is None
    assert document["algorithm_config"]["tier_timestamp_alignment"] == "not_required"
    assert document["algorithm_config"]["training_tier_support"] == list(
        TRAINING_TIERS
    )
    assert json.loads(dumps_behavior_pace_prior(document))["aggregate_weights"] == {
        "circadian": pytest.approx(0.2),
        "deadline": pytest.approx(0.5),
        "recent": pytest.approx(0.3),
    }


def test_source_builder_fingerprints_exact_final_module_bytes():
    builder_source = Path(behavior_pace_prior.__file__)
    builder_bytes = builder_source.read_bytes()

    assert behavior_pace_prior._builder_source() == {
        "module": "behavior_pace_prior",
        "file_name": "behavior_pace_prior.py",
        "sha256": hashlib.sha256(builder_bytes).hexdigest(),
        "size_bytes": len(builder_bytes),
    }


def test_builder_passes_configured_utc_offset_to_model_event(tmp_path: Path):
    cache = tmp_path / "192.json"
    _write_cache(cache, _payload(192))
    model_source = tmp_path / "behavior_pace_model.py"
    model_source.write_text("# model\n", encoding="utf-8")
    calls: dict = {}
    backend = _backend(calls)
    config = backend.PaceModelConfig(utc_offset_hours=-5.5)

    build_behavior_pace_prior(
        {192: cache},
        event_ids=(192,),
        backend=backend,
        model_config=config,
        model_source_path=model_source,
        minimum_included_events=1,
        minimum_included_fraction=1.0,
        generated_at="2026-08-21T00:00:00+00:00",
    )

    model_event = calls["fit"][0][0]
    assert model_event.event_start_local_hour == pytest.approx(
        (START_AT / 3_600_000 - 5.5) % 24.0
    )


def test_builder_does_not_guess_an_alternate_cache_filename(tmp_path: Path):
    supplied_missing = tmp_path / "192.json"
    _write_cache(tmp_path / "event_192.json", _payload(192))
    model_source = tmp_path / "behavior_pace_model.py"
    model_source.write_text("# model\n", encoding="utf-8")

    with pytest.raises(CoverageGateError) as captured:
        build_behavior_pace_prior(
            {192: supplied_missing},
            event_ids=(192,),
            backend=_backend({}),
            model_source_path=model_source,
            minimum_included_events=1,
            minimum_included_fraction=1.0,
            generated_at="2026-08-21T00:00:00+00:00",
        )

    document = captured.value.document
    assert document["included_event_ids"] == []
    assert document["excluded_events"] == [
        {
            "event_id": 192,
            "file_name": "192.json",
            "reason": "unreadable_input",
            "detail": f"cannot read exact cache path {supplied_missing}",
        }
    ]
    assert document["coverage_gate"]["passed"] is False


def test_builder_integrates_with_public_behavior_pace_model_api(tmp_path: Path):
    cache = tmp_path / "192.json"
    _write_cache(cache, _payload(192))

    document = build_behavior_pace_prior(
        {192: cache},
        event_ids=(192,),
        backend=behavior_pace_model,
        minimum_included_events=1,
        minimum_included_fraction=1.0,
        generated_at="2026-08-21T00:00:00+00:00",
    )

    assert document["included_event_ids"] == [192]
    assert document["component_names"] == ["sustain", "launch", "deadline"]
    assert sum(document["aggregate_weights"].values()) == pytest.approx(1.0)
    diagnostics = document["event_fits"][0]["diagnostics"]["model"]
    assert diagnostics["tier_count"] == 6
    assert diagnostics["interval_count"] == 18
    assert diagnostics["objective"] >= 0.0
    assert document["source_model"]["file_name"] == "behavior_pace_model.py"


def test_missing_mapping_is_an_explicit_exclusion_and_gate_failure(tmp_path: Path):
    model_source = tmp_path / "behavior_pace_model.py"
    model_source.write_text("# model\n", encoding="utf-8")

    with pytest.raises(CoverageGateError) as captured:
        build_behavior_pace_prior(
            {},
            event_ids=(192,),
            backend=_backend({}),
            model_source_path=model_source,
            minimum_included_events=1,
            minimum_included_fraction=1.0,
            generated_at="2026-08-21T00:00:00+00:00",
        )

    assert captured.value.document["excluded_events"] == [
        {
            "event_id": 192,
            "reason": "missing_file_mapping",
            "detail": "no exact cache path was supplied for this requested event",
        }
    ]


def test_invalid_event_ids_are_rejected_before_mapping_access():
    class ExplodingMapping(dict):
        def items(self):
            raise AssertionError("event_files must not be inspected")

    with pytest.raises(TrainingBoundaryError, match="integers"):
        build_behavior_pace_prior(
            ExplodingMapping(),
            event_ids=(192, "193"),
            backend=_backend({}),
            model_source_path=Path(__file__),
        )


def test_default_training_range_is_explicit_and_stable():
    assert DEFAULT_TRAINING_EVENT_IDS == tuple(range(192, 284))


def test_training_range_rejects_newer_events():
    with pytest.raises(TrainingBoundaryError, match="frozen range"):
        build_behavior_pace_prior(
            {},
            event_ids=(284,),
            backend=_backend({}),
            model_source_path=Path(__file__),
        )


def test_exact_event_files_never_probes_alternate_names(tmp_path: Path):
    result = exact_event_files(tmp_path, (192, 193))

    assert result == {
        192: tmp_path / "192.json",
        193: tmp_path / "193.json",
    }


def test_estimator_expands_only_canonical_exact_cache_names(tmp_path: Path):
    _write_cache(tmp_path / "192.json", _payload(192))
    model_source = tmp_path / "behavior_pace_model.py"
    model_source.write_text("# model\n", encoding="utf-8")

    document = estimate_behavior_pace_prior(
        tmp_path,
        event_ids=(192,),
        backend=_backend({}),
        model_source_path=model_source,
        minimum_included_events=1,
        minimum_included_fraction=1.0,
        generated_at="2026-08-21T00:00:00+00:00",
    )

    assert document["input_files"][0]["file_name"] == "192.json"


def test_atomic_writer_replaces_only_exact_destination(tmp_path: Path):
    destination = tmp_path / "prior.json"
    destination.write_text("old", encoding="utf-8")
    document = {
        "schema_version": SCHEMA_VERSION,
        "value": {"b": 2, "a": 1},
    }

    returned = write_behavior_pace_prior_atomic(destination, document)

    assert returned == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == document
    assert list(tmp_path.iterdir()) == [destination]
