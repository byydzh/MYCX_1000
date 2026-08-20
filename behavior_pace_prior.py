"""Training-only prior builder for the three-component pace model.

The builder deliberately accepts an explicit ``event_id -> cache path`` mapping.
It never scans a directory, guesses a cache naming convention, or substitutes a
different event.  Training uses the same six-tier measurement panel for every
event; T1000 is one symmetric observation in that panel, not a scale or rank
anchor.  Every event is decoded independently, fitted once, and contributes one
simplex vector to the event-equal aggregate.

The numerical model lives in :mod:`behavior_pace_model`.  Importing that module
is delayed so cache-contract tests can use a tiny injected backend while the
model and loader evolve independently.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import numbers
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from config import TRACKER_TIER_SET


SCHEMA_VERSION = "behavior-pace-prior-v1"
TRAIN_EVENT_MIN = 192
TRAIN_EVENT_MAX = 283
DEFAULT_TRAINING_EVENT_IDS = tuple(range(TRAIN_EVENT_MIN, TRAIN_EVENT_MAX + 1))
DEFAULT_MINIMUM_INCLUDED_EVENTS = 30
DEFAULT_MINIMUM_INCLUDED_FRACTION = 0.80
DEFAULT_FINAL_BUCKET_MAX_DELAY_MS = 20 * 60 * 1000
MS_PER_HOUR = 3_600_000
SIMPLEX_COMPONENT_COUNT = 3
# Common measurement support across every frozen 192..283 cache.  T1000 is
# one ordinary member of this six-tier panel, never a scale or rank anchor.
TRAINING_TIERS = (50, 100, 300, 500, 1_000, 2_000)


class TrainingBoundaryError(ValueError):
    """Raised before file access when requested event identifiers are invalid."""


class CacheFormatError(ValueError):
    """Raised when one explicitly named cache violates the training contract."""


class CoverageGateError(RuntimeError):
    """Raised when too few explicitly requested events produce valid weights."""

    def __init__(self, message: str, document: Mapping[str, Any]):
        super().__init__(message)
        self.document = dict(document)


@dataclass(frozen=True)
class ParsedPaceTrainingEvent:
    """Strictly decoded, asynchronous fixed-tier observations for one event."""

    event_id: int
    start_at: int
    end_at: int
    reward_tiers: tuple[int, ...]
    tier_observations: Mapping[int, np.ndarray]
    data_quality: Mapping[str, Any]

    @property
    def duration_hours(self) -> float:
        return (self.end_at - self.start_at) / MS_PER_HOUR


@dataclass(frozen=True)
class _InputFile:
    event_id: int
    path: Path
    file_name: str
    sha256: str
    size_bytes: int
    raw: bytes


def validate_training_event_ids(
    event_ids: Iterable[int] = DEFAULT_TRAINING_EVENT_IDS,
) -> tuple[int, ...]:
    """Return sorted unique positive IDs, rejecting the whole request first."""

    try:
        requested = tuple(event_ids)
    except TypeError as exc:
        raise TrainingBoundaryError("event_ids must be an iterable of integers") from exc
    if not requested:
        raise TrainingBoundaryError("at least one training event ID is required")

    normalized: list[int] = []
    for value in requested:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TrainingBoundaryError(
                f"training event IDs must be integers, got {value!r}"
            )
        event_id = int(value)
        if not TRAIN_EVENT_MIN <= event_id <= TRAIN_EVENT_MAX:
            raise TrainingBoundaryError(
                f"training event {event_id} is outside the frozen range "
                f"[{TRAIN_EVENT_MIN}, {TRAIN_EVENT_MAX}]"
            )
        normalized.append(event_id)
    if len(normalized) != len(set(normalized)):
        raise TrainingBoundaryError("training event IDs must be unique")
    return tuple(sorted(normalized))


def _validate_event_file_mapping(
    event_files: Mapping[int, str | Path],
) -> dict[int, Path]:
    if not isinstance(event_files, Mapping):
        raise TrainingBoundaryError(
            "event_files must explicitly map integer event IDs to cache paths"
        )
    result: dict[int, Path] = {}
    for raw_event_id, raw_path in event_files.items():
        if isinstance(raw_event_id, bool) or not isinstance(
            raw_event_id, numbers.Integral
        ):
            raise TrainingBoundaryError(
                f"event_files keys must be integer event IDs, got {raw_event_id!r}"
            )
        event_id = int(raw_event_id)
        if event_id <= 0:
            raise TrainingBoundaryError("event_files keys must be positive")
        if not isinstance(raw_path, (str, Path)):
            raise TrainingBoundaryError(
                f"event_files[{event_id}] must be one exact path"
            )
        result[event_id] = Path(raw_path)
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CacheFormatError(f"{label} is not UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise CacheFormatError(f"{label} is not strict JSON: {exc}") from exc


def _strict_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise CacheFormatError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise CacheFormatError(f"{label} must be a positive integer")
    return result


def _strict_timestamp(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise CacheFormatError(f"{label} must be an integer millisecond timestamp")
    result = int(value)
    if result < 0:
        raise CacheFormatError(f"{label} must be non-negative")
    return result


def _strict_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CacheFormatError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise CacheFormatError(f"{label} must be a finite non-negative number")
    return result


def _canonical_reward_tiers(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise CacheFormatError(f"{label} must be a JSON list")
    tiers = tuple(
        _strict_positive_int(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if tiers != tuple(sorted(set(tiers))):
        raise CacheFormatError(f"{label} must be sorted and unique")
    unsupported = [tier for tier in tiers if tier not in TRACKER_TIER_SET]
    if unsupported:
        raise CacheFormatError(
            f"{label} contains unsupported public tracker tiers: {unsupported}"
        )
    return tiers


def _validated_reward_provenance(
    value: Any,
    *,
    event_id: int,
    reward_tiers: tuple[int, ...],
) -> dict[str, Any]:
    """Validate the exact evidence behind one cache's reward-tier targets."""

    label = f"event {event_id} reward_tier_provenance"
    if not isinstance(value, Mapping):
        raise CacheFormatError(f"{label} must be an object")

    source = value.get("source")
    if source != "hhwx":
        raise CacheFormatError(f"{label}.source must be 'hhwx'")

    server = value.get("server")
    if isinstance(server, bool) or not isinstance(server, numbers.Integral):
        raise CacheFormatError(f"{label}.server must be integer 3")
    if int(server) != 3:
        raise CacheFormatError(f"{label}.server must be 3")

    observed_at = _strict_positive_int(
        value.get("observed_at"), f"{label}.observed_at"
    )
    last_appearance = value.get("last_appearance")
    if not isinstance(last_appearance, Mapping):
        raise CacheFormatError(f"{label}.last_appearance must be an object")

    canonical_last_appearance: dict[str, int] = {}
    for raw_key, raw_tier in last_appearance.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise CacheFormatError(
                f"{label}.last_appearance keys must be non-empty strings"
            )
        tier = _strict_positive_int(
            raw_tier, f"{label}.last_appearance[{raw_key!r}]"
        )
        if tier not in TRACKER_TIER_SET:
            raise CacheFormatError(
                f"{label}.last_appearance[{raw_key!r}] names unsupported "
                f"tracker tier T{tier}"
            )
        canonical_last_appearance[raw_key] = tier

    derived_tiers = tuple(sorted(set(canonical_last_appearance.values())))
    if derived_tiers != reward_tiers:
        raise CacheFormatError(
            f"{label}.last_appearance tiers {derived_tiers} do not match "
            f"reward_tiers {reward_tiers}"
        )

    return {
        "source": "hhwx",
        "server": 3,
        "observed_at": observed_at,
        "last_appearance": dict(sorted(canonical_last_appearance.items())),
    }


def _canonical_tier_key(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.isdecimal():
        raise CacheFormatError(f"{label} must be a canonical decimal string")
    tier = int(value)
    if tier <= 0 or str(tier) != value:
        raise CacheFormatError(f"{label} must be a canonical decimal string")
    if tier not in TRACKER_TIER_SET:
        raise CacheFormatError(f"{label} names unsupported tracker tier T{tier}")
    return tier


def _parse_tier_observations(
    records: Any,
    *,
    event_id: int,
    tier: int,
    start_at: int,
    end_at: int,
    final_bucket_max_delay_ms: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    label = f"event {event_id} T{tier}"
    if not isinstance(records, list) or not records:
        raise CacheFormatError(f"{label} tier_records entry must be a non-empty list")

    parsed: list[tuple[int, float]] = []
    seen_timestamps: set[int] = set()
    for row_index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise CacheFormatError(f"{label} row {row_index} must be an object")
        if "time" not in row or "ep" not in row:
            raise CacheFormatError(f"{label} row {row_index} requires time and ep")
        timestamp = _strict_timestamp(
            row["time"], f"{label} row {row_index}.time"
        )
        score = _strict_score(row["ep"], f"{label} row {row_index}.ep")
        if timestamp in seen_timestamps:
            raise CacheFormatError(f"{label} repeats timestamp {timestamp}")
        seen_timestamps.add(timestamp)
        parsed.append((timestamp, score))
    parsed.sort(key=lambda item: item[0])

    before_start = [item for item in parsed if item[0] < start_at]
    if before_start:
        raise CacheFormatError(f"{label} contains tracker rows before event start")

    in_event = [item for item in parsed if start_at <= item[0] <= end_at]
    if len(in_event) < 2:
        raise CacheFormatError(
            f"{label} needs at least two observations inside the event window"
        )

    bucket_end = end_at + final_bucket_max_delay_ms
    post_end = [item for item in parsed if item[0] > end_at]

    model_rows = list(in_event)
    mapped_terminal_delay_ms: int | None = None
    mapped_terminal_score: float | None = None
    if post_end and post_end[0][0] <= bucket_end:
        # The first post-end bucket is the canonical terminal observation.
        # Later tracker rows can themselves be broken revisions (including
        # large score regressions), so retain them only as diagnostics.
        terminal_time, terminal_score = post_end[0]
        mapped_terminal_delay_ms = terminal_time - end_at
        mapped_terminal_score = terminal_score
        model_rows = [item for item in model_rows if item[0] < end_at]
        model_rows.append((end_at, terminal_score))

    followups = post_end[1:] if mapped_terminal_score is not None else post_end
    disagreements = [
        {
            "time": int(timestamp),
            "delay_ms": int(timestamp - end_at),
            "score": float(score),
        }
        for timestamp, score in followups
        if mapped_terminal_score is None
        or not np.isclose(score, mapped_terminal_score, rtol=0.0, atol=0.0)
    ]

    timestamps = np.asarray([item[0] for item in model_rows], dtype=np.int64)
    raw_scores = np.asarray([item[1] for item in model_rows], dtype=float)
    negative_steps = int(np.sum(np.diff(raw_scores) < 0))
    scores = np.maximum.accumulate(raw_scores)

    hours = (timestamps.astype(float) - float(start_at)) / MS_PER_HOUR
    if hours[0] > 0.0:
        hours = np.concatenate(([0.0], hours))
        scores = np.concatenate(([0.0], scores))
    elif hours[0] == 0.0 and scores[0] != 0.0:
        raise CacheFormatError(
            f"{label} has a non-zero score exactly at the event start"
        )

    observations = np.column_stack((hours, scores)).astype(float, copy=False)
    observations.setflags(write=False)
    diagnostic = {
        "raw_row_count": len(parsed),
        "in_event_row_count": len(in_event),
        "model_row_count": len(observations),
        "post_end_rows_observed": len(post_end),
        "post_end_rows_mapped": int(mapped_terminal_score is not None),
        "terminal_mapping_delay_ms": mapped_terminal_delay_ms,
        "terminal_mapping_score": mapped_terminal_score,
        "post_end_followup_count": len(followups),
        "post_end_disagreement_count": len(disagreements),
        "post_end_disagreement_values": disagreements,
        "non_monotonic_steps_repaired": negative_steps,
        "first_model_hour": float(observations[0, 0]),
        "last_model_hour": float(observations[-1, 0]),
        "last_model_score": float(observations[-1, 1]),
    }
    return observations, diagnostic


def decode_training_event(
    raw: bytes,
    *,
    expected_event_id: int,
    final_bucket_max_delay_ms: int = DEFAULT_FINAL_BUCKET_MAX_DELAY_MS,
    label: str = "training cache",
) -> ParsedPaceTrainingEvent:
    """Decode one cache without assuming synchronized tier timestamps."""

    event_id = _strict_positive_int(expected_event_id, "expected_event_id")
    if (
        isinstance(final_bucket_max_delay_ms, bool)
        or not isinstance(final_bucket_max_delay_ms, numbers.Integral)
        or int(final_bucket_max_delay_ms) < 0
    ):
        raise ValueError("final_bucket_max_delay_ms must be a non-negative integer")
    maximum_delay = int(final_bucket_max_delay_ms)

    payload = _strict_json_loads(raw, label)
    if not isinstance(payload, Mapping):
        raise CacheFormatError(f"{label} root must be an object")
    actual_event_id = _strict_positive_int(payload.get("event_id"), "event_id")
    if actual_event_id != event_id:
        raise CacheFormatError(
            f"cache event_id={actual_event_id} does not match requested {event_id}"
        )

    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise CacheFormatError(f"event {event_id} meta must be an object")
    start_at = _strict_timestamp(meta.get("start_at"), "meta.start_at")
    end_at = _strict_timestamp(meta.get("end_at"), "meta.end_at")
    if end_at <= start_at:
        raise CacheFormatError("meta.end_at must be later than meta.start_at")
    if "event_id" in meta:
        meta_event_id = _strict_positive_int(meta["event_id"], "meta.event_id")
        if meta_event_id != event_id:
            raise CacheFormatError("meta.event_id disagrees with cache event_id")

    if "reward_tiers" not in payload:
        raise CacheFormatError(f"event {event_id} is missing reward_tiers")
    reward_tiers = _canonical_reward_tiers(
        payload["reward_tiers"], f"event {event_id} reward_tiers"
    )
    reward_provenance = _validated_reward_provenance(
        payload.get("reward_tier_provenance"),
        event_id=event_id,
        reward_tiers=reward_tiers,
    )

    tier_records = payload.get("tier_records")
    if not isinstance(tier_records, Mapping) or not tier_records:
        raise CacheFormatError(f"event {event_id} tier_records must be a non-empty object")

    observations: dict[int, np.ndarray] = {}
    tier_quality: dict[str, Any] = {}
    for raw_tier, records in tier_records.items():
        tier = _canonical_tier_key(raw_tier, f"event {event_id} tier_records key")
        series, diagnostic = _parse_tier_observations(
            records,
            event_id=event_id,
            tier=tier,
            start_at=start_at,
            end_at=end_at,
            final_bucket_max_delay_ms=maximum_delay,
        )
        observations[tier] = series
        tier_quality[str(tier)] = diagnostic

    missing_training_tiers = [
        tier for tier in TRAINING_TIERS if tier not in observations
    ]
    if missing_training_tiers:
        raise CacheFormatError(
            f"event {event_id} is missing fixed training tiers: "
            f"{missing_training_tiers}"
        )
    canonical_observations = {
        tier: observations[tier] for tier in TRAINING_TIERS
    }
    available_tiers = tuple(sorted(observations))
    quality = {
        "fixed_tiers": list(TRAINING_TIERS),
        "available_tiers": list(available_tiers),
        "ignored_extra_tiers": [
            tier for tier in available_tiers if tier not in TRAINING_TIERS
        ],
        "tier_count": len(canonical_observations),
        "reward_tiers": list(reward_tiers),
        "reward_tier_provenance": reward_provenance,
        "asynchronous_tier_timestamps_allowed": True,
        "rank_reference_tier": None,
        "tiers": tier_quality,
    }
    return ParsedPaceTrainingEvent(
        event_id=event_id,
        start_at=start_at,
        end_at=end_at,
        reward_tiers=reward_tiers,
        tier_observations=MappingProxyType(canonical_observations),
        data_quality=MappingProxyType(quality),
    )


def load_training_event(
    path: str | Path,
    *,
    expected_event_id: int,
    final_bucket_max_delay_ms: int = DEFAULT_FINAL_BUCKET_MAX_DELAY_MS,
) -> ParsedPaceTrainingEvent:
    """Read exactly ``path`` and decode it; no alternate filename is tried."""

    cache_path = Path(path)
    if cache_path.is_symlink():
        raise CacheFormatError("training cache paths must not be symbolic links")
    try:
        raw = cache_path.read_bytes()
    except OSError as exc:
        raise CacheFormatError(f"cannot read exact cache path {cache_path}") from exc
    return decode_training_event(
        raw,
        expected_event_id=expected_event_id,
        final_bucket_max_delay_ms=final_bucket_max_delay_ms,
        label=str(cache_path),
    )


def exact_event_files(
    cache_dir: str | Path,
    event_ids: Iterable[int] = DEFAULT_TRAINING_EVENT_IDS,
) -> dict[int, Path]:
    """Construct the one canonical ``<event_id>.json`` path per requested ID.

    This helper performs no directory listing and does not probe an
    ``event_<id>.json`` alternate.  Missing files remain missing and are later
    recorded as explicit exclusions by :func:`build_behavior_pace_prior`.
    """

    requested = validate_training_event_ids(event_ids)
    directory = Path(cache_dir)
    return {event_id: directory / f"{event_id}.json" for event_id in requested}


def _read_input_file(path: Path, event_id: int) -> _InputFile:
    if path.is_symlink():
        raise CacheFormatError("training cache paths must not be symbolic links")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CacheFormatError(f"cannot read exact cache path {path}") from exc
    return _InputFile(
        event_id=event_id,
        path=path,
        file_name=path.name,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        raw=raw,
    )


def _load_backend(backend: Any | None) -> Any:
    model = importlib.import_module("behavior_pace_model") if backend is None else backend
    required = (
        "PaceModelConfig",
        "PaceTrainingEvent",
        "fit_event_weights",
        "aggregate_event_weights",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise RuntimeError(f"pace-model backend lacks public API: {missing}")
    return model


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _model_source(backend: Any, model_source_path: str | Path | None) -> dict[str, Any]:
    if model_source_path is None:
        raw_path = getattr(backend, "__file__", None)
        if raw_path is None:
            raise ValueError("model_source_path is required for an injected backend")
        path = Path(raw_path)
    else:
        path = Path(model_source_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pace model source is not one exact regular file: {path}")
    raw = path.read_bytes()
    return {
        "module": getattr(backend, "__name__", type(backend).__name__),
        "file_name": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _builder_source() -> dict[str, Any]:
    """Fingerprint this builder's exact on-disk source bytes."""

    path = Path(__file__)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pace prior builder is not one exact regular file: {path}")
    raw = path.read_bytes()
    return {
        "module": path.stem,
        "file_name": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _construct_model_event(
    backend: Any,
    event: ParsedPaceTrainingEvent,
    config: Any,
) -> Any:
    constructor = backend.PaceTrainingEvent
    if not hasattr(config, "utc_offset_hours"):
        raise TypeError(
            "PaceModelConfig must expose utc_offset_hours; refusing to assume "
            "that every event starts at local midnight"
        )
    utc_offset = float(config.utc_offset_hours)
    if not math.isfinite(utc_offset):
        raise ValueError("PaceModelConfig.utc_offset_hours must be finite")
    candidates = {
        "event_id": event.event_id,
        "start_at": event.start_at,
        "end_at": event.end_at,
        "duration_hours": event.duration_hours,
        "event_start_local_hour": (
            event.start_at / MS_PER_HOUR + utc_offset
        ) % 24.0,
        "reward_tiers": event.reward_tiers,
        "tier_observations": event.tier_observations,
    }
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        return constructor(**candidates)

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = (
        candidates
        if accepts_kwargs
        else {name: value for name, value in candidates.items() if name in signature.parameters}
    )
    required_unknown = [
        name
        for name, parameter in signature.parameters.items()
        if name not in kwargs
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]
    if required_unknown:
        raise TypeError(
            "PaceTrainingEvent has unsupported required fields: "
            f"{required_unknown}; loader provides {sorted(candidates)}"
        )
    return constructor(**kwargs)


def _component_names(backend: Any, weights: Any) -> tuple[str, str, str]:
    if isinstance(weights, Mapping):
        names = tuple(str(name) for name in weights)
    elif is_dataclass(weights) and not isinstance(weights, type):
        names = tuple(str(name) for name in weights.__dataclass_fields__)
    else:
        raw = getattr(backend, "PACE_COMPONENT_NAMES", None)
        names = tuple(str(name) for name in raw) if raw is not None else (
            "component_0",
            "component_1",
            "component_2",
        )
    if len(names) != SIMPLEX_COMPONENT_COUNT or len(set(names)) != len(names):
        raise ValueError("pace model must expose exactly three unique components")
    return names  # type: ignore[return-value]


def _simplex_weights(
    value: Any,
    *,
    backend: Any,
    expected_names: tuple[str, str, str] | None = None,
) -> tuple[tuple[str, str, str], np.ndarray]:
    names = _component_names(backend, value)
    if expected_names is not None and names != expected_names:
        raise ValueError(
            f"pace component names changed: expected {expected_names}, got {names}"
        )
    if isinstance(value, Mapping):
        raw_values = list(value.values())
    elif hasattr(value, "as_array") and callable(value.as_array):
        raw_values = value.as_array()
    elif is_dataclass(value) and not isinstance(value, type):
        raw_values = [getattr(value, name) for name in names]
    else:
        raw_values = value
    vector = np.asarray(raw_values, dtype=float)
    if vector.shape != (SIMPLEX_COMPONENT_COUNT,):
        raise ValueError("pace weights must contain exactly three values")
    if np.any(~np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("pace weights must be finite and non-negative")
    if not np.isclose(float(np.sum(vector)), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("pace weights must lie on the simplex and sum to one")
    vector = vector.copy()
    vector.setflags(write=False)
    return names, vector


def _split_fit_result(result: Any) -> tuple[Any, Any]:
    if isinstance(result, Mapping) and "weights" in result:
        return result["weights"], result.get("diagnostics", {})
    if hasattr(result, "weights"):
        diagnostics = getattr(result, "diagnostics", None)
        if diagnostics is None and is_dataclass(result):
            diagnostics = {
                name: getattr(result, name)
                for name in result.__dataclass_fields__
                if name not in {"weights", "event_id"}
            }
        return result.weights, diagnostics or {}
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _split_aggregate_result(result: Any) -> tuple[Any, Any]:
    if isinstance(result, Mapping) and "weights" in result:
        return result["weights"], result.get("diagnostics", {})
    if hasattr(result, "weights"):
        return result.weights, getattr(result, "diagnostics", {})
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _weights_document(names: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return {str(name): float(value) for name, value in zip(names, values)}


def _generated_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("generated_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    return value


def build_behavior_pace_prior(
    event_files: Mapping[int, str | Path],
    *,
    event_ids: Iterable[int] = DEFAULT_TRAINING_EVENT_IDS,
    model_config: Any | None = None,
    backend: Any | None = None,
    model_source_path: str | Path | None = None,
    minimum_included_events: int = DEFAULT_MINIMUM_INCLUDED_EVENTS,
    minimum_included_fraction: float = DEFAULT_MINIMUM_INCLUDED_FRACTION,
    final_bucket_max_delay_ms: int = DEFAULT_FINAL_BUCKET_MAX_DELAY_MS,
    generated_at: str | None = None,
    enforce_coverage: bool = True,
) -> dict[str, Any]:
    """Fit one simplex vector per requested event and aggregate event-equally.

    Invalid or missing events are recorded in ``excluded_events``.  A failed
    coverage gate raises :class:`CoverageGateError` with the complete rejected
    document attached, so callers cannot accidentally publish it as a prior.
    """

    requested = validate_training_event_ids(event_ids)
    exact_files = _validate_event_file_mapping(event_files)
    if (
        isinstance(minimum_included_events, bool)
        or not isinstance(minimum_included_events, numbers.Integral)
        or int(minimum_included_events) < 1
    ):
        raise ValueError("minimum_included_events must be a positive integer")
    minimum_events = int(minimum_included_events)
    if (
        isinstance(minimum_included_fraction, bool)
        or not isinstance(minimum_included_fraction, numbers.Real)
        or not 0.0 < float(minimum_included_fraction) <= 1.0
    ):
        raise ValueError("minimum_included_fraction must be in (0, 1]")
    minimum_fraction = float(minimum_included_fraction)
    if not isinstance(enforce_coverage, bool):
        raise ValueError("enforce_coverage must be boolean")

    pace_backend = _load_backend(backend)
    config = pace_backend.PaceModelConfig() if model_config is None else model_config
    if not isinstance(config, pace_backend.PaceModelConfig):
        raise TypeError("model_config must be a PaceModelConfig instance")
    if not hasattr(config, "utc_offset_hours"):
        raise TypeError("PaceModelConfig must expose utc_offset_hours")
    if not math.isfinite(float(config.utc_offset_hours)):
        raise ValueError("PaceModelConfig.utc_offset_hours must be finite")
    source_model = _model_source(pace_backend, model_source_path)

    input_files: list[dict[str, Any]] = []
    event_fits: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    fitted_vectors: list[np.ndarray] = []
    names: tuple[str, str, str] | None = None

    for event_id in requested:
        path = exact_files.get(event_id)
        if path is None:
            excluded.append({
                "event_id": event_id,
                "reason": "missing_file_mapping",
                "detail": "no exact cache path was supplied for this requested event",
            })
            continue
        try:
            input_file = _read_input_file(path, event_id)
        except CacheFormatError as exc:
            excluded.append({
                "event_id": event_id,
                "file_name": path.name,
                "reason": "unreadable_input",
                "detail": str(exc),
            })
            continue

        input_descriptor = {
            "event_id": event_id,
            "file_name": input_file.file_name,
            "sha256": input_file.sha256,
            "size_bytes": input_file.size_bytes,
        }
        input_files.append(input_descriptor)
        try:
            parsed = decode_training_event(
                input_file.raw,
                expected_event_id=event_id,
                final_bucket_max_delay_ms=final_bucket_max_delay_ms,
                label=str(path),
            )
            model_event = _construct_model_event(pace_backend, parsed, config)
            fit_result = pace_backend.fit_event_weights(model_event, config)
            raw_weights, diagnostics = _split_fit_result(fit_result)
            event_names, weights = _simplex_weights(
                raw_weights,
                backend=pace_backend,
                expected_names=names,
            )
            if names is None:
                names = event_names
        except (CacheFormatError, TypeError, ValueError, RuntimeError) as exc:
            excluded.append({
                **input_descriptor,
                "reason": (
                    "invalid_cache" if isinstance(exc, CacheFormatError) else "fit_failed"
                ),
                "error_type": type(exc).__name__,
                "detail": str(exc),
            })
            continue

        fitted_vectors.append(weights)
        event_fits.append({
            **input_descriptor,
            "weights": _weights_document(event_names, weights),
            "diagnostics": {
                "loader": _jsonable(parsed.data_quality),
                "model": _jsonable(diagnostics),
            },
        })

    required_count = max(
        minimum_events,
        int(math.ceil(minimum_fraction * len(requested))),
    )
    included_count = len(event_fits)
    actual_fraction = included_count / len(requested)
    gate_passed = included_count >= required_count

    aggregate_weights: dict[str, float] | None = None
    aggregate_diagnostics: Any = None
    if fitted_vectors:
        aggregate_result = pace_backend.aggregate_event_weights(
            tuple(vector.copy() for vector in fitted_vectors)
        )
        raw_aggregate, aggregate_diagnostics = _split_aggregate_result(
            aggregate_result
        )
        if names is None:
            raise AssertionError("fitted vectors require component names")
        _, aggregate_vector = _simplex_weights(
            raw_aggregate,
            backend=pace_backend,
            expected_names=names,
        )
        aggregate_weights = _weights_document(names, aggregate_vector)

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _generated_at(generated_at),
        "training_event_ids": list(requested),
        "included_event_ids": [item["event_id"] for item in event_fits],
        "component_names": list(names or ()),
        "aggregate_weights": aggregate_weights,
        "aggregate_diagnostics": _jsonable(aggregate_diagnostics or {}),
        "event_fits": event_fits,
        "excluded_events": excluded,
        "input_files": input_files,
        "coverage_gate": {
            "minimum_included_events": minimum_events,
            "minimum_included_fraction": minimum_fraction,
            "requested_event_count": len(requested),
            "required_included_event_count": required_count,
            "actual_included_event_count": included_count,
            "actual_included_fraction": actual_fraction,
            "passed": gate_passed,
        },
        "algorithm_config": {
            "component_count": SIMPLEX_COMPONENT_COUNT,
            "event_weighting": "equal_one_simplex_vector_per_event",
            "tier_timestamp_alignment": "not_required",
            "training_tier_support": list(TRAINING_TIERS),
            "training_tier_support_semantics": (
                "fixed common measurement panel; every tier is symmetric and "
                "no tier is a scale/reference anchor"
            ),
            "rank_reference_tier": None,
            "start_boundary": "inject score zero at hour zero",
            "tracker_revision_rule": "causal cumulative maximum within each tier",
            "post_end_terminal_rule": (
                "map the first tracker row no later than end_at + "
                "final_bucket_max_delay_ms onto end_at"
            ),
            "final_bucket_max_delay_ms": int(final_bucket_max_delay_ms),
            "model_config": _jsonable(config),
        },
        "source_model": source_model,
        "source_builder": _builder_source(),
    }
    # Prove the result is strict JSON before returning it to a caller that may
    # persist the artifact elsewhere.
    json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True)

    if enforce_coverage and not gate_passed:
        raise CoverageGateError(
            "behavior pace prior coverage gate failed: "
            f"included={included_count}, requested={len(requested)}, "
            f"required={required_count}",
            document,
        )
    return document


def estimate_behavior_pace_prior(
    cache_dir: str | Path,
    *,
    event_ids: Iterable[int] = DEFAULT_TRAINING_EVENT_IDS,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a prior from canonical exact paths below one cache directory.

    This is the convenient training entrypoint.  It expands requested IDs to
    ``cache_dir/<event_id>.json`` without listing or probing the directory,
    then delegates to :func:`build_behavior_pace_prior`.
    """

    requested = validate_training_event_ids(event_ids)
    return build_behavior_pace_prior(
        exact_event_files(cache_dir, requested),
        event_ids=requested,
        **kwargs,
    )


def dumps_behavior_pace_prior(document: Mapping[str, Any]) -> str:
    """Serialize a returned document deterministically without writing a file."""

    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"document must use schema {SCHEMA_VERSION!r}")
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_behavior_pace_prior_atomic(
    path: str | Path,
    document: Mapping[str, Any],
) -> Path:
    """Atomically persist one already validated prior document.

    Only the exact destination's parent is used.  The temporary file is
    removed if replacement fails; this function never creates backup copies.
    """

    destination = Path(path)
    if destination.is_symlink():
        raise ValueError("prior destination must not be a symbolic link")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"prior destination parent does not exist: {destination.parent}"
        )
    payload = dumps_behavior_pace_prior(document).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the event-equal behavior-pace prior from exact cached events."
        )
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--event-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAINING_EVENT_IDS),
        help="explicit frozen training IDs (default: 192 through 283)",
    )
    parser.add_argument(
        "--minimum-included-events",
        type=int,
        default=DEFAULT_MINIMUM_INCLUDED_EVENTS,
    )
    parser.add_argument(
        "--minimum-included-fraction",
        type=float,
        default=DEFAULT_MINIMUM_INCLUDED_FRACTION,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = estimate_behavior_pace_prior(
        args.cache_dir,
        event_ids=args.event_ids,
        minimum_included_events=args.minimum_included_events,
        minimum_included_fraction=args.minimum_included_fraction,
    )
    destination = write_behavior_pace_prior_atomic(args.output, document)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "output": str(destination),
                "sha256": digest,
                "included_event_count": len(document["included_event_ids"]),
                "excluded_event_count": len(document["excluded_events"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
