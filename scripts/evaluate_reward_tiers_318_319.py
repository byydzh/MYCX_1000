"""Replay reward-tier forecasts for completed HHWX events.

The default scope remains events 318/319, but callers may select explicit event
IDs or an inclusive interval.  Every included event supplies its own canonical
reward tiers from the frozen cache:

* every target-tier model input and T10 scale sees only rows with
  ``time <= origin_at``;
* origins are 24, 48, ... hours after event start and strictly before event end;
* final truth is the first post-end bucket and is read only after every
  forecast for the event has returned (never the latest row);
* target reward tiers stay on frozen HHWX exact-tier records; missing historical
  exact tiers and T10 observations use the authorized HHWX-to-Bestdori route;
* an explicit empty canonical reward-tier list is recorded as not applicable;
  it is never replaced with a guessed target.

Skeleton+KF is reported with its operationally available same-rank and inferred
reward-behavior histories, plus a paired-intersection ablation.  The paired
methods use identical historical event IDs and differ only in which exact fixed
tier represents the reward behavior.  No method uses rank interpolation.

The script writes a reproducible JSON result and a dynamic static comparison plot.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from behavior_pace_model import (
    MS_PER_HOUR,
    PACE_COMPONENT_NAMES,
    PaceModelConfig,
    PaceWeights,
    canonicalize_reward_tiers,
    predict_tier_curve,
)
from config import DEFAULT_CONFIG
from data_source import create_data_source
from domain_models import EventData, EventMeta
from main_pipeline import PredictionPipeline
from math_models import CosineModeler, SeasonalityHandler
from prediction_engine import PredictionEngine


DEFAULT_EVENT_IDS = (318, 319)
DEFAULT_REWARD_TIERS = (500, 1500, 2000)
ORIGIN_STEP_HOURS = 24
FORECAST_STEP_HOURS = 1.0
HHWX_SERVER = 3
DEFAULT_CACHE_DIR = PROJECT_ROOT / "event_data" / "tier_surface_cache"
DEFAULT_OUTPUT_JSON = (
    PROJECT_ROOT / "event_data" / "reward_tier_evaluation_318_319.json"
)
DEFAULT_OUTPUT_PNG = (
    PROJECT_ROOT / "event_data" / "reward_tier_evaluation_318_319.png"
)
DEFAULT_PACE_PRIOR_PATH = (
    PROJECT_ROOT / "configs" / "behavior_model" / "pace_prior_train192_283.json"
)
FULL_COVERAGE_METHODS = (
    "behavior_pace_model",
    "persistence",
    "last_two_nonnegative_slope",
    "planned_duration_average_speed",
)
SKELETON_METHODS = (
    "skeleton_kf_same_rank_history",
    "skeleton_kf_reward_behavior_history",
    "skeleton_kf_same_rank_paired_intersection",
    "skeleton_kf_reward_behavior_paired_intersection",
)
METHODS = (FULL_COVERAGE_METHODS[0], *SKELETON_METHODS, *FULL_COVERAGE_METHODS[1:])
SKELETON_VARIANTS = {
    "skeleton_kf_same_rank_history": "same_rank",
    "skeleton_kf_reward_behavior_history": "reward_behavior",
    "skeleton_kf_same_rank_paired_intersection": (
        "same_rank_paired_intersection"
    ),
    "skeleton_kf_reward_behavior_paired_intersection": (
        "reward_behavior_paired_intersection"
    ),
}
EVALUATOR_SOURCE_NAME = "scripts/evaluate_reward_tiers_318_319.py"
PACE_MODEL_SOURCE_NAME = "behavior_pace_model.py"
PACE_BUILDER_SOURCE_NAME = "behavior_pace_prior.py"
SKELETON_PRESET_PATH = (
    PROJECT_ROOT / "configs" / "models" / "skeleton_kf" / "learned_notebook.json"
)
SKELETON_SOURCE_FILES = (
    "data_source.py",
    "prediction_engine.py",
    "math_models.py",
    "domain_models.py",
    "main_pipeline.py",
    "config.py",
    "base_distribution.py",
    "base_speed_distribution.json",
)


@dataclass(frozen=True)
class LoadedEvaluationEvent:
    event: "EvaluationEvent"
    raw_tier_records: Mapping[int, tuple[Mapping[str, Any], ...]]
    cache_path: Path
    cache_sha256: str
    reward_provenance: Mapping[str, Any]
    collection_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ExcludedEvaluationEvent:
    event_id: int
    status: str
    reason: str
    reward_tiers: tuple[int, ...]
    event_start_at: int
    event_end_at: int
    cache_path: Path
    cache_sha256: str
    reward_provenance: Mapping[str, Any]
    collection_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenHHWXCacheHeader:
    event_id: int
    start_at: int
    end_at: int
    event_type: str
    reward_tiers: tuple[int, ...]
    tracker_tiers: tuple[int, ...]
    path: Path
    sha256: str
    reward_provenance: Mapping[str, Any]
    collection_metadata: Mapping[str, Any]


@dataclass
class EvaluationEvent:
    """Minimal causal event container owned by this evaluator."""

    event_id: int
    start_at: int
    end_at: int
    frame: pd.DataFrame
    reward_tiers: tuple[int, ...]
    event_type: str = "unknown"
    availability_status: str = "hhwx_timestamp_masked_debug_replay"

    def __post_init__(self) -> None:
        self.event_id = _positive_int(self.event_id, "event_id")
        self.start_at = _integer_timestamp(self.start_at, "event.start_at")
        self.end_at = _integer_timestamp(self.end_at, "event.end_at")
        if self.end_at <= self.start_at:
            raise ValueError("event.end_at must be after event.start_at")
        self.reward_tiers = canonicalize_reward_tiers(
            self.reward_tiers,
            label=f"event {self.event_id} reward_tiers",
        )
        if not self.reward_tiers:
            raise ValueError("evaluated event reward_tiers must not be empty")
        if not isinstance(self.frame, pd.DataFrame) or self.frame.empty:
            raise ValueError(f"event {self.event_id} has no observations")
        required = {"time", "score", "tier"}
        if not required.issubset(self.frame.columns):
            raise ValueError(
                f"event {self.event_id} frame must contain time/score/tier"
            )
        cleaned = self.frame.loc[:, ["time", "score", "tier"]].copy(deep=True)
        cleaned["time"] = pd.to_numeric(cleaned["time"], errors="raise").astype(
            "int64"
        )
        cleaned["score"] = pd.to_numeric(
            cleaned["score"], errors="raise"
        ).astype(float)
        raw_tiers = pd.to_numeric(cleaned["tier"], errors="raise")
        if (
            not bool(np.isfinite(cleaned["score"]).all())
            or bool((cleaned["score"] < 0.0).any())
            or not bool(np.isfinite(raw_tiers).all())
            or not bool((raw_tiers == np.floor(raw_tiers)).all())
        ):
            raise ValueError(f"event {self.event_id} frame values are invalid")
        cleaned["tier"] = raw_tiers.astype(int)
        if bool((cleaned["tier"] <= 0).any()):
            raise ValueError("event tiers must be positive")
        self.frame = (
            cleaned.sort_values(["time", "tier"], kind="mergesort")
            .drop_duplicates(["time", "tier"], keep="last")
            .reset_index(drop=True)
        )

    @property
    def tiers(self) -> tuple[int, ...]:
        return tuple(int(value) for value in sorted(self.frame["tier"].unique()))


@dataclass(frozen=True)
class SkeletonPreset:
    path: Path
    sha256: str
    metadata: Mapping[str, Any]
    params: Mapping[str, Any]
    effective_config: Mapping[str, Any]


@dataclass(frozen=True)
class PacePrior:
    path: Path
    sha256: str
    schema_version: str
    training_event_ids: tuple[int, ...]
    weights: PaceWeights
    config: PaceModelConfig
    coverage_gate: Mapping[str, Any]
    source_model_sha256: str
    source_builder_sha256: str


class SkeletonUnavailableError(RuntimeError):
    """One Skeleton method has no legal exact-tier/scale support for a row."""


@dataclass(frozen=True)
class _NonFiniteJSONConstant:
    token: str


@dataclass
class PreparedSkeletonHistory:
    event_data: EventData
    mode: str
    source_tiers: tuple[int, ...]
    provenance: Mapping[str, Any]


@dataclass
class SkeletonHistorySelection:
    same_rank: tuple[PreparedSkeletonHistory, ...]
    reward_behavior: tuple[PreparedSkeletonHistory, ...]
    paired_same_rank: tuple[PreparedSkeletonHistory, ...]
    paired_reward_behavior: tuple[PreparedSkeletonHistory, ...]
    audit: Mapping[str, Any]


def _paired_history_intersection(
    same_rank: Sequence[PreparedSkeletonHistory],
    reward_behavior: Sequence[PreparedSkeletonHistory],
    *,
    count: int,
) -> tuple[
    tuple[PreparedSkeletonHistory, ...],
    tuple[PreparedSkeletonHistory, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Align the two history strategies on the same descending event IDs."""

    if int(count) <= 0:
        raise ValueError("paired history count must be positive")

    def by_event_id(
        histories: Sequence[PreparedSkeletonHistory],
        label: str,
    ) -> dict[int, PreparedSkeletonHistory]:
        indexed: dict[int, PreparedSkeletonHistory] = {}
        for history in histories:
            event_id = int(history.event_data.meta.event_id)
            if event_id in indexed:
                raise ValueError(f"{label} repeats history event {event_id}")
            indexed[event_id] = history
        return indexed

    same_by_id = by_event_id(same_rank, "same-rank history")
    reward_by_id = by_event_id(reward_behavior, "reward-behavior history")
    candidate_ids = tuple(sorted(set(same_by_id) & set(reward_by_id), reverse=True))
    selected_ids = candidate_ids[: int(count)]
    return (
        tuple(same_by_id[event_id] for event_id in selected_ids),
        tuple(reward_by_id[event_id] for event_id in selected_ids),
        candidate_ids,
        selected_ids,
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0 or numeric != np.floor(numeric):
        raise ValueError(f"{label} must be a positive integer")
    return int(numeric)


def _integer_timestamp(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer millisecond timestamp")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer millisecond timestamp") from exc
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{label} must be an integer millisecond timestamp")
    return int(numeric)


def resolve_event_ids(
    event_ids: Sequence[int] | None,
    min_event_id: int | None,
    max_event_id: int | None,
) -> tuple[int, ...]:
    """Resolve one canonical chronological event scope without guessing."""

    has_explicit = event_ids is not None
    has_interval = min_event_id is not None or max_event_id is not None
    if has_explicit and has_interval:
        raise ValueError("--event-ids cannot be combined with an event interval")
    if (min_event_id is None) != (max_event_id is None):
        raise ValueError("--min-event-id and --max-event-id must be provided together")
    if has_explicit:
        if not event_ids:
            raise ValueError("--event-ids cannot be empty")
        normalized = tuple(
            _positive_int(value, "event id") for value in event_ids
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"duplicate event IDs are not allowed: {list(normalized)}")
        return tuple(sorted(normalized))
    if has_interval:
        first = _positive_int(min_event_id, "min event id")
        last = _positive_int(max_event_id, "max event id")
        if first > last:
            raise ValueError("--min-event-id must be <= --max-event-id")
        return tuple(range(first, last + 1))
    return DEFAULT_EVENT_IDS


def _cache_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluator_source_sha256() -> str:
    path = PROJECT_ROOT / EVALUATOR_SOURCE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"required evaluator source does not exist: {path}")
    return _cache_sha256(path)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _parse_frozen_cache_json(raw: bytes, path: Path) -> Mapping[str, Any]:
    """Parse cache JSON while accepting only legacy ``isFinal: NaN`` markers."""

    def mark_constant(value: str) -> _NonFiniteJSONConstant:
        return _NonFiniteJSONConstant(str(value))

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=mark_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid frozen HHWX cache JSON: {path}") from exc

    def normalize(value: Any, *, key: str | None, location: str) -> Any:
        if isinstance(value, _NonFiniteJSONConstant):
            if key == "isFinal" and value.token == "NaN":
                return None
            raise ValueError(
                f"non-finite JSON constant {value.token!r} is forbidden at {location}"
            )
        if isinstance(value, Mapping):
            return {
                str(child_key): normalize(
                    child_value,
                    key=str(child_key),
                    location=f"{location}.{child_key}",
                )
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [
                normalize(child, key=key, location=f"{location}[{index}]")
                for index, child in enumerate(value)
            ]
        return value

    normalized = normalize(payload, key=None, location="$" )
    if not isinstance(normalized, Mapping):
        raise ValueError(f"{path} root must be an object")
    return normalized


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_pace_prior(path: str | Path) -> PacePrior:
    """Load the one frozen pace prior and bind it to its exact model source."""

    prior_path = Path(path)
    if not prior_path.is_file():
        raise FileNotFoundError(f"required behavior pace prior does not exist: {prior_path}")
    model_path = PROJECT_ROOT / PACE_MODEL_SOURCE_NAME
    if not model_path.is_file():
        raise FileNotFoundError(f"required behavior pace model does not exist: {model_path}")
    builder_path = PROJECT_ROOT / PACE_BUILDER_SOURCE_NAME
    if not builder_path.is_file():
        raise FileNotFoundError(
            f"required behavior pace prior builder does not exist: {builder_path}"
        )
    raw = prior_path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid behavior pace prior JSON: {prior_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("behavior pace prior root must be an object")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != "behavior-pace-prior-v1":
        raise ValueError(f"unsupported behavior pace prior schema: {schema_version!r}")
    if tuple(payload.get("component_names", ())) != tuple(PACE_COMPONENT_NAMES):
        raise ValueError("behavior pace prior component names do not match the model")
    raw_weights = payload.get("aggregate_weights")
    if not isinstance(raw_weights, Mapping) or set(raw_weights) != set(
        PACE_COMPONENT_NAMES
    ):
        raise ValueError("behavior pace prior aggregate_weights are incomplete")
    weights = PaceWeights(
        sustain=float(raw_weights["sustain"]),
        launch=float(raw_weights["launch"]),
        deadline=float(raw_weights["deadline"]),
    )
    raw_training_ids = payload.get("training_event_ids")
    if not isinstance(raw_training_ids, list) or not raw_training_ids:
        raise ValueError("behavior pace prior needs explicit training_event_ids")
    training_event_ids = tuple(
        _positive_int(value, "pace prior training event id")
        for value in raw_training_ids
    )
    if (
        training_event_ids != tuple(sorted(training_event_ids))
        or len(set(training_event_ids)) != len(training_event_ids)
    ):
        raise ValueError("behavior pace prior training_event_ids must be sorted and unique")
    coverage_gate = payload.get("coverage_gate")
    if not isinstance(coverage_gate, Mapping) or coverage_gate.get("passed") is not True:
        raise ValueError("behavior pace prior coverage gate did not pass")
    algorithm_config = payload.get("algorithm_config")
    if not isinstance(algorithm_config, Mapping):
        raise ValueError("behavior pace prior lacks algorithm_config")
    model_config = algorithm_config.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("behavior pace prior lacks model_config")
    try:
        config = PaceModelConfig(**dict(model_config))
    except (TypeError, ValueError) as exc:
        raise ValueError("behavior pace prior model_config is invalid") from exc
    source_model = payload.get("source_model")
    if not isinstance(source_model, Mapping):
        raise ValueError("behavior pace prior lacks source_model provenance")
    source_model_sha256 = str(source_model.get("sha256", "")).lower()
    current_model_sha256 = _cache_sha256(model_path)
    if source_model_sha256 != current_model_sha256:
        raise ValueError(
            "behavior pace prior was not trained with the current model source"
        )
    source_builder = payload.get("source_builder")
    if not isinstance(source_builder, Mapping):
        raise ValueError("behavior pace prior lacks source_builder provenance")
    source_builder_sha256 = str(source_builder.get("sha256", "")).lower()
    current_builder_sha256 = _cache_sha256(builder_path)
    if source_builder_sha256 != current_builder_sha256:
        raise ValueError(
            "behavior pace prior was not built with the current prior builder source"
        )
    return PacePrior(
        path=prior_path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=schema_version,
        training_event_ids=training_event_ids,
        weights=weights,
        config=config,
        coverage_gate=dict(coverage_gate),
        source_model_sha256=source_model_sha256,
        source_builder_sha256=source_builder_sha256,
    )


def _skeleton_source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in SKELETON_SOURCE_FILES:
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"required Skeleton+KF source does not exist: {path}")
        hashes[name] = _cache_sha256(path)
    return hashes


def load_skeleton_preset(path: str | Path = SKELETON_PRESET_PATH) -> SkeletonPreset:
    """Load the online default preset without ``load_preset``'s silent fallback."""

    preset_path = Path(path).resolve()
    if not preset_path.is_file():
        raise FileNotFoundError(f"required Skeleton+KF preset does not exist: {preset_path}")
    raw = preset_path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Skeleton+KF preset JSON: {preset_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Skeleton+KF preset root must be an object")
    metadata = payload.get("_meta")
    params = payload.get("params")
    if not isinstance(metadata, Mapping) or not isinstance(params, Mapping):
        raise ValueError("Skeleton+KF preset requires _meta and params objects")
    if metadata.get("model_id") != "skeleton_kf":
        raise ValueError("Skeleton+KF preset model_id must be skeleton_kf")
    effective = DEFAULT_CONFIG.copy()
    effective.update(dict(params))
    if str(effective.get("api_source", "")).lower() != "hhwx":
        raise ValueError("Skeleton+KF replay requires the HHWX preset source")
    if int(effective.get("similar_count", -1)) != 5:
        raise ValueError("online Skeleton+KF learned preset must use similar_count=5")
    ignored = tuple(sorted(int(value) for value in effective.get("ignore_event_ids", ())))
    if ignored != (297, 298):
        raise ValueError("online Skeleton+KF learned preset must ignore 297/298")
    return SkeletonPreset(
        path=preset_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        metadata=dict(metadata),
        params=dict(params),
        effective_config=effective,
    )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _tracker_frame_sha256(frame: pd.DataFrame) -> str:
    records = [
        {str(key): _json_scalar(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scale_from_t10_frame(frame: pd.DataFrame) -> float:
    """Use the production HHWX cutoff-scale formula on an already masked frame."""

    if frame is None or frame.empty or "time" not in frame:
        raise ValueError("HHWX T10 scale needs non-empty time-series data")
    score_column = next(
        (name for name in ("ep", "value", "score", "points", "pt") if name in frame),
        None,
    )
    if score_column is None:
        raise ValueError("HHWX T10 scale lacks a score column")
    work = frame[["time", score_column]].copy()
    work["time"] = pd.to_numeric(work["time"], errors="raise")
    work[score_column] = pd.to_numeric(work[score_column], errors="raise")
    work = work.sort_values("time", kind="mergesort")
    work["speed"] = work[score_column].diff() / (work["time"].diff() / 60_000.0)
    valid = work.loc[
        (work["speed"] > 0.0) & (work["speed"] < 1_000_000.0),
        "speed",
    ]
    if valid.empty:
        raise ValueError("HHWX T10 prefix has no valid positive scale observation")
    return float(np.mean(valid.nlargest(3).to_numpy(dtype=float)))


def _prepare_skeleton_event_data(
    *,
    event_id: int,
    event_type: str,
    start_at: int,
    end_at: int,
    frame: pd.DataFrame,
    scale: float,
    tier: int,
) -> tuple[EventData, int]:
    """Run the production derived-column routine, including start correction."""

    raw_start = int(start_at)
    event_data = EventData(
        meta=EventMeta(
            event_id=int(event_id),
            event_type=str(event_type),
            start_at=raw_start,
            end_at=int(end_at),
            aggregate_at=int(end_at),
        ),
        df=frame.copy(deep=True),
        scale=float(scale),
        tier=int(tier),
    )
    # This unbound call is intentional: it executes the exact production
    # preprocessing implementation without constructing another HTTP client.
    prepared = PredictionPipeline._calculate_derived_columns(None, event_data)
    return prepared, raw_start


def _clone_event_data(event: EventData) -> EventData:
    return EventData(
        meta=EventMeta(
            event_id=int(event.meta.event_id),
            event_type=str(event.meta.event_type),
            start_at=int(event.meta.start_at),
            end_at=int(event.meta.end_at),
            aggregate_at=int(event.meta.aggregate_at),
        ),
        df=event.df.copy(deep=True),
        scale=float(event.scale),
        tier=int(event.tier),
    )


def _reward_behavior_class(reward_key: str) -> str | None:
    """Infer the stable reward-behaviour class used by this diagnostic replay."""

    reward_type, separator, raw_id = str(reward_key).partition(":")
    if reward_type == "deco_pins":
        return "deco_pins"
    if reward_type != "voice_stamp" or not separator:
        return None
    try:
        reward_id = int(raw_id)
    except ValueError as exc:
        raise ValueError(f"voice_stamp rewardId is not an integer: {reward_key}") from exc
    # This threshold is an explicit inference from the HHWX master data for the
    # evaluated candidate window, not an upstream HHWX or game taxonomy.
    return "voice_stamp_premium" if reward_id >= 10_000 else "voice_stamp_standard"


def _reward_class_rank_map(
    last_appearance: Mapping[str, Any],
    *,
    event_id: int,
) -> dict[str, int]:
    by_class: dict[str, set[int]] = {}
    for reward_key, raw_rank in last_appearance.items():
        reward_class = _reward_behavior_class(str(reward_key))
        if reward_class is None:
            continue
        by_class.setdefault(reward_class, set()).add(int(raw_rank))
    ambiguous = {
        reward_class: sorted(ranks)
        for reward_class, ranks in by_class.items()
        if len(ranks) != 1
    }
    if ambiguous:
        raise ValueError(
            f"event {event_id} has ambiguous reward-class ranks: {ambiguous}"
        )
    return {
        reward_class: next(iter(ranks))
        for reward_class, ranks in sorted(by_class.items())
    }


def _validate_hhwx_provenance(
    payload: Mapping[str, Any],
    *,
    event_id: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    reward = payload.get("reward_tier_provenance")
    collection = payload.get("collection_metadata")
    if not isinstance(reward, Mapping):
        raise ValueError(f"event {event_id} cache lacks reward_tier_provenance")
    if not isinstance(collection, Mapping):
        raise ValueError(f"event {event_id} cache lacks collection_metadata")
    for label, provenance in (
        ("reward_tier_provenance", reward),
        ("collection_metadata", collection),
    ):
        if str(provenance.get("source", "")).strip().lower() != "hhwx":
            raise ValueError(f"event {event_id} {label}.source must be hhwx")
        if _positive_int(provenance.get("server"), f"{label}.server") != HHWX_SERVER:
            raise ValueError(
                f"event {event_id} {label}.server must be {HHWX_SERVER}"
            )
    failed = collection.get("failed_tiers")
    if failed not in (None, []):
        raise ValueError(
            f"event {event_id} HHWX collection has failed tiers: {failed!r}"
        )
    return dict(reward), dict(collection)


def _load_frozen_hhwx_header(path: Path, event_id: int) -> FrozenHHWXCacheHeader:
    raw = path.read_bytes()
    payload = _parse_frozen_cache_json(raw, path)
    if _positive_int(payload.get("event_id"), "event_id") != int(event_id):
        raise ValueError(f"frozen cache identity does not match {event_id}: {path}")
    reward_provenance, collection_metadata = _validate_hhwx_provenance(
        payload,
        event_id=int(event_id),
    )
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError(f"event {event_id} frozen cache lacks meta")
    start_at = _integer_timestamp(meta.get("start_at"), "meta.start_at")
    end_at = _integer_timestamp(meta.get("end_at"), "meta.end_at")
    if end_at <= start_at:
        raise ValueError(f"event {event_id} frozen end_at must be after start_at")
    reward_tiers = canonicalize_reward_tiers(
        payload.get("reward_tiers"),
        label=f"event {event_id} frozen reward_tiers",
    )
    tier_records = payload.get("tier_records")
    if not isinstance(tier_records, Mapping) or not tier_records:
        raise ValueError(f"event {event_id} frozen cache lacks tier_records")
    tracker_tiers = tuple(
        sorted(
            _positive_int(raw_tier, "frozen tier_records key")
            for raw_tier in tier_records
        )
    )
    if len(set(tracker_tiers)) != len(tracker_tiers):
        raise ValueError(f"event {event_id} frozen cache repeats a tracker tier")
    return FrozenHHWXCacheHeader(
        event_id=int(event_id),
        start_at=start_at,
        end_at=end_at,
        event_type=str(meta.get("event_type", "unknown")),
        reward_tiers=reward_tiers,
        tracker_tiers=tracker_tiers,
        path=path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        reward_provenance=reward_provenance,
        collection_metadata=collection_metadata,
    )


def _index_frozen_hhwx_caches(
    cache_dir: str | Path,
    *,
    maximum_event_id: int,
) -> dict[int, FrozenHHWXCacheHeader]:
    root = Path(cache_dir)
    headers: dict[int, FrozenHHWXCacheHeader] = {}
    for path in root.glob("*.json"):
        if not path.stem.isdecimal():
            continue
        event_id = int(path.stem)
        if event_id > int(maximum_event_id):
            continue
        if path.name != f"{event_id}.json":
            raise ValueError(f"non-canonical frozen cache filename: {path}")
        if event_id in headers:
            raise ValueError(f"duplicate frozen cache for event {event_id}")
        headers[event_id] = _load_frozen_hhwx_header(path, event_id)
    return dict(sorted(headers.items()))


def load_hhwx_event(
    cache_dir: str | Path,
    event_id: int,
    *,
    completed_as_of_ms: int | None = None,
) -> LoadedEvaluationEvent | ExcludedEvaluationEvent:
    """Load one completed HHWX cache without inferring a missing reward target."""

    event_id = _positive_int(event_id, "event id")
    cache_path = Path(cache_dir) / f"{event_id}.json"
    if not cache_path.is_file():
        raise FileNotFoundError(f"required HHWX cache does not exist: {cache_path}")
    try:
        raw_cache = cache_path.read_bytes()
        payload = json.loads(raw_cache.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid HHWX cache JSON: {cache_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{cache_path} root must be an object")
    if _positive_int(payload.get("event_id"), "event_id") != event_id:
        raise ValueError(f"cache identity does not match requested event {event_id}")

    reward_tiers = canonicalize_reward_tiers(
        payload.get("reward_tiers"),
        label=f"event {event_id} reward_tiers",
    )
    reward_provenance, collection_metadata = _validate_hhwx_provenance(
        payload,
        event_id=event_id,
    )

    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError(f"event {event_id} cache lacks meta")
    start_at = _integer_timestamp(meta.get("start_at"), "meta.start_at")
    end_at = _integer_timestamp(meta.get("end_at"), "meta.end_at")
    if end_at <= start_at:
        raise ValueError(f"event {event_id} end_at must be after start_at")
    if completed_as_of_ms is None:
        completed_as_of_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    completed_as_of_ms = _integer_timestamp(
        completed_as_of_ms,
        "completed_as_of_ms",
    )
    if end_at >= completed_as_of_ms:
        raise ValueError(
            f"event {event_id} is not completed as of {completed_as_of_ms}: "
            f"end_at={end_at}"
        )

    cache_sha256 = hashlib.sha256(raw_cache).hexdigest()
    if not reward_tiers:
        return ExcludedEvaluationEvent(
            event_id=event_id,
            status="not_applicable",
            reason="empty_canonical_reward_tiers",
            reward_tiers=(),
            event_start_at=start_at,
            event_end_at=end_at,
            cache_path=cache_path.resolve(),
            cache_sha256=cache_sha256,
            reward_provenance=reward_provenance,
            collection_metadata=collection_metadata,
        )

    raw_by_tier = payload.get("tier_records")
    if not isinstance(raw_by_tier, Mapping) or not raw_by_tier:
        raise ValueError(f"event {event_id} cache lacks tier_records")
    rows: list[dict[str, int | float]] = []
    normalized_raw: dict[int, tuple[Mapping[str, Any], ...]] = {}
    for raw_tier, raw_records in raw_by_tier.items():
        tier = _positive_int(raw_tier, "tier_records key")
        if tier in normalized_raw:
            raise ValueError(f"event {event_id} repeats tier {tier}")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(f"event {event_id} tier {tier} has no tracker rows")
        copied_records: list[Mapping[str, Any]] = []
        for row_number, record in enumerate(raw_records):
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"event {event_id} tier {tier} row {row_number} is not an object"
                )
            timestamp = _integer_timestamp(
                record.get("time"),
                f"event {event_id} tier {tier} row {row_number}.time",
            )
            if timestamp > end_at or record.get("isFinal") is True:
                # Every post-end bucket stays behind _actual_final_bucket().
                # Do not inspect its ``ep`` value while building causal input.
                copied_records.append(record)
                continue
            try:
                score = float(record["ep"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"event {event_id} tier {tier} row {row_number}.ep must be numeric"
                ) from exc
            if not np.isfinite(score) or score < 0:
                raise ValueError(
                    f"event {event_id} tier {tier} row {row_number}.ep is invalid"
                )
            rows.append({"time": timestamp, "score": score, "tier": tier})
            copied_records.append(dict(record))
        normalized_raw[tier] = tuple(copied_records)

    missing_rewards = sorted(set(reward_tiers) - set(normalized_raw))
    if missing_rewards:
        raise ValueError(
            f"event {event_id} cache lacks reward tracker tiers {missing_rewards}"
        )
    missing_post_end_truth = [
        tier
        for tier in reward_tiers
        if not any(
            _integer_timestamp(
                record.get("time"),
                f"event {event_id} tier {tier} sealed row time",
            )
            > end_at
            for record in normalized_raw[tier]
        )
    ]
    if missing_post_end_truth:
        raise ValueError(
            f"event {event_id} reward tiers lack a first post-end HHWX bucket: "
            f"{missing_post_end_truth}"
        )
    requested_tiers = collection_metadata.get("requested_tiers")
    if not isinstance(requested_tiers, list):
        raise ValueError(f"event {event_id} collection requested_tiers is missing")
    normalized_requested = canonicalize_reward_tiers(
        requested_tiers,
        label=f"event {event_id} collection requested_tiers",
    )
    missing_requested = sorted(set(normalized_requested) - set(normalized_raw))
    unrequested_rewards = sorted(set(reward_tiers) - set(normalized_requested))
    if missing_requested or unrequested_rewards:
        raise ValueError(
            f"event {event_id} HHWX collection contract is inconsistent: "
            f"missing_requested={missing_requested}, "
            f"unrequested_rewards={unrequested_rewards}"
        )

    event = EvaluationEvent(
        event_id=event_id,
        start_at=start_at,
        end_at=end_at,
        frame=pd.DataFrame(rows, columns=["time", "score", "tier"]),
        reward_tiers=reward_tiers,
        event_type=str(meta.get("event_type", "unknown")),
        availability_status="hhwx_timestamp_masked_debug_replay",
    )
    return LoadedEvaluationEvent(
        event=event,
        raw_tier_records=normalized_raw,
        cache_path=cache_path.resolve(),
        cache_sha256=cache_sha256,
        reward_provenance=reward_provenance,
        collection_metadata=collection_metadata,
    )


class SkeletonKFReplay:
    """Causal Skeleton+KF replay over frozen exact-tier cache records."""

    def __init__(
        self,
        preset: SkeletonPreset,
        loaded_events: Sequence[LoadedEvaluationEvent],
        *,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        if not loaded_events:
            raise ValueError("SkeletonKFReplay needs at least one evaluated event")
        self.preset = preset
        self.cache_headers = _index_frozen_hhwx_caches(
            cache_dir,
            maximum_event_id=max(
                int(loaded.event.event_id) for loaded in loaded_events
            ),
        )
        for loaded in loaded_events:
            event_id = int(loaded.event.event_id)
            header = self.cache_headers.get(event_id)
            if header is None:
                raise FileNotFoundError(
                    f"target event {event_id} is absent from frozen HHWX cache index"
                )
            if header.sha256 != loaded.cache_sha256:
                raise RuntimeError(
                    f"target event {event_id} cache changed between target and "
                    "history indexing"
                )
        self.data_source = create_data_source(
            "hhwx",
            server_index=HHWX_SERVER,
            allow_fallback=True,
        )
        if getattr(self.data_source, "api_source", None) != "hhwx":
            raise RuntimeError("Skeleton+KF replay did not receive an HHWX client")
        self._tracker_cache: dict[
            tuple[int, int], tuple[pd.DataFrame | None, Mapping[str, Any]]
        ] = {}
        self._scale_cache: dict[
            tuple[int, int | None], tuple[float | None, Mapping[str, Any]]
        ] = {}
        self.selections: dict[tuple[int, int], SkeletonHistorySelection] = {}
        self._build_selections(loaded_events)

        config = dict(self.preset.effective_config)
        self.seasonality = SeasonalityHandler(
            weekend_multiplier=float(config["weekend_multiplier"]),
            panic_scaler=float(config["panic_scaler"]),
            panic_ease_power=float(config["panic_ease_power"]),
        )
        self.engine = PredictionEngine(
            self.seasonality,
            CosineModeler(),
            config=config,
        )

    def close(self) -> None:
        close = getattr(self.data_source, "close", None)
        if callable(close):
            close()

    def _fetch_meta(self, event_id: int) -> Mapping[str, Any]:
        event_id = int(event_id)
        header = self.cache_headers.get(event_id)
        if header is None:
            raise FileNotFoundError(f"frozen HHWX cache is missing event {event_id}")
        return {
            "event_id": event_id,
            "start_at": int(header.start_at),
            "end_at": int(header.end_at),
            "event_type": str(header.event_type),
            "source": "hhwx",
            "server": HHWX_SERVER,
            "cache_path": str(header.path),
            "cache_sha256": header.sha256,
        }

    def _fetch_rewards(self, event_id: int) -> Mapping[str, Any]:
        event_id = int(event_id)
        header = self.cache_headers.get(event_id)
        if header is None:
            raise FileNotFoundError(f"frozen HHWX cache is missing event {event_id}")
        last = header.reward_provenance.get("last_appearance", {})
        if not isinstance(last, Mapping):
            raise ValueError(
                f"event {event_id} frozen reward last_appearance is invalid"
            )
        return {
            "source": "hhwx",
            "server": HHWX_SERVER,
            "target_tiers": [int(value) for value in header.reward_tiers],
            "last_appearance": {
                str(key): _positive_int(value, "reward last_appearance rank")
                for key, value in last.items()
            },
            "cache_path": str(header.path),
            "cache_sha256": header.sha256,
        }

    @staticmethod
    def _history_frame(
        frame: pd.DataFrame,
        *,
        event_id: int,
        tier: int,
        end_at: int,
    ) -> tuple[pd.DataFrame, Mapping[str, Any]]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"event {event_id} T{tier} tracker frame is empty")
        score_column = next(
            (name for name in ("ep", "value", "score", "points", "pt") if name in frame),
            None,
        )
        if "time" not in frame or score_column is None:
            raise ValueError(f"event {event_id} T{tier} tracker lacks time/score")
        work = frame.loc[:, ["time", score_column]].copy(deep=True)
        work.columns = ["time", "ep"]
        work["time"] = pd.to_numeric(work["time"], errors="raise").astype("int64")
        work["ep"] = pd.to_numeric(work["ep"], errors="raise").astype(float)
        if (
            not bool(np.isfinite(work["ep"]).all())
            or bool((work["ep"] < 0.0).any())
        ):
            raise ValueError(f"event {event_id} T{tier} tracker scores are invalid")
        work = (
            work.sort_values("time", kind="mergesort")
            .drop_duplicates("time", keep="last")
            .reset_index(drop=True)
        )
        post_end = work.loc[work["time"] > int(end_at)]
        terminal_source_time: int | None = None
        terminal_mapped = False
        in_event = work.loc[work["time"] <= int(end_at)].copy(deep=True)
        if not post_end.empty:
            terminal = post_end.iloc[0].copy()
            terminal_source_time = int(terminal["time"])
            terminal["time"] = int(end_at)
            in_event = pd.concat(
                [in_event.loc[in_event["time"] < int(end_at)], terminal.to_frame().T],
                ignore_index=True,
            )
            terminal_mapped = True
        in_event = (
            in_event.sort_values("time", kind="mergesort")
            .drop_duplicates("time", keep="last")
            .reset_index(drop=True)
        )
        if len(in_event) < 2:
            raise ValueError(
                f"event {event_id} T{tier} needs two complete-history tracker rows"
            )
        return in_event, {
            "raw_row_count": int(len(work)),
            "model_row_count": int(len(in_event)),
            "min_time": int(in_event["time"].min()),
            "max_time": int(in_event["time"].max()),
            "terminal_rule": "first_post_end_bucket_mapped_to_end",
            "terminal_mapped": bool(terminal_mapped),
            "terminal_source_time": terminal_source_time,
            "model_frame_sha256": _tracker_frame_sha256(in_event),
        }

    def _read_frozen_payload(
        self,
        header: FrozenHHWXCacheHeader,
    ) -> Mapping[str, Any]:
        raw = header.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != header.sha256:
            raise RuntimeError(f"frozen HHWX cache changed during replay: {header.path}")
        payload = _parse_frozen_cache_json(raw, header.path)
        if _positive_int(payload.get("event_id"), "event_id") != header.event_id:
            raise ValueError(f"frozen HHWX cache identity changed: {header.path}")
        return payload

    def _fetch_tracker(
        self,
        event_id: int,
        tier: int,
    ) -> tuple[pd.DataFrame | None, Mapping[str, Any]]:
        key = (int(event_id), int(tier))
        cached = self._tracker_cache.get(key)
        if cached is not None:
            return cached
        header = self.cache_headers.get(key[0])
        if header is None:
            raise FileNotFoundError(f"frozen HHWX cache is missing event {key[0]}")
        if key[1] in header.tracker_tiers:
            payload = self._read_frozen_payload(header)
            raw_by_tier = payload.get("tier_records")
            if not isinstance(raw_by_tier, Mapping):
                raise ValueError(f"event {key[0]} frozen tier_records is invalid")
            raw_records = raw_by_tier.get(str(key[1]))
            if not isinstance(raw_records, list) or not raw_records:
                raise ValueError(
                    f"event {key[0]} frozen exact T{key[1]} records are invalid"
                )
            frame, frame_audit = self._history_frame(
                pd.DataFrame(raw_records),
                event_id=key[0],
                tier=key[1],
                end_at=header.end_at,
            )
            provenance = {
                "source": "hhwx",
                "requested_source": "hhwx",
                "server": HHWX_SERVER,
                "fallback_used": False,
                "tier_interpolation_used": False,
                "event_id": key[0],
                "tier": key[1],
                "availability": "frozen_hhwx_cache_exact_fixed_tier",
                "cache_path": str(header.path),
                "cache_sha256": header.sha256,
                **dict(frame_audit),
            }
            result = (frame, provenance)
        else:
            try:
                routed = self.data_source.fetch_tier_data(
                    key[0],
                    tier=key[1],
                    allow_fallback=True,
                )
            except RuntimeError as exc:
                result = (
                    None,
                    {
                        "source": None,
                        "requested_source": "hhwx",
                        "server": HHWX_SERVER,
                        "fallback_used": None,
                        "tier_interpolation_used": False,
                        "event_id": key[0],
                        "tier": key[1],
                        "availability": "provider_route_exact_tier_unavailable",
                        "failure": str(exc),
                    },
                )
                self._tracker_cache[key] = result
                return result
            if routed is None or routed.empty:
                result = (
                    None,
                    {
                        "source": None,
                        "requested_source": "hhwx",
                        "server": HHWX_SERVER,
                        "fallback_used": None,
                        "tier_interpolation_used": False,
                        "event_id": key[0],
                        "tier": key[1],
                        "availability": "provider_route_exact_tier_unavailable",
                        "failure": "provider route returned no exact-tier rows",
                    },
                )
                self._tracker_cache[key] = result
                return result
            frame, frame_audit = self._history_frame(
                routed,
                event_id=key[0],
                tier=key[1],
                end_at=header.end_at,
            )
            provenance = {
                "source": str(routed.attrs.get("source")),
                "requested_source": str(
                    routed.attrs.get("requested_source", "hhwx")
                ),
                "server": HHWX_SERVER,
                "fallback_used": bool(routed.attrs.get("fallback_used", False)),
                "primary_error": routed.attrs.get("primary_error"),
                "fetched_at": routed.attrs.get("fetched_at"),
                "tier_interpolation_used": False,
                "event_id": key[0],
                "tier": key[1],
                "availability": "provider_route_exact_fixed_tier",
                **dict(frame_audit),
            }
            result = (frame, provenance)
        self._tracker_cache[key] = result
        return result

    def _fetch_routed_scale(
        self,
        event_id: int,
        *,
        origin_as_of: int | None,
    ) -> tuple[float | None, Mapping[str, Any]]:
        key = (int(event_id), None if origin_as_of is None else int(origin_as_of))
        cached = self._scale_cache.get(key)
        if cached is not None:
            return cached
        try:
            observation = self.data_source.fetch_top10_max_speed_observation(
                int(event_id),
                origin_as_of=origin_as_of,
                allow_fallback=True,
                primary_timeout=10,
                fallback_timeout=10,
                primary_retry=True,
                fallback_retry=True,
                suppress_fallback_log=True,
            )
        except RuntimeError as exc:
            result = (
                None,
                {
                    "value": None,
                    "source": None,
                    "requested_source": "hhwx",
                    "server": HHWX_SERVER,
                    "origin_as_of": origin_as_of,
                    "fallback_used": None,
                    "route": "hhwx_then_bestdori",
                    "tier_interpolation_used": False,
                    "availability": "provider_route_t10_unavailable",
                    "failure": str(exc),
                },
            )
            self._scale_cache[key] = result
            return result
        observation_fields = asdict(observation)
        observation_fields.pop("availability_status", None)
        provenance = {
            **observation_fields,
            "requested_source": "hhwx",
            "server": HHWX_SERVER,
            "route": "hhwx_then_bestdori",
            "tier_interpolation_used": False,
            "archive_availability_contract": "no_per_record_available_at",
            "availability": (
                "provider_route_t10_available"
                if observation.value is not None
                else "provider_route_t10_unavailable"
            ),
        }
        value = observation.value
        result = (
            None if value is None else float(value),
            provenance,
        )
        self._scale_cache[key] = result
        return result

    def _candidate_ids(
        self,
        target_event_id: int,
        event_type: str,
    ) -> tuple[int, ...]:
        ignored = {
            int(value)
            for value in self.preset.effective_config.get("ignore_event_ids", ())
        }
        candidates: list[int] = []
        for event_id, header in self.cache_headers.items():
            if event_id >= int(target_event_id) or event_id in ignored:
                continue
            if str(header.event_type).lower() == str(event_type).lower():
                candidates.append(event_id)
        candidates.sort(reverse=True)
        count = int(self.preset.effective_config["similar_count"])
        return tuple(candidates[: min(len(candidates), count + 3)])

    def _prepare_history(
        self,
        *,
        event_id: int,
        tier: int,
        frame: pd.DataFrame,
        scale: float,
        scale_provenance: Mapping[str, Any],
        mode: str,
        source_tiers: tuple[int, ...],
        target_start_at: int,
        tracker_provenance: Mapping[str, Any],
        source_provenance: Sequence[Mapping[str, Any]],
    ) -> PreparedSkeletonHistory:
        meta = self._fetch_meta(event_id)
        if int(meta["end_at"]) >= int(target_start_at):
            raise ValueError(
                f"history event {event_id} was not complete before target start"
            )
        if not np.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError(f"history event {event_id} scale must be positive")
        prepared, raw_start = _prepare_skeleton_event_data(
            event_id=event_id,
            event_type=str(meta["event_type"]),
            start_at=int(meta["start_at"]),
            end_at=int(meta["end_at"]),
            frame=frame,
            scale=scale,
            tier=tier,
        )
        provenance = {
            "event_id": int(event_id),
            "mode": str(mode),
            "source": "hhwx",
            "server": HHWX_SERVER,
            "source_tiers": [int(value) for value in source_tiers],
            "target_tier": int(tier),
            "raw_start_at": int(raw_start),
            "engine_corrected_start_at": int(prepared.meta.start_at),
            "end_at": int(meta["end_at"]),
            "historical_t10_scale": float(scale),
            "target_or_interpolated_tracker": dict(tracker_provenance),
            "source_tracker_inputs": [dict(value) for value in source_provenance],
            "historical_t10_scale_provenance": dict(scale_provenance),
            "reward_tier_provenance": dict(self._fetch_rewards(event_id)),
        }
        return PreparedSkeletonHistory(
            event_data=prepared,
            mode=str(mode),
            source_tiers=tuple(int(value) for value in source_tiers),
            provenance=provenance,
        )

    def _select_history(
        self,
        loaded: LoadedEvaluationEvent,
        tier: int,
    ) -> SkeletonHistorySelection:
        event = loaded.event
        count = int(self.preset.effective_config["similar_count"])
        candidate_ids = self._candidate_ids(event.event_id, event.event_type)
        same_rank_successes: list[PreparedSkeletonHistory] = []
        reward_behavior_successes: list[PreparedSkeletonHistory] = []
        candidate_audit: list[dict[str, Any]] = []

        target_last_appearance = loaded.reward_provenance.get("last_appearance")
        if not isinstance(target_last_appearance, Mapping):
            raise ValueError(
                f"event {event.event_id} lacks reward last_appearance provenance"
            )
        target_class_ranks = _reward_class_rank_map(
            target_last_appearance,
            event_id=event.event_id,
        )
        target_classes = [
            reward_class
            for reward_class, rank in target_class_ranks.items()
            if int(rank) == int(tier)
        ]
        target_reward_class = target_classes[0] if len(target_classes) == 1 else None
        target_reward_class_status = (
            "available"
            if target_reward_class is not None
            else "unavailable_not_exactly_one_reward_class"
        )

        for candidate_id in candidate_ids:
            meta = self._fetch_meta(candidate_id)
            if str(meta["event_type"]).lower() != str(event.event_type).lower():
                raise AssertionError("HHWX history type changed after candidate selection")
            if candidate_id >= event.event_id:
                raise AssertionError("future event ID entered history selection")
            rewards = self._fetch_rewards(candidate_id)
            reward_class_ranks = _reward_class_rank_map(
                rewards["last_appearance"],
                event_id=candidate_id,
            )
            audit_row: dict[str, Any] = {
                "event_id": int(candidate_id),
                "event_type": str(meta["event_type"]),
                "start_at": int(meta["start_at"]),
                "end_at": int(meta["end_at"]),
                "completed_before_target": int(meta["end_at"]) < int(event.start_at),
                "reward_tier_provenance": dict(rewards),
                "reward_class_rank_map": reward_class_ranks,
            }
            if int(meta["end_at"]) >= int(event.start_at):
                audit_row["same_rank_status"] = "excluded_not_completed_before_target"
                audit_row["reward_behavior_status"] = (
                    "excluded_not_completed_before_target"
                )
                candidate_audit.append(audit_row)
                continue

            historical_scale, scale_provenance = self._fetch_routed_scale(
                candidate_id,
                origin_as_of=None,
            )
            audit_row["historical_t10_scale"] = dict(scale_provenance)
            same_rank_frame, same_rank_provenance = self._fetch_tracker(
                candidate_id, tier
            )
            audit_row["same_rank_tracker"] = dict(same_rank_provenance)

            if same_rank_frame is None:
                audit_row["same_rank_status"] = "excluded_missing_same_rank_tracker"
            elif historical_scale is None:
                audit_row["same_rank_status"] = (
                    "excluded_provider_route_t10_unavailable"
                )
            else:
                try:
                    history = self._prepare_history(
                        event_id=candidate_id,
                        tier=tier,
                        frame=same_rank_frame,
                        scale=historical_scale,
                        scale_provenance=scale_provenance,
                        mode="same_rank",
                        source_tiers=(tier,),
                        target_start_at=event.start_at,
                        tracker_provenance=same_rank_provenance,
                        source_provenance=(same_rank_provenance,),
                    )
                except ValueError as exc:
                    audit_row["same_rank_status"] = (
                        "excluded_invalid_engine_history"
                    )
                    audit_row["same_rank_failure"] = str(exc)
                else:
                    same_rank_successes.append(history)
                    audit_row["same_rank_status"] = "selected_candidate"

            mapped_rank = (
                None
                if target_reward_class is None
                else reward_class_ranks.get(target_reward_class)
            )
            audit_row["target_reward_class"] = target_reward_class
            audit_row["reward_behavior_mapped_rank"] = mapped_rank
            if target_reward_class is None:
                audit_row["reward_behavior_status"] = (
                    "excluded_target_reward_class_unavailable"
                )
            elif mapped_rank is None:
                audit_row["reward_behavior_status"] = (
                    "excluded_reward_class_absent"
                )
            else:
                mapped_frame, mapped_provenance = self._fetch_tracker(
                    candidate_id,
                    int(mapped_rank),
                )
                audit_row["reward_behavior_tracker"] = dict(mapped_provenance)
                if mapped_frame is None:
                    audit_row["reward_behavior_status"] = (
                        "excluded_missing_mapped_rank_tracker"
                    )
                elif historical_scale is None:
                    audit_row["reward_behavior_status"] = (
                        "excluded_provider_route_t10_unavailable"
                    )
                else:
                    try:
                        history = self._prepare_history(
                            event_id=candidate_id,
                            tier=tier,
                            frame=mapped_frame,
                            scale=historical_scale,
                            scale_provenance=scale_provenance,
                            mode="reward_behavior_class",
                            source_tiers=(int(mapped_rank),),
                            target_start_at=event.start_at,
                            tracker_provenance=mapped_provenance,
                            source_provenance=(mapped_provenance,),
                        )
                    except ValueError as exc:
                        audit_row["reward_behavior_status"] = (
                            "excluded_invalid_engine_history"
                        )
                        audit_row["reward_behavior_failure"] = str(exc)
                    else:
                        reward_behavior_successes.append(history)
                        audit_row["reward_behavior_status"] = "selected_candidate"
            candidate_audit.append(audit_row)

        same_rank_successes.sort(
            key=lambda item: item.event_data.meta.event_id,
            reverse=True,
        )
        reward_behavior_successes.sort(
            key=lambda item: item.event_data.meta.event_id,
            reverse=True,
        )
        same_rank_selected = same_rank_successes[:count]
        reward_behavior_selected = reward_behavior_successes[:count]
        (
            paired_same_rank_selected,
            paired_reward_behavior_selected,
            paired_candidate_ids,
            paired_selected_ids,
        ) = _paired_history_intersection(
            same_rank_successes,
            reward_behavior_successes,
            count=count,
        )
        audit = {
            "provider": "hhwx",
            "server": HHWX_SERVER,
            "target_event_id": int(event.event_id),
            "target_event_type": str(event.event_type),
            "target_tier": int(tier),
            "selection_rule": (
                "production same-type event_id<target; ignore 297/298; "
                "descending IDs; scan_limit=similar_count+3; count=5"
            ),
            "candidate_window": [int(value) for value in candidate_ids],
            "candidate_audit": candidate_audit,
            "reward_behavior_taxonomy": {
                "status": "inferred_not_upstream_defined",
                "deco_pins": "rewardType == deco_pins",
                "voice_stamp_standard": (
                    "rewardType == voice_stamp and integer rewardId < 10000"
                ),
                "voice_stamp_premium": (
                    "rewardType == voice_stamp and integer rewardId >= 10000"
                ),
            },
            "target_reward_class": target_reward_class,
            "target_reward_class_status": target_reward_class_status,
            "target_reward_class_rank_map": target_class_ranks,
            "reward_behavior_rule": (
                "map the target reward class to each historical event's own "
                "HHWX rankingRewards[3] toRank; use that exact fixed-tier curve; "
                "no rank interpolation"
            ),
            "same_rank_selected": [
                dict(item.provenance) for item in same_rank_selected
            ],
            "reward_behavior_selected": [
                dict(item.provenance) for item in reward_behavior_selected
            ],
            "paired_intersection": {
                "semantics": (
                    "same historical event IDs, target prefix, T10 scale, engine, "
                    "preset, and history count; only the exact fixed historical "
                    "tier mapping differs"
                ),
                "candidate_event_ids": [int(value) for value in paired_candidate_ids],
                "selected_event_ids": [int(value) for value in paired_selected_ids],
                "same_rank_selected": [
                    dict(item.provenance) for item in paired_same_rank_selected
                ],
                "reward_behavior_selected": [
                    dict(item.provenance) for item in paired_reward_behavior_selected
                ],
            },
            "method_availability": {
                "skeleton_kf_same_rank_history": (
                    "available" if same_rank_selected else "unavailable_no_legal_history"
                ),
                "skeleton_kf_reward_behavior_history": (
                    "available"
                    if reward_behavior_selected
                    else "unavailable_no_legal_history"
                ),
                "skeleton_kf_same_rank_paired_intersection": (
                    "available"
                    if paired_same_rank_selected
                    else "unavailable_no_common_legal_history"
                ),
                "skeleton_kf_reward_behavior_paired_intersection": (
                    "available"
                    if paired_reward_behavior_selected
                    else "unavailable_no_common_legal_history"
                ),
            },
        }
        return SkeletonHistorySelection(
            same_rank=tuple(same_rank_selected),
            reward_behavior=tuple(reward_behavior_selected),
            paired_same_rank=tuple(paired_same_rank_selected),
            paired_reward_behavior=tuple(paired_reward_behavior_selected),
            audit=audit,
        )

    def _build_selections(
        self,
        loaded_events: Sequence[LoadedEvaluationEvent],
    ) -> None:
        for loaded in loaded_events:
            for tier in loaded.event.reward_tiers:
                key = (int(loaded.event.event_id), int(tier))
                self.selections[key] = self._select_history(loaded, tier)

    def _target_scale(
        self,
        event: EvaluationEvent,
        prefix: pd.DataFrame,
        *,
        origin_at: int,
    ) -> tuple[float, Mapping[str, Any]]:
        header = self.cache_headers.get(int(event.event_id))
        has_frozen_t10 = header is not None and 10 in header.tracker_tiers
        if has_frozen_t10:
            t10_prefix = prefix.loc[
                prefix["tier"] == 10,
                ["time", "score"],
            ].copy(deep=True)
            if bool((t10_prefix["time"] > int(origin_at)).any()):
                raise AssertionError("future frozen T10 row entered target scale")
            if len(t10_prefix) < 2:
                raise SkeletonUnavailableError(
                    "target_frozen_hhwx_t10_has_fewer_than_two_visible_rows"
                )
            try:
                scale = _scale_from_t10_frame(t10_prefix)
            except ValueError as exc:
                raise SkeletonUnavailableError(
                    f"target_frozen_hhwx_t10_scale_invalid: {exc}"
                ) from exc
            return float(scale), {
                "source": "hhwx",
                "requested_source": "hhwx",
                "server": HHWX_SERVER,
                "fallback_used": False,
                "route": "frozen_hhwx_exact_t10",
                "tier_interpolation_used": False,
                "availability": "frozen_hhwx_t10_origin_prefix_available",
                "origin_as_of": int(origin_at),
                "visible_row_count": int(len(t10_prefix)),
                "prefix_max_time": int(t10_prefix["time"].max()),
                "cache_path": str(header.path),
                "cache_sha256": header.sha256,
                "prefix_frame_sha256": _tracker_frame_sha256(t10_prefix),
            }
        scale, provenance = self._fetch_routed_scale(
            int(event.event_id),
            origin_as_of=int(origin_at),
        )
        if scale is None:
            raise SkeletonUnavailableError(
                "target_provider_route_t10_unavailable: "
                + str(provenance.get("failure") or provenance.get("fallback_error"))
            )
        if int(provenance.get("origin_as_of")) != int(origin_at):
            raise AssertionError("routed target T10 scale was not masked at origin")
        return float(scale), dict(provenance)

    def predict(
        self,
        event: EvaluationEvent,
        prefix: pd.DataFrame,
        *,
        origin_at: int,
        tier: int,
        variant: str,
    ) -> tuple[float, Mapping[str, Any]]:
        if bool((prefix["time"] > int(origin_at)).any()):
            raise AssertionError("Skeleton+KF prefix contains rows after origin")
        selection = self.selections[(int(event.event_id), int(tier))]
        if variant == "same_rank":
            selected = selection.same_rank
        elif variant == "reward_behavior":
            selected = selection.reward_behavior
        elif variant == "same_rank_paired_intersection":
            selected = selection.paired_same_rank
        elif variant == "reward_behavior_paired_intersection":
            selected = selection.paired_reward_behavior
        else:
            raise ValueError(f"unknown Skeleton+KF history variant: {variant}")
        if not selected:
            method = next(
                method_name
                for method_name, method_variant in SKELETON_VARIANTS.items()
                if method_variant == variant
            )
            reason = selection.audit.get("method_availability", {}).get(
                method,
                "unavailable_no_legal_history",
            )
            raise SkeletonUnavailableError(str(reason))

        target_scale, target_scale_provenance = self._target_scale(
            event,
            prefix,
            origin_at=int(origin_at),
        )
        target_frame = prefix.loc[
            prefix["tier"] == int(tier), ["time", "score"]
        ].copy()
        if len(target_frame) < 2:
            raise ValueError(
                f"event {event.event_id} T{tier} origin {origin_at} lacks two rows"
            )
        target, raw_start = _prepare_skeleton_event_data(
            event_id=event.event_id,
            event_type=event.event_type,
            start_at=event.start_at,
            end_at=event.end_at,
            frame=target_frame,
            scale=target_scale,
            tier=tier,
        )
        corrected_start = int(target.meta.start_at)
        debug_hours = (int(origin_at) - corrected_start) / MS_PER_HOUR
        if debug_hours < 0:
            raise ValueError("engine-corrected start is after the absolute origin")
        histories = [_clone_event_data(item.event_data) for item in selected]
        result = self.engine.predict(
            target,
            histories,
            debug_hours=float(debug_hours),
        )
        return float(result.final_score), {
            "variant": str(variant),
            "provider": target_scale_provenance.get("source"),
            "server": HHWX_SERVER,
            "fallback_used": target_scale_provenance.get("fallback_used"),
            "tier_interpolation_used": False,
            "history_event_ids": [
                int(item.event_data.meta.event_id) for item in selected
            ],
            "history_modes": [str(item.mode) for item in selected],
            "history_source_tiers": [
                [int(value) for value in item.source_tiers] for item in selected
            ],
            "target_scale": float(target_scale),
            "target_scale_provenance": dict(target_scale_provenance),
            "target_tier_prefix_max_time": int(target_frame["time"].max()),
            "absolute_origin_at": int(origin_at),
            "raw_start_at": int(raw_start),
            "engine_corrected_start_at": corrected_start,
            "engine_debug_hours": float(debug_hours),
            "engine_end_at": int(target.meta.end_at),
            "ratio": float(result.ratio),
            "kalman_scale_factor": float(result.scale_factor),
            "used_params": [float(value) for value in result.used_params],
        }

    def provenance(self) -> list[Mapping[str, Any]]:
        return [
            dict(self.selections[key].audit)
            for key in sorted(self.selections)
        ]


def origin_hours(event: EvaluationEvent) -> tuple[int, ...]:
    """Return 24-hour origins whose exact timestamps are before the event end."""

    values: list[int] = []
    hour = ORIGIN_STEP_HOURS
    while event.start_at + hour * MS_PER_HOUR < event.end_at:
        values.append(hour)
        hour += ORIGIN_STEP_HOURS
    if not values:
        raise ValueError(f"event {event.event_id} has no 24-hour replay origin")
    return tuple(values)


def _visible_prefix(event: EvaluationEvent, origin_at: int) -> pd.DataFrame:
    """Materialize the causal prefix before constructing any model snapshot."""

    if origin_at < event.start_at or origin_at >= event.end_at:
        raise ValueError("origin_at must satisfy start_at <= origin_at < end_at")
    prefix = event.frame.loc[event.frame["time"] <= int(origin_at)].copy(deep=True)
    if prefix.empty:
        raise ValueError(
            f"event {event.event_id} has no tracker row at or before origin {origin_at}"
        )
    if bool((prefix["time"] > origin_at).any()):
        raise AssertionError("future tracker row entered the visible prefix")
    return prefix.reset_index(drop=True)


def _forecast_times(origin_at: int, end_at: int) -> np.ndarray:
    step_ms = int(round(FORECAST_STEP_HOURS * MS_PER_HOUR))
    values = np.arange(origin_at + step_ms, end_at, step_ms, dtype=np.int64)
    return np.append(values, np.int64(end_at))


def _target_prefix(prefix: pd.DataFrame, tier: int) -> pd.DataFrame:
    target = prefix.loc[prefix["tier"] == int(tier), ["time", "score"]].copy()
    target = (
        target.sort_values("time", kind="mergesort")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )
    if len(target) < 2:
        raise ValueError(f"tier {tier} needs at least two visible tracker rows")
    return target


def _baseline_predictions(
    event: EvaluationEvent,
    prefix: pd.DataFrame,
    tier: int,
) -> dict[str, float]:
    target = _target_prefix(prefix, tier)
    latest = target.iloc[-1]
    previous = target.iloc[-2]
    latest_time = int(latest["time"])
    latest_score = float(latest["score"])
    previous_time = int(previous["time"])
    previous_score = float(previous["score"])
    interval_hours = (latest_time - previous_time) / MS_PER_HOUR
    if interval_hours <= 0:
        raise ValueError(f"event {event.event_id} tier {tier} has duplicate last times")
    remaining_hours = (event.end_at - latest_time) / MS_PER_HOUR
    last_two_speed = max(0.0, (latest_score - previous_score) / interval_hours)
    elapsed_hours = (latest_time - event.start_at) / MS_PER_HOUR
    if elapsed_hours <= 0:
        raise ValueError(
            "planned-duration average baseline needs a tracker row after event start"
        )
    duration_hours = (event.end_at - event.start_at) / MS_PER_HOUR
    return {
        "persistence": latest_score,
        "last_two_nonnegative_slope": (
            latest_score + last_two_speed * max(remaining_hours, 0.0)
        ),
        "planned_duration_average_speed": (
            latest_score * duration_hours / elapsed_hours
        ),
    }


def _predict_behavior_pace(
    event: EvaluationEvent,
    prefix: pd.DataFrame,
    *,
    origin_at: int,
    tier: int,
    pace_prior: PacePrior,
) -> tuple[float, dict[str, Any]]:
    if bool((prefix["time"] > int(origin_at)).any()):
        raise AssertionError("behavior pace prefix contains rows after origin")
    target = _target_prefix(prefix, tier)
    forecast_times = np.concatenate(
        (
            np.asarray([int(origin_at)], dtype=np.int64),
            _forecast_times(origin_at, event.end_at),
        )
    )
    prediction = predict_tier_curve(
        target,
        int(tier),
        int(event.start_at),
        int(event.end_at),
        forecast_times,
        event.reward_tiers,
        pace_prior.weights,
        pace_prior.config,
    )
    diagnostics = dict(prediction.diagnostics)
    if diagnostics.get("fallback_used") is not False:
        raise RuntimeError("behavior pace model did not affirm fallback_used=False")
    if int(prediction.tier) != int(tier):
        raise AssertionError("behavior pace model returned a different tier")
    if int(prediction.forecast_times[-1]) != int(event.end_at):
        raise AssertionError("behavior pace forecast does not terminate at event end")
    final_score = float(prediction.scores[-1])
    if not np.isfinite(final_score) or final_score < 0.0:
        raise RuntimeError("behavior pace model returned an invalid final score")
    return final_score, {
        "model": PACE_MODEL_SOURCE_NAME,
        "prior_schema_version": pace_prior.schema_version,
        "prior_sha256": pace_prior.sha256,
        "training_event_min": int(min(pace_prior.training_event_ids)),
        "training_event_max": int(max(pace_prior.training_event_ids)),
        "training_event_count": len(pace_prior.training_event_ids),
        "weights": {
            name: float(value)
            for name, value in zip(
                PACE_COMPONENT_NAMES,
                pace_prior.weights.as_array(),
            )
        },
        "anchor_at": int(prediction.anchor_at),
        "anchor_score": float(prediction.anchor_score),
        "forecast_end_at": int(prediction.forecast_times[-1]),
        "diagnostics": diagnostics,
        "fallback_used": False,
        "provider_used": None,
        "tier_interpolation_used": False,
    }


def _actual_final_bucket(
    loaded: LoadedEvaluationEvent,
    tier: int,
) -> tuple[float, int]:
    """Read exactly the first post-end bucket, never the latest tracker row."""

    raw_records = loaded.raw_tier_records[tier]
    post_end_rows = [
        (
            _integer_timestamp(
                record.get("time"),
                f"event {loaded.event.event_id} tier {tier} truth time",
            ),
            record,
        )
        for record in raw_records
        if _integer_timestamp(
            record.get("time"),
            f"event {loaded.event.event_id} tier {tier} truth time",
        )
        > int(loaded.event.end_at)
    ]
    if not post_end_rows:
        raise ValueError(
            f"event {loaded.event.event_id} tier {tier} lacks a post-end HHWX bucket"
        )
    first_time = min(timestamp for timestamp, _record in post_end_rows)
    first_rows = [
        record for timestamp, record in post_end_rows if timestamp == first_time
    ]
    if len(first_rows) != 1:
        raise ValueError(
            f"event {loaded.event.event_id} tier {tier} has ambiguous first "
            f"post-end bucket at {first_time}"
        )
    try:
        score = float(first_rows[0]["ep"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("HHWX first post-end score is invalid") from exc
    if not np.isfinite(score) or score < 0:
        raise ValueError("HHWX first post-end score is invalid")
    return score, first_time


def _actual_final_score(loaded: LoadedEvaluationEvent, tier: int) -> float:
    """Compatibility helper for callers that only need first post-end score."""

    score, _timestamp = _actual_final_bucket(loaded, tier)
    return score


def _metrics(predicted: float, actual: float) -> dict[str, float]:
    error = float(predicted - actual)
    absolute_error = abs(error)
    if actual == 0:
        raise ValueError("percentage metrics are undefined for zero final truth")
    denominator = abs(predicted) + abs(actual)
    return {
        "error": error,
        "absolute_error": absolute_error,
        "signed_percent_error": 100.0 * error / abs(actual),
        "absolute_percent_error": 100.0 * absolute_error / abs(actual),
        "smape_percent": (
            0.0 if denominator == 0 else 200.0 * absolute_error / denominator
        ),
    }


def evaluate_event(
    loaded: LoadedEvaluationEvent,
    *,
    pace_prior: PacePrior,
    skeleton_replay: SkeletonKFReplay,
) -> list[dict[str, Any]]:
    event = loaded.event
    pending_rows: list[dict[str, Any]] = []
    for hour in origin_hours(event):
        origin_at = event.start_at + hour * MS_PER_HOUR
        if origin_at >= event.end_at:
            raise AssertionError("origin must be strictly before event end")
        prefix = event.frame.loc[
            event.frame["time"] <= int(origin_at)
        ].copy(deep=True).reset_index(drop=True)
        if bool((prefix["time"] > int(origin_at)).any()):
            raise AssertionError("future tracker row entered the scheduled prefix")
        for tier in event.reward_tiers:
            visible_target_rows = int((prefix["tier"] == int(tier)).sum())
            common = {
                "event_id": int(event.event_id),
                "tier": int(tier),
                "origin_hours": int(hour),
                "origin_at": int(origin_at),
                "event_end_at": int(event.end_at),
                "prefix_all_tier_rows": int(len(prefix)),
                "prefix_target_tier_rows": visible_target_rows,
                "prefix_max_time": (
                    None if prefix.empty else int(prefix["time"].max())
                ),
            }
            if visible_target_rows < 2:
                reason = "target_tier_has_fewer_than_two_visible_rows"
                pending_rows.append(
                    {
                        **common,
                        "scoring_status": "input_unavailable",
                        "input_status": {
                            "scheduled": True,
                            "evaluable": False,
                            "reason": reason,
                        },
                        "predictions": {method: None for method in METHODS},
                        "method_status": {
                            method: {
                                "attempted": False,
                                "success": False,
                                "status": "not_attempted_common_input_unavailable",
                                "failure_reason": reason,
                            }
                            for method in METHODS
                        },
                        "skeleton_kf": {
                            SKELETON_VARIANTS[method]: {
                                "status": "not_attempted_common_input_unavailable",
                                "reason": reason,
                            }
                            for method in SKELETON_METHODS
                        },
                        "behavior_pace_model": {
                            "status": "not_attempted_common_input_unavailable",
                            "reason": reason,
                        },
                    }
                )
                continue

            baselines = _baseline_predictions(event, prefix, tier)
            pace_prediction, pace_diagnostics = _predict_behavior_pace(
                event,
                prefix,
                origin_at=origin_at,
                tier=tier,
                pace_prior=pace_prior,
            )
            predictions: dict[str, float | None] = {
                "behavior_pace_model": float(pace_prediction),
                **{method: None for method in SKELETON_METHODS},
                **{method: float(value) for method, value in baselines.items()},
            }
            method_status: dict[str, Mapping[str, Any]] = {
                method: {
                    "attempted": True,
                    "success": True,
                    "status": "success",
                    "failure_reason": None,
                }
                for method in FULL_COVERAGE_METHODS
            }
            skeleton_diagnostics: dict[str, Mapping[str, Any]] = {}
            for method, variant in SKELETON_VARIANTS.items():
                try:
                    prediction, diagnostics = skeleton_replay.predict(
                        event,
                        prefix,
                        origin_at=origin_at,
                        tier=tier,
                        variant=variant,
                    )
                except SkeletonUnavailableError as exc:
                    reason = str(exc)
                    predictions[method] = None
                    method_status[method] = {
                        "attempted": True,
                        "success": False,
                        "status": "unavailable",
                        "failure_reason": reason,
                    }
                    skeleton_diagnostics[variant] = {
                        "status": "unavailable",
                        "reason": reason,
                        "variant": variant,
                    }
                else:
                    predictions[method] = float(prediction)
                    method_status[method] = {
                        "attempted": True,
                        "success": True,
                        "status": "success",
                        "failure_reason": None,
                    }
                    skeleton_diagnostics[variant] = {
                        "status": "success",
                        **dict(diagnostics),
                    }

            paired_same = skeleton_diagnostics[
                "same_rank_paired_intersection"
            ]
            paired_reward = skeleton_diagnostics[
                "reward_behavior_paired_intersection"
            ]
            paired_same_success = paired_same.get("status") == "success"
            paired_reward_success = paired_reward.get("status") == "success"
            if paired_same_success != paired_reward_success:
                raise AssertionError(
                    "paired Skeleton methods did not share availability"
                )
            if paired_same_success and (
                paired_same["history_event_ids"]
                != paired_reward["history_event_ids"]
            ):
                raise AssertionError(
                    "paired Skeleton histories changed event IDs between strategies"
                )

            pending_rows.append(
                {
                    **common,
                    "scoring_status": "evaluable",
                    "input_status": {
                        "scheduled": True,
                        "evaluable": True,
                        "reason": None,
                    },
                    "predictions": {
                        method: predictions[method] for method in METHODS
                    },
                    "method_status": {
                        method: dict(method_status[method]) for method in METHODS
                    },
                    "skeleton_kf": {
                        "same_rank_history": dict(
                            skeleton_diagnostics["same_rank"]
                        ),
                        "reward_behavior_history": dict(
                            skeleton_diagnostics["reward_behavior"]
                        ),
                        "same_rank_paired_intersection": dict(
                            skeleton_diagnostics[
                                "same_rank_paired_intersection"
                            ]
                        ),
                        "reward_behavior_paired_intersection": dict(
                            skeleton_diagnostics[
                                "reward_behavior_paired_intersection"
                            ]
                        ),
                    },
                    "behavior_pace_model": pace_diagnostics,
                }
            )
    # Keep every final score sealed until every model call for this event has
    # returned.  This is stronger than merely reading each tier's truth after
    # its own prediction and rules out cross-tier/origin feedback.
    actual_by_tier = {
        tier: _actual_final_bucket(loaded, tier) for tier in event.reward_tiers
    }
    rows: list[dict[str, Any]] = []
    for pending in pending_rows:
        actual, actual_truth_at = actual_by_tier[int(pending["tier"])]
        predictions = pending["predictions"]
        rows.append(
            {
                **pending,
                "actual_final_score": float(actual),
                "actual_truth_at": int(actual_truth_at),
                "actual_truth_rule": "first_post_end_bucket",
                "metrics": {
                    method: (
                        None
                        if predictions[method] is None
                        else _metrics(float(predictions[method]), actual)
                    )
                    for method in METHODS
                },
            }
        )
    return rows


def _aggregate_method_rows(
    rows: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot aggregate an empty evaluation slice")
    metric_names = (
        "error",
        "absolute_error",
        "signed_percent_error",
        "absolute_percent_error",
        "smape_percent",
    )
    result: dict[str, float | int] = {"origin_count": len(rows)}
    for name in metric_names:
        values = np.asarray(
            [float(row["metrics"][method][name]) for row in rows],
            dtype=float,
        )
        result[f"mean_{name}"] = float(np.mean(values))
    errors = np.asarray(
        [float(row["metrics"][method]["error"]) for row in rows],
        dtype=float,
    )
    result["rmse"] = float(np.sqrt(np.mean(errors**2)))
    return result


def _empty_method_metrics() -> dict[str, None | int]:
    return {
        "origin_count": 0,
        "mean_error": None,
        "mean_absolute_error": None,
        "mean_signed_percent_error": None,
        "mean_absolute_percent_error": None,
        "mean_smape_percent": None,
        "rmse": None,
    }


def _mean_supported_metrics(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    count_name: str,
    source_rmse_name: str,
    result_rmse_name: str,
) -> dict[str, Any]:
    metric_names = (
        "mean_error",
        "mean_absolute_error",
        "mean_signed_percent_error",
        "mean_absolute_percent_error",
        "mean_smape_percent",
    )
    if not aggregates:
        return {
            count_name: 0,
            **{name: None for name in metric_names},
            result_rmse_name: None,
        }
    return {
        count_name: len(aggregates),
        **{
            name: float(np.mean([float(item[name]) for item in aggregates]))
            for name in metric_names
        },
        result_rmse_name: float(
            np.mean(
                [float(item[source_rmse_name]) for item in aggregates]
            )
        ),
    }


def _canonical_event_tiers(
    event_tiers: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    if not isinstance(event_tiers, Mapping):
        raise ValueError("event_tiers must map event IDs to canonical reward tiers")
    normalized: dict[int, tuple[int, ...]] = {}
    for raw_event_id, raw_tiers in event_tiers.items():
        event_id = _positive_int(raw_event_id, "event_tiers event id")
        if event_id in normalized:
            raise ValueError(f"event_tiers repeats normalized event {event_id}")
        tiers = canonicalize_reward_tiers(
            raw_tiers,
            label=f"event {event_id} aggregate reward tiers",
        )
        if not tiers:
            raise ValueError(
                f"evaluated event {event_id} cannot have empty reward tiers"
            )
        normalized[event_id] = tiers
    return dict(sorted(normalized.items()))


def _validate_common_scoring_rows(
    rows: Sequence[Mapping[str, Any]],
) -> set[tuple[int, int]]:
    expected_methods = set(METHODS)
    required_metrics = {
        "error",
        "absolute_error",
        "signed_percent_error",
        "absolute_percent_error",
        "smape_percent",
    }
    observed_cells: set[tuple[int, int]] = set()
    observed_origins: set[tuple[int, int, int]] = set()
    for row_number, row in enumerate(rows):
        event_id = _positive_int(row.get("event_id"), "row event_id")
        tier = _positive_int(row.get("tier"), "row tier")
        origin = _positive_int(row.get("origin_hours"), "row origin_hours")
        row_key = (event_id, tier, origin)
        if row_key in observed_origins:
            raise ValueError(f"duplicate event/tier/origin scoring row: {row_key}")
        observed_origins.add(row_key)
        observed_cells.add((event_id, tier))

        predictions = row.get("predictions")
        metrics = row.get("metrics")
        method_status = row.get("method_status")
        if not isinstance(predictions, Mapping) or set(predictions) != expected_methods:
            raise ValueError(
                f"row {row_number} predictions must contain exactly METHODS"
            )
        if not isinstance(metrics, Mapping) or set(metrics) != expected_methods:
            raise ValueError(f"row {row_number} metrics must contain exactly METHODS")
        if (
            not isinstance(method_status, Mapping)
            or set(method_status) != expected_methods
        ):
            raise ValueError(
                f"row {row_number} method_status must contain exactly METHODS"
            )
        scoring_status = row.get("scoring_status")
        if scoring_status not in {"evaluable", "input_unavailable"}:
            raise ValueError(f"row {row_number} has an invalid scoring_status")
        input_status = row.get("input_status")
        if not isinstance(input_status, Mapping):
            raise ValueError(f"row {row_number} lacks input_status")
        evaluable = scoring_status == "evaluable"
        if bool(input_status.get("evaluable")) != evaluable:
            raise ValueError(f"row {row_number} input/scoring status disagree")
        for method in METHODS:
            status = method_status[method]
            if not isinstance(status, Mapping):
                raise ValueError(f"row {row_number} {method} status is invalid")
            attempted = status.get("attempted") is True
            success = status.get("success") is True
            prediction_value = predictions[method]
            method_metrics = metrics[method]
            if not evaluable:
                if attempted or success:
                    raise ValueError(
                        f"row {row_number} {method} cannot run without common input"
                    )
                if prediction_value is not None or method_metrics is not None:
                    raise ValueError(
                        f"row {row_number} {method} input-unavailable result must be null"
                    )
                continue
            if not attempted:
                raise ValueError(
                    f"row {row_number} {method} must be attempted on evaluable input"
                )
            if method in FULL_COVERAGE_METHODS and not success:
                raise ValueError(
                    f"row {row_number} full-coverage method {method} failed"
                )
            if not success:
                if prediction_value is not None or method_metrics is not None:
                    raise ValueError(
                        f"row {row_number} failed {method} result must be null"
                    )
                if not str(status.get("failure_reason", "")).strip():
                    raise ValueError(
                        f"row {row_number} failed {method} lacks a reason"
                    )
                continue
            try:
                prediction = float(prediction_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"row {row_number} {method} prediction must be numeric"
                ) from exc
            if not np.isfinite(prediction):
                raise ValueError(
                    f"row {row_number} {method} prediction must be finite"
                )
            if (
                not isinstance(method_metrics, Mapping)
                or set(method_metrics) != required_metrics
            ):
                raise ValueError(
                    f"row {row_number} {method} metrics are incomplete"
                )
            values = np.asarray(
                [float(method_metrics[name]) for name in required_metrics],
                dtype=float,
            )
            if not bool(np.isfinite(values).all()):
                raise ValueError(f"row {row_number} {method} metrics must be finite")
    return observed_cells


def _failure_reason_counts(
    rows: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row["method_status"][method]
        if status.get("attempted") is not True or status.get("success") is True:
            continue
        reason = str(status.get("failure_reason") or "unspecified")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _cell_method_aggregate(
    rows: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    evaluable_rows = [row for row in rows if row["scoring_status"] == "evaluable"]
    attempted_rows = [
        row
        for row in evaluable_rows
        if row["method_status"][method].get("attempted") is True
    ]
    successful_rows = [
        row
        for row in attempted_rows
        if row["method_status"][method].get("success") is True
    ]
    metrics = (
        _aggregate_method_rows(successful_rows, method)
        if successful_rows
        else _empty_method_metrics()
    )
    evaluable_count = len(evaluable_rows)
    success_count = len(successful_rows)
    if evaluable_count == 0:
        support_status = "not_applicable_no_evaluable_origin"
    elif success_count == evaluable_count:
        support_status = "complete"
    elif success_count:
        support_status = "partial"
    else:
        support_status = "unavailable"
    return {
        "scheduled_origin_count": len(rows),
        "common_input_unavailable_origin_count": len(rows) - evaluable_count,
        "evaluable_origin_count": evaluable_count,
        "attempted_origin_count": len(attempted_rows),
        "success_origin_count": success_count,
        "failure_origin_count": len(attempted_rows) - success_count,
        "evaluable_origin_coverage_fraction": (
            None if evaluable_count == 0 else success_count / evaluable_count
        ),
        "scheduled_origin_coverage_fraction": (
            None if not rows else success_count / len(rows)
        ),
        "support_status": support_status,
        "failure_reasons": _failure_reason_counts(evaluable_rows, method),
        **metrics,
    }


def _group_method_aggregate(
    child_aggregates: Sequence[Mapping[str, Any]],
    *,
    child_count_name: str,
    supported_count_name: str,
    complete_count_name: str,
    source_rmse_name: str,
    result_rmse_name: str,
) -> dict[str, Any]:
    supported = [
        item for item in child_aggregates if int(item["success_origin_count"]) > 0
    ]
    complete = [item for item in child_aggregates if item["support_status"] == "complete"]
    metric_summary = _mean_supported_metrics(
        supported,
        count_name=supported_count_name,
        source_rmse_name=source_rmse_name,
        result_rmse_name=result_rmse_name,
    )
    evaluable_count = sum(int(item["evaluable_origin_count"]) for item in child_aggregates)
    success_count = sum(int(item["success_origin_count"]) for item in child_aggregates)
    if evaluable_count == 0:
        support_status = "not_applicable_no_evaluable_origin"
    elif success_count == evaluable_count:
        support_status = "complete"
    elif success_count:
        support_status = "partial"
    else:
        support_status = "unavailable"
    failure_reasons: dict[str, int] = {}
    for item in child_aggregates:
        for reason, count in item["failure_reasons"].items():
            failure_reasons[reason] = failure_reasons.get(reason, 0) + int(count)
    return {
        child_count_name: len(child_aggregates),
        complete_count_name: len(complete),
        "scheduled_origin_count": sum(
            int(item["scheduled_origin_count"]) for item in child_aggregates
        ),
        "common_input_unavailable_origin_count": sum(
            int(item["common_input_unavailable_origin_count"])
            for item in child_aggregates
        ),
        "evaluable_origin_count": evaluable_count,
        "attempted_origin_count": sum(
            int(item["attempted_origin_count"]) for item in child_aggregates
        ),
        "success_origin_count": success_count,
        "failure_origin_count": sum(
            int(item["failure_origin_count"]) for item in child_aggregates
        ),
        "evaluable_origin_coverage_fraction": (
            None if evaluable_count == 0 else success_count / evaluable_count
        ),
        "support_status": support_status,
        "failure_reasons": dict(sorted(failure_reasons.items())),
        **metric_summary,
    }


def _paired_pace_comparison(
    rows: Sequence[Mapping[str, Any]],
    skeleton_method: str,
) -> dict[str, Any]:
    support_rows = [
        row
        for row in rows
        if row["scoring_status"] == "evaluable"
        and row["method_status"][skeleton_method].get("success") is True
    ]
    support_cells = sorted(
        {(int(row["event_id"]), int(row["tier"])) for row in support_rows}
    )
    support_events = sorted({event_id for event_id, _tier in support_cells})

    def hierarchical(method: str) -> dict[str, Any] | None:
        if not support_rows:
            return None
        cells: list[dict[str, Any]] = []
        for event_id, tier in support_cells:
            selected = [
                row
                for row in support_rows
                if int(row["event_id"]) == event_id and int(row["tier"]) == tier
            ]
            cells.append(
                {
                    "event_id": event_id,
                    "tier": tier,
                    **_aggregate_method_rows(selected, method),
                }
            )
        events: list[dict[str, Any]] = []
        for event_id in support_events:
            selected_cells = [
                cell for cell in cells if int(cell["event_id"]) == event_id
            ]
            events.append(
                {
                    "event_id": event_id,
                    **_mean_supported_metrics(
                        selected_cells,
                        count_name="supported_reward_tier_count",
                        source_rmse_name="rmse",
                        result_rmse_name="mean_reward_tier_rmse",
                    ),
                }
            )
        return _mean_supported_metrics(
            events,
            count_name="supported_event_count",
            source_rmse_name="mean_reward_tier_rmse",
            result_rmse_name="mean_event_reward_tier_rmse",
        )

    pace = hierarchical("behavior_pace_model")
    skeleton = hierarchical(skeleton_method)
    delta = None
    if pace is not None and skeleton is not None:
        comparable_names = (
            "mean_error",
            "mean_absolute_error",
            "mean_signed_percent_error",
            "mean_absolute_percent_error",
            "mean_smape_percent",
            "mean_event_reward_tier_rmse",
        )
        delta = {
            name: float(skeleton[name]) - float(pace[name])
            for name in comparable_names
        }
    return {
        "support_semantics": (
            "identical event/tier/origin rows where the Skeleton method succeeded; "
            "origin mean, then equal reward tiers, then equal events"
        ),
        "paired_success_row_count": len(support_rows),
        "paired_cell_count": len(support_cells),
        "paired_event_count": len(support_events),
        "paired_event_ids": support_events,
        "paired_event_tiers": [
            {"event_id": event_id, "tier": tier}
            for event_id, tier in support_cells
        ],
        "behavior_pace_model": pace,
        skeleton_method: skeleton,
        "skeleton_minus_behavior_pace": delta,
    }


def aggregate_results(
    rows: Sequence[Mapping[str, Any]],
    event_tiers: Mapping[int, Sequence[int]],
) -> dict[str, Any]:
    normalized_event_tiers = _canonical_event_tiers(event_tiers)
    expected_cells = {
        (event_id, tier)
        for event_id, tiers in normalized_event_tiers.items()
        for tier in tiers
    }
    observed_cells = _validate_common_scoring_rows(rows)
    unexpected_cells = sorted(observed_cells - expected_cells)
    missing_cells = sorted(expected_cells - observed_cells)
    if unexpected_cells or missing_cells:
        raise ValueError(
            "evaluation scoring cells differ from canonical reward targets: "
            f"unexpected={unexpected_cells}, missing={missing_cells}"
        )
    if not normalized_event_tiers:
        return {
            "status": "not_applicable",
            "weighting_hierarchy": (
                "mean origins within event×tier; equal reward tiers within event; "
                "equal completed events overall"
            ),
            "per_event_tier": [],
            "by_event_reward_tier_equal": [],
            "by_tier_event_equal": [],
            "overall_event_equal": None,
            "input_coverage": None,
            "coverage_by_method": {},
            "paired_behavior_pace_vs_skeleton": {},
        }

    cells: list[dict[str, Any]] = []
    for event_id, tiers in normalized_event_tiers.items():
        event_origin_sets: list[set[int]] = []
        for tier in tiers:
            selected = [
                row
                for row in rows
                if int(row["event_id"]) == event_id and int(row["tier"]) == tier
            ]
            event_origin_sets.append(
                {int(row["origin_hours"]) for row in selected}
            )
            cells.append(
                {
                    "event_id": event_id,
                    "tier": tier,
                    "scheduled_origin_count": len(selected),
                    "evaluable_origin_count": sum(
                        row["scoring_status"] == "evaluable" for row in selected
                    ),
                    "input_unavailable_origin_count": sum(
                        row["scoring_status"] == "input_unavailable"
                        for row in selected
                    ),
                    "methods": {
                        method: _cell_method_aggregate(selected, method)
                        for method in METHODS
                    },
                }
            )
        if any(origins != event_origin_sets[0] for origins in event_origin_sets[1:]):
            raise ValueError(
                f"event {event_id} reward tiers do not share the same replay origins"
            )

    by_event = []
    for event_id in normalized_event_tiers:
        selected_cells = [cell for cell in cells if cell["event_id"] == event_id]
        by_event.append(
            {
                "event_id": event_id,
                "reward_tier_count": len(selected_cells),
                "weighting": "equal_across_canonical_reward_tiers_after_origin_mean",
                "methods": {
                    method: _group_method_aggregate(
                        [cell["methods"][method] for cell in selected_cells],
                        child_count_name="canonical_reward_tier_count",
                        supported_count_name="supported_reward_tier_count",
                        complete_count_name="complete_reward_tier_count",
                        source_rmse_name="rmse",
                        result_rmse_name="mean_reward_tier_rmse",
                    )
                    for method in METHODS
                },
            }
        )

    by_tier = []
    for tier in sorted({tier for tiers in normalized_event_tiers.values() for tier in tiers}):
        selected_cells = [cell for cell in cells if cell["tier"] == tier]
        by_tier.append(
            {
                "tier": tier,
                "event_count": len(selected_cells),
                "weighting": "equal_across_events_containing_this_reward_tier",
                "methods": {
                    method: _group_method_aggregate(
                        [cell["methods"][method] for cell in selected_cells],
                        child_count_name="canonical_event_count",
                        supported_count_name="supported_event_count",
                        complete_count_name="complete_event_count",
                        source_rmse_name="rmse",
                        result_rmse_name="mean_event_tier_rmse",
                    )
                    for method in METHODS
                },
            }
        )
    overall_methods: dict[str, Any] = {}
    for method in METHODS:
        event_methods = [event["methods"][method] for event in by_event]
        supported_events = [
            item for item in event_methods if int(item["success_origin_count"]) > 0
        ]
        metric_summary = _mean_supported_metrics(
            supported_events,
            count_name="supported_event_count",
            source_rmse_name="mean_reward_tier_rmse",
            result_rmse_name="mean_event_reward_tier_rmse",
        )
        evaluable_count = sum(
            int(item["evaluable_origin_count"]) for item in event_methods
        )
        success_count = sum(
            int(item["success_origin_count"]) for item in event_methods
        )
        overall_methods[method] = {
            "evaluated_event_count": len(event_methods),
            "complete_event_count": sum(
                item["support_status"] == "complete" for item in event_methods
            ),
            "scheduled_origin_count": sum(
                int(item["scheduled_origin_count"]) for item in event_methods
            ),
            "common_input_unavailable_origin_count": sum(
                int(item["common_input_unavailable_origin_count"])
                for item in event_methods
            ),
            "evaluable_origin_count": evaluable_count,
            "attempted_origin_count": sum(
                int(item["attempted_origin_count"]) for item in event_methods
            ),
            "success_origin_count": success_count,
            "failure_origin_count": sum(
                int(item["failure_origin_count"]) for item in event_methods
            ),
            "evaluable_origin_coverage_fraction": (
                None if evaluable_count == 0 else success_count / evaluable_count
            ),
            "support_status": (
                "not_applicable_no_evaluable_origin"
                if evaluable_count == 0
                else "complete"
                if success_count == evaluable_count
                else "partial"
                if success_count
                else "unavailable"
            ),
            **metric_summary,
        }

    scheduled_row_count = len(rows)
    evaluable_row_count = sum(row["scoring_status"] == "evaluable" for row in rows)
    coverage_by_method: dict[str, Any] = {}
    for method in METHODS:
        method_cells = [cell["methods"][method] for cell in cells]
        attempted = sum(int(item["attempted_origin_count"]) for item in method_cells)
        success = sum(int(item["success_origin_count"]) for item in method_cells)
        failure_reasons: dict[str, int] = {}
        for item in method_cells:
            for reason, count in item["failure_reasons"].items():
                failure_reasons[reason] = failure_reasons.get(reason, 0) + int(count)
        coverage_by_method[method] = {
            "scheduled_row_count": scheduled_row_count,
            "common_evaluable_row_count": evaluable_row_count,
            "common_input_unavailable_row_count": (
                scheduled_row_count - evaluable_row_count
            ),
            "attempted_row_count": attempted,
            "success_row_count": success,
            "failure_row_count": attempted - success,
            "evaluable_row_coverage_fraction": (
                None if evaluable_row_count == 0 else success / evaluable_row_count
            ),
            "scheduled_row_coverage_fraction": (
                None if scheduled_row_count == 0 else success / scheduled_row_count
            ),
            "scheduled_cell_count": len(cells),
            "cells_with_evaluable_origins": sum(
                int(item["evaluable_origin_count"]) > 0 for item in method_cells
            ),
            "cells_with_any_success": sum(
                int(item["success_origin_count"]) > 0 for item in method_cells
            ),
            "complete_evaluable_cell_count": sum(
                item["support_status"] == "complete" for item in method_cells
            ),
            "failure_reasons": dict(sorted(failure_reasons.items())),
        }
    return {
        "status": "complete",
        "weighting_hierarchy": (
            "mean successful evaluable origins within event×tier; equal supported "
            "reward tiers within event; equal supported completed events overall"
        ),
        "available_case_warning": (
            "Skeleton metrics use only explicitly successful support and must be read "
            "with coverage or paired behavior-pace comparisons"
        ),
        "per_event_tier": cells,
        "by_event_reward_tier_equal": by_event,
        "by_tier_event_equal": by_tier,
        "overall_event_equal": {
            "event_count": len(by_event),
            "weighting": (
                "equal_across_events_after_equal_reward_tier_and_origin_means"
            ),
            "methods": overall_methods,
        },
        "input_coverage": {
            "scheduled_row_count": scheduled_row_count,
            "evaluable_row_count": evaluable_row_count,
            "input_unavailable_row_count": scheduled_row_count - evaluable_row_count,
            "evaluable_fraction": (
                None
                if scheduled_row_count == 0
                else evaluable_row_count / scheduled_row_count
            ),
            "scheduled_cell_count": len(cells),
            "cells_with_evaluable_origins": sum(
                int(cell["evaluable_origin_count"]) > 0 for cell in cells
            ),
            "fully_evaluable_cell_count": sum(
                int(cell["evaluable_origin_count"])
                == int(cell["scheduled_origin_count"])
                for cell in cells
            ),
        },
        "coverage_by_method": coverage_by_method,
        "paired_behavior_pace_vs_skeleton": {
            method: _paired_pace_comparison(rows, method)
            for method in SKELETON_METHODS
        },
    }


def render_plot(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    event_tiers: Mapping[int, Sequence[int]],
) -> None:
    """Render every evaluated event×reward-tier cell without a fixed scope."""

    styles = {
        "behavior_pace_model": {
            "label": "Behavior pace model",
            "color": "#0891B2",
            "linestyle": "-",
            "marker": "*",
        },
        "skeleton_kf_same_rank_history": {
            "label": "Skeleton+KF (same-rank history)",
            "color": "#111827",
            "linestyle": "-",
            "marker": "D",
        },
        "skeleton_kf_reward_behavior_history": {
            "label": "Skeleton+KF (reward-behavior history)",
            "color": "#0F766E",
            "linestyle": "--",
            "marker": "P",
        },
        "skeleton_kf_same_rank_paired_intersection": {
            "label": "Skeleton+KF paired (same-rank)",
            "color": "#7C3AED",
            "linestyle": "-.",
            "marker": "X",
        },
        "skeleton_kf_reward_behavior_paired_intersection": {
            "label": "Skeleton+KF paired (reward-behavior)",
            "color": "#BE185D",
            "linestyle": ":",
            "marker": "v",
        },
        "persistence": {
            "label": "Persistence",
            "color": "#6B7280",
            "linestyle": ":",
            "marker": "x",
        },
        "last_two_nonnegative_slope": {
            "label": "Last-two slope (>=0)",
            "color": "#EA580C",
            "linestyle": "--",
            "marker": "^",
        },
        "planned_duration_average_speed": {
            "label": "Planned-duration avg speed",
            "color": "#CA8A04",
            "linestyle": "-.",
            "marker": "s",
        },
    }
    normalized_event_tiers = _canonical_event_tiers(event_tiers)
    aggregates = aggregate_results(rows, normalized_event_tiers)
    cells = [
        (event_id, tier)
        for event_id, tiers in normalized_event_tiers.items()
        for tier in tiers
    ]
    if not cells:
        fig, axis = plt.subplots(1, 1, figsize=(9, 3.5))
        axis.axis("off")
        axis.text(
            0.5,
            0.55,
            "No applicable canonical reward-tier events in the requested scope",
            ha="center",
            va="center",
            fontsize=13,
            color="#374151",
        )
        fig.suptitle(
            "HHWX completed-event reward-tier rolling replay",
            fontsize=14,
            weight="bold",
            color="#111827",
        )
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    if len(cells) > 6:
        fig, axis = plt.subplots(1, 1, figsize=(16.5, 7.4))
        by_event = {
            int(item["event_id"]): item["methods"]
            for item in aggregates["by_event_reward_tier_equal"]
        }
        event_ids = list(normalized_event_tiers)
        x_values = np.arange(len(event_ids), dtype=float)
        highlighted = (
            "behavior_pace_model",
            "skeleton_kf_same_rank_history",
            "planned_duration_average_speed",
        )
        for method in highlighted:
            style = styles[method]
            values = [
                by_event[event_id][method]["mean_absolute_percent_error"]
                for event_id in event_ids
            ]
            axis.plot(
                x_values,
                [np.nan if value is None else float(value) for value in values],
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=5.0,
            )
        axis.set_xticks(x_values)
        axis.set_xticklabels([str(value) for value in event_ids], rotation=60)
        axis.set_xlabel("Completed target event ID")
        axis.set_ylabel("Reward-tier-equal MAPE (%)")
        axis.set_title(
            "Broad rolling replay: per-event final-score error",
            fontsize=14,
            weight="bold",
            pad=16,
        )
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="upper left", frameon=False, ncol=3, fontsize=10)
        input_coverage = aggregates["input_coverage"]
        coverage = aggregates["coverage_by_method"]

        def summary(method: str) -> str:
            overall = aggregates["overall_event_equal"]["methods"][method]
            value = overall["mean_absolute_percent_error"]
            value_label = "n/a" if value is None else f"{float(value):.2f}%"
            row_coverage = coverage[method]["evaluable_row_coverage_fraction"]
            coverage_label = (
                "n/a"
                if row_coverage is None
                else f"{100.0 * float(row_coverage):.1f}%"
            )
            return f"{styles[method]['label']}: {value_label} @ {coverage_label} coverage"

        fig.text(
            0.5,
            0.035,
            (
                f"Common evaluable origins: {input_coverage['evaluable_row_count']}/"
                f"{input_coverage['scheduled_row_count']}\n"
                + "  |  ".join(summary(method) for method in METHODS)
            ),
            ha="center",
            va="bottom",
            fontsize=8.4,
            color="#4B5563",
            wrap=True,
        )
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.25, top=0.91)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    column_count = min(3, len(cells))
    row_count = (len(cells) + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.3 * column_count, max(6.5, 3.5 * row_count + 3.0)),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.12,
        top=0.73,
        wspace=0.20,
        hspace=0.34,
    )
    formatter = FuncFormatter(lambda value, _position: f"{value / 1_000_000:.1f}M")
    flattened_axes = list(axes.flat)
    for axis, (event_id, tier) in zip(flattened_axes, cells):
        selected = sorted(
            (
                row
                for row in rows
                if int(row["event_id"]) == event_id
                and int(row["tier"]) == tier
            ),
            key=lambda row: int(row["origin_hours"]),
        )
        hours = np.asarray([row["origin_hours"] for row in selected], dtype=float)
        actual_values = {float(row["actual_final_score"]) for row in selected}
        if len(actual_values) != 1:
            raise ValueError("actual final score changed across origins")
        actual = actual_values.pop()
        for method in METHODS:
            style = styles[method]
            available = [
                row
                for row in selected
                if row["predictions"][method] is not None
            ]
            if not available:
                continue
            method_hours = np.asarray(
                [row["origin_hours"] for row in available],
                dtype=float,
            )
            predicted = np.asarray(
                [row["predictions"][method] for row in available], dtype=float
            )
            axis.plot(
                method_hours,
                predicted,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4.5,
            )
        axis.axhline(
            actual,
            color="#111827",
            linestyle=(0, (5, 3)),
            linewidth=2.0,
            label="Actual final",
        )
        axis.set_title(f"Event {event_id} — T{tier}", fontsize=11, weight="bold")
        axis.set_xlabel("Origin (hours after start)")
        axis.set_ylabel("Final score")
        axis.yaxis.set_major_formatter(formatter)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors="#374151", labelsize=9)
    for axis in flattened_axes[len(cells) :]:
        axis.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "HHWX reward-tier final-score forecasts by masked origin",
        y=0.975,
        fontsize=14,
        weight="bold",
        color="#111827",
    )
    fig.text(
        0.5,
        0.94,
        (
            f"{len(normalized_event_tiers)} completed events; {len(cells)} canonical "
            "reward-tier cells; each predictor uses time <= origin only"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#4B5563",
    )
    overall = aggregates["overall_event_equal"]["methods"]

    def overall_mape(method: str) -> str:
        value = overall[method]["mean_absolute_percent_error"]
        return "n/a" if value is None else f"{float(value):.2f}%"

    fig.text(
        0.5,
        0.018,
        (
            "Event-equal MAPE after reward-tier/origin means — "
            f"paired same-rank {overall_mape('skeleton_kf_same_rank_paired_intersection')}  |  "
            f"paired reward-behavior {overall_mape('skeleton_kf_reward_behavior_paired_intersection')}\n"
            f"available same-rank {overall_mape('skeleton_kf_same_rank_history')}  |  "
            f"available reward-behavior {overall_mape('skeleton_kf_reward_behavior_history')}  |  "
            f"pace {overall_mape('behavior_pace_model')}\n"
            f"planned-duration {overall_mape('planned_duration_average_speed')}  |  "
            f"last-two {overall_mape('last_two_nonnegative_slope')}  |  "
            f"persistence {overall_mape('persistence')}"
        ),
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#374151",
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_document(
    loaded_events: Sequence[LoadedEvaluationEvent],
    excluded_events: Sequence[ExcludedEvaluationEvent],
    rows: Sequence[Mapping[str, Any]],
    *,
    requested_event_ids: Sequence[int],
    pace_prior: PacePrior,
    skeleton_preset: SkeletonPreset,
    skeleton_source_hashes: Mapping[str, str],
    skeleton_history_provenance: Sequence[Mapping[str, Any]],
    skeleton_failure_count: int,
    execution: Mapping[str, Any],
    evaluator_source_sha256: str,
) -> dict[str, Any]:
    if len(pace_prior.source_model_sha256) != 64:
        raise ValueError("behavior pace model source hash is invalid")
    if len(pace_prior.source_builder_sha256) != 64:
        raise ValueError("behavior pace prior builder source hash is invalid")
    if set(skeleton_source_hashes) != set(SKELETON_SOURCE_FILES):
        raise ValueError("Skeleton+KF source hash snapshot is incomplete")
    skeleton_failure_count = int(skeleton_failure_count)
    if skeleton_failure_count < 0:
        raise ValueError("skeleton_failure_count cannot be negative")
    normalized_evaluator_hash = str(evaluator_source_sha256).lower()
    if len(normalized_evaluator_hash) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized_evaluator_hash
    ):
        raise ValueError("evaluator_source_sha256 must be a SHA-256 hex digest")
    requested_ids = tuple(
        _positive_int(event_id, "requested event id")
        for event_id in requested_event_ids
    )
    if not requested_ids or len(set(requested_ids)) != len(requested_ids):
        raise ValueError("requested_event_ids must be non-empty and unique")
    loaded_by_id = {int(loaded.event.event_id): loaded for loaded in loaded_events}
    excluded_by_id = {int(excluded.event_id): excluded for excluded in excluded_events}
    if len(loaded_by_id) != len(loaded_events):
        raise ValueError("loaded_events repeats an event ID")
    if len(excluded_by_id) != len(excluded_events):
        raise ValueError("excluded_events repeats an event ID")
    if set(loaded_by_id) & set(excluded_by_id):
        raise ValueError("an event cannot be both evaluated and excluded")
    if set(requested_ids) != set(loaded_by_id) | set(excluded_by_id):
        raise ValueError(
            "requested events must be partitioned exactly into evaluated and excluded"
        )
    if any(
        excluded.status != "not_applicable"
        or excluded.reason != "empty_canonical_reward_tiers"
        or excluded.reward_tiers
        for excluded in excluded_events
    ):
        raise ValueError("only explicit empty canonical reward tiers may be excluded")

    event_tiers = {
        event_id: tuple(loaded_by_id[event_id].event.reward_tiers)
        for event_id in sorted(loaded_by_id)
    }
    aggregates = aggregate_results(rows, event_tiers)
    observed_skeleton_failures = sum(
        row.get("scoring_status") == "evaluable"
        and row["method_status"][method].get("success") is not True
        for row in rows
        for method in SKELETON_METHODS
    )
    if observed_skeleton_failures != skeleton_failure_count:
        raise ValueError(
            "skeleton_failure_count does not match per-origin method statuses"
        )
    expected_scoring_rows = {
        (event_id, int(tier), int(hour))
        for event_id, tiers in event_tiers.items()
        for tier in tiers
        for hour in origin_hours(loaded_by_id[event_id].event)
    }
    observed_scoring_rows = {
        (
            int(row["event_id"]),
            int(row["tier"]),
            int(row["origin_hours"]),
        )
        for row in rows
    }
    if observed_scoring_rows != expected_scoring_rows:
        raise ValueError(
            "per-origin rows do not match the complete 24-hour canonical reward grid"
        )
    if tuple(requested_ids) == DEFAULT_EVENT_IDS:
        input_unavailable_count = sum(
            row.get("scoring_status") != "evaluable" for row in rows
        )
        if input_unavailable_count or skeleton_failure_count:
            raise ValueError(
                "default 318/319 publication requires complete common input and "
                "all Skeleton methods"
            )
    evaluates_t1000 = any(1000 in tiers for tiers in event_tiers.values())
    if loaded_by_id and max(pace_prior.training_event_ids) >= min(loaded_by_id):
        raise ValueError(
            "behavior pace prior training events must precede every evaluated event"
        )
    return {
        "schema_version": "reward-tier-evaluation-v6",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution": dict(execution),
        "evaluation_pipeline": {
            "implementation": {
                "hash_algorithm": "sha256",
                "files": {
                    EVALUATOR_SOURCE_NAME: normalized_evaluator_hash,
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
        },
        "scope": {
            "requested_event_ids": list(requested_ids),
            "evaluated_event_ids": list(sorted(loaded_by_id)),
            "excluded_event_ids": list(sorted(excluded_by_id)),
            "canonical_reward_tiers_by_event": {
                str(event_id): list(tiers)
                for event_id, tiers in event_tiers.items()
            },
            "status": "complete" if loaded_by_id else "not_applicable",
            "origin_hours": "24, 48, ... with origin_at < end_at",
            "completion_rule": (
                "each target event end_at is strictly before the frozen evaluation "
                "start timestamp"
            ),
            "history_completion_rule": (
                "every historical event must satisfy history.end_at < target.start_at"
            ),
            "behavior_pace_training_boundary": {
                "training_event_min": int(min(pace_prior.training_event_ids)),
                "training_event_max": int(max(pace_prior.training_event_ids)),
                "training_event_count": len(pace_prior.training_event_ids),
                "rule": "max(training_event_ids) < min(evaluated_event_ids)",
            },
            "visibility_rule": (
                "raw frozen HHWX target-tier rows with time <= origin_at; target "
                "T10 uses frozen HHWX rows when present, otherwise the authorized "
                "HHWX-to-Bestdori route with origin_as_of=origin_at"
            ),
            "truth_access_rule": (
                "the earliest HHWX tracker bucket with time > meta.end_at remains "
                "sealed until every model prediction for the event has returned; "
                "latest and isFinal rows are not used to select truth"
            ),
            "truth_bucket_rule": "min(tier_records[].time where time > meta.end_at)",
            "alternate_model": None,
            "t1000_evaluated": evaluates_t1000,
            "provider_routing": {
                "target_reward_tiers": "frozen_hhwx_exact_tier_only",
                "historical_reward_tiers": (
                    "frozen_hhwx_exact_tier_then_hhwx_to_bestdori_exact_tier"
                ),
                "t10_scale": "hhwx_then_bestdori",
                "bestdori_authorized": True,
                "rank_interpolation_used": False,
                "semantic_status": "normal_provider_routing",
            },
            "reward_behavior_history_semantics": (
                "reported as a separate Skeleton+KF method: map each target reward "
                "class to the historical event's own HHWX rankingRewards[3] toRank; "
                "use the same exact rank through the authorized provider route; no "
                "rank interpolation, target input, or model substitution"
            ),
            "paired_intersection_semantics": (
                "for each target event/tier, intersect legal same-rank and "
                "reward-behavior histories by event ID before count truncation; paired "
                "methods then use identical selected event IDs, engine, preset, target "
                "prefix, and T10 data, differing only in historical fixed-tier mapping"
            ),
            "source_contract": {
                "target_provider": "hhwx_frozen_cache",
                "history_and_t10_provider_route": "hhwx_then_bestdori",
                "server": HHWX_SERVER,
                "event_start_field": "meta.start_at",
                "event_end_field": "meta.end_at",
                "tracker_time_field": "tier_records[].time",
                "tracker_score_field": "tier_records[].ep",
                "final_truth_flag_field": "not_used_for_truth_selection",
                "timestamp_unit": "unix_epoch_milliseconds",
                "event_display_timezone": "Asia/Shanghai",
            },
        },
        "inputs": {
            "evaluated": [
                {
                    "event_id": loaded.event.event_id,
                    "evaluation_status": "evaluated",
                    "event_start_at": int(loaded.event.start_at),
                    "event_end_at": int(loaded.event.end_at),
                    "cache_path": str(loaded.cache_path),
                    "cache_sha256": loaded.cache_sha256,
                    "canonical_reward_tiers": list(loaded.event.reward_tiers),
                    "reward_tier_provenance": dict(loaded.reward_provenance),
                    "collection_source": loaded.collection_metadata.get("source"),
                    "collection_server": loaded.collection_metadata.get("server"),
                    "collection_fetched_at": loaded.collection_metadata.get(
                        "fetched_at"
                    ),
                    "collection_availability_status": (
                        loaded.collection_metadata.get("availability_status")
                    ),
                }
                for loaded in sorted(
                    loaded_events,
                    key=lambda value: int(value.event.event_id),
                )
            ],
            "excluded": [
                {
                    "event_id": excluded.event_id,
                    "evaluation_status": excluded.status,
                    "exclusion_reason": excluded.reason,
                    "event_start_at": int(excluded.event_start_at),
                    "event_end_at": int(excluded.event_end_at),
                    "cache_path": str(excluded.cache_path),
                    "cache_sha256": excluded.cache_sha256,
                    "canonical_reward_tiers": list(excluded.reward_tiers),
                    "reward_tier_provenance": dict(excluded.reward_provenance),
                    "collection_source": excluded.collection_metadata.get("source"),
                    "collection_server": excluded.collection_metadata.get("server"),
                    "collection_fetched_at": excluded.collection_metadata.get(
                        "fetched_at"
                    ),
                    "collection_availability_status": (
                        excluded.collection_metadata.get("availability_status")
                    ),
                }
                for excluded in sorted(
                    excluded_events,
                    key=lambda value: int(value.event_id),
                )
            ],
        },
        "behavior_pace_model": {
            "class": "behavior_pace_model.predict_tier_curve",
            "implementation": {
                "hash_algorithm": "sha256",
                "files": {
                    PACE_MODEL_SOURCE_NAME: pace_prior.source_model_sha256,
                    PACE_BUILDER_SOURCE_NAME: pace_prior.source_builder_sha256,
                },
            },
            "prior": {
                "path": str(pace_prior.path),
                "sha256": pace_prior.sha256,
                "schema_version": pace_prior.schema_version,
                "training_event_ids": list(pace_prior.training_event_ids),
                "coverage_gate": dict(pace_prior.coverage_gate),
                "aggregate_weights": {
                    name: float(value)
                    for name, value in zip(
                        PACE_COMPONENT_NAMES,
                        pace_prior.weights.as_array(),
                    )
                },
                "model_config": asdict(pace_prior.config),
                "source_model_sha256": pace_prior.source_model_sha256,
                "source_builder_sha256": pace_prior.source_builder_sha256,
            },
            "prediction_rule": (
                "anchor each target tier at its last score visible at time <= origin; "
                "apply the frozen event-equal 192..283 pace prior through event end"
            ),
            "diagnostics_location": "per_origin[].behavior_pace_model",
            "provider_used": None,
            "fallback_used": False,
            "tier_interpolation_used": False,
            "target_runs_by_event": {
                str(event_id): [f"T{tier}" for tier in tiers]
                for event_id, tiers in event_tiers.items()
            },
        },
        "production_baseline_model": {
            "class": "PredictionEngine (Skeleton + Kalman Filter)",
            "methods": {
                "skeleton_kf_same_rank_history": (
                    "unchanged engine using same-rank HHWX histories"
                ),
                "skeleton_kf_reward_behavior_history": (
                    "unchanged engine using each historical event's own exact fixed-tier "
                    "curve for the target reward-behavior class"
                ),
                "skeleton_kf_same_rank_paired_intersection": (
                    "unchanged engine using same-rank curves from the paired common "
                    "history event IDs"
                ),
                "skeleton_kf_reward_behavior_paired_intersection": (
                    "unchanged engine using reward-behavior curves from the paired "
                    "common history event IDs"
                ),
            },
            "preset": {
                "path": str(skeleton_preset.path),
                "sha256": str(skeleton_preset.sha256),
                "metadata": dict(skeleton_preset.metadata),
                "params": dict(skeleton_preset.params),
                "effective_config": dict(skeleton_preset.effective_config),
            },
            "implementation": {
                "hash_algorithm": "sha256",
                "files": {
                    name: str(skeleton_source_hashes[name])
                    for name in SKELETON_SOURCE_FILES
                },
            },
            "target_scale_rule": (
                "production mean of the three largest valid positive T10 speeds; "
                "frozen HHWX exact T10 prefix is used when present, otherwise the "
                "authorized HHWX-to-Bestdori route is called with the same origin"
            ),
            "start_time_rule": (
                "raw HHWX start/end retained in provenance; unchanged production "
                "maintenance-delay start correction is applied inside engine inputs; "
                "debug_hours=(absolute_origin-engine_corrected_start)/1h"
            ),
            "history_selection": list(skeleton_history_provenance),
            "prediction_failure_count": skeleton_failure_count,
            "provider_routing": "hhwx_then_bestdori_same_exact_tier_or_t10",
            "fallback_usage_location": (
                "per_origin[].skeleton_kf and history_selection[].candidate_audit"
            ),
            "rank_interpolation_used": False,
            "t1000_evaluated": evaluates_t1000,
        },
        "baseline_definitions": {
            "skeleton_kf_same_rank_history": (
                "online learned_notebook Skeleton+KF with same-rank HHWX history"
            ),
            "skeleton_kf_reward_behavior_history": (
                "same Skeleton+KF with inferred reward-behavior-class HHWX history "
                "mapping; every mapped curve is an exact public fixed tier"
            ),
            "skeleton_kf_same_rank_paired_intersection": (
                "same-rank side of the paired history-event intersection ablation"
            ),
            "skeleton_kf_reward_behavior_paired_intersection": (
                "reward-behavior side of the paired history-event intersection ablation"
            ),
            "persistence": "last visible target-tier score",
            "last_two_nonnegative_slope": (
                "last visible score plus max(0, last two point slope) to end"
            ),
            "planned_duration_average_speed": (
                "last visible score * raw planned event duration / raw scheduled "
                "elapsed time since meta.start_at; no maintenance-delay start correction"
            ),
        },
        "metric_definitions": {
            "error": "prediction - actual, in score points",
            "absolute_error": "abs(prediction - actual), in score points",
            "signed_percent_error": "100 * (prediction - actual) / actual",
            "absolute_percent_error": "100 * abs(prediction - actual) / actual",
            "smape_percent": (
                "200 * abs(prediction - actual) / (abs(prediction) + abs(actual))"
            ),
        },
        "per_origin": list(rows),
        "aggregates": aggregates,
    }


def _default_output_paths(
    event_ids: Sequence[int],
) -> tuple[Path, Path]:
    normalized = tuple(int(value) for value in event_ids)
    if normalized == DEFAULT_EVENT_IDS:
        return DEFAULT_OUTPUT_JSON, DEFAULT_OUTPUT_PNG
    if len(normalized) <= 8:
        scope_label = "_".join(str(value) for value in normalized)
    else:
        digest = hashlib.sha256(
            ",".join(str(value) for value in normalized).encode("ascii")
        ).hexdigest()[:8]
        scope_label = (
            f"{normalized[0]}_{normalized[-1]}_{len(normalized)}events_{digest}"
        )
    base = PROJECT_ROOT / "event_data" / f"reward_tier_evaluation_{scope_label}"
    return base.with_suffix(".json"), base.with_suffix(".png")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pace and learned-notebook Skeleton+KF forecasts for "
            "canonical HHWX reward tiers on completed events."
        )
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--event-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit completed event IDs (default: 318 319).",
    )
    scope.add_argument(
        "--event-id-range",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        default=None,
        help="Inclusive completed-event ID interval.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--pace-prior-json",
        type=Path,
        default=DEFAULT_PACE_PRIOR_PATH,
        help="Required frozen behavior pace prior (default: train192_283).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path; omitted derives a scope-specific name.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Output PNG path; omitted derives a scope-specific name.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = datetime.now(timezone.utc)
    started_counter = perf_counter()
    interval = args.event_id_range
    event_ids = resolve_event_ids(
        args.event_ids,
        None if interval is None else interval[0],
        None if interval is None else interval[1],
    )
    default_json, default_png = _default_output_paths(event_ids)
    output_json = Path(args.output_json or default_json)
    output_png = Path(args.output_png or default_png)
    evaluator_source_sha256 = _evaluator_source_sha256()
    skeleton_source_hashes = _skeleton_source_hashes()
    skeleton_preset = load_skeleton_preset()
    pace_prior = load_pace_prior(args.pace_prior_json)
    if min(event_ids) <= max(pace_prior.training_event_ids):
        raise ValueError(
            "evaluation event IDs must be strictly newer than every behavior pace "
            "prior training event"
        )
    loaded_events: list[LoadedEvaluationEvent] = []
    excluded_events: list[ExcludedEvaluationEvent] = []
    completed_as_of_ms = int(started_at.timestamp() * 1000)
    for event_id in event_ids:
        loaded = load_hhwx_event(
            args.cache_dir,
            event_id,
            completed_as_of_ms=completed_as_of_ms,
        )
        if isinstance(loaded, ExcludedEvaluationEvent):
            excluded_events.append(loaded)
        else:
            loaded_events.append(loaded)
    rows: list[dict[str, Any]] = []
    skeleton_history_provenance: list[Mapping[str, Any]] = []
    if loaded_events:
        skeleton_replay = SkeletonKFReplay(
            skeleton_preset,
            loaded_events,
            cache_dir=args.cache_dir,
        )
        try:
            for loaded in loaded_events:
                rows.extend(
                    evaluate_event(
                        loaded,
                        pace_prior=pace_prior,
                        skeleton_replay=skeleton_replay,
                    )
                )
            skeleton_history_provenance = skeleton_replay.provenance()
        finally:
            skeleton_replay.close()
    if _skeleton_source_hashes() != skeleton_source_hashes:
        raise RuntimeError("Skeleton+KF source changed during evaluation")
    if _evaluator_source_sha256() != evaluator_source_sha256:
        raise RuntimeError("evaluator source changed during evaluation")
    if _cache_sha256(skeleton_preset.path) != skeleton_preset.sha256:
        raise RuntimeError("Skeleton+KF preset changed during evaluation")
    if _cache_sha256(pace_prior.path) != pace_prior.sha256:
        raise RuntimeError("behavior pace prior changed during evaluation")
    for loaded in loaded_events:
        if _cache_sha256(loaded.cache_path) != loaded.cache_sha256:
            raise RuntimeError(
                f"target event {loaded.event.event_id} cache changed during evaluation"
            )
    if (
        _cache_sha256(PROJECT_ROOT / PACE_MODEL_SOURCE_NAME)
        != pace_prior.source_model_sha256
    ):
        raise RuntimeError("behavior pace model source changed during evaluation")
    if (
        _cache_sha256(PROJECT_ROOT / PACE_BUILDER_SOURCE_NAME)
        != pace_prior.source_builder_sha256
    ):
        raise RuntimeError("behavior pace prior builder source changed during evaluation")
    completed_at = datetime.now(timezone.utc)
    skeleton_failure_count = sum(
        row.get("scoring_status") == "evaluable"
        and row["method_status"][method].get("success") is not True
        for row in rows
        for method in SKELETON_METHODS
    )
    document = build_document(
        loaded_events,
        excluded_events,
        rows,
        requested_event_ids=event_ids,
        pace_prior=pace_prior,
        skeleton_preset=skeleton_preset,
        skeleton_source_hashes=skeleton_source_hashes,
        skeleton_history_provenance=skeleton_history_provenance,
        skeleton_failure_count=skeleton_failure_count,
        execution={
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": float(perf_counter() - started_counter),
            "timestamp_timezone": "UTC",
        },
        evaluator_source_sha256=evaluator_source_sha256,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    event_tiers = {
        int(loaded.event.event_id): tuple(loaded.event.reward_tiers)
        for loaded in loaded_events
    }
    render_plot(rows, output_png, event_tiers=event_tiers)
    print(
        json.dumps(
            {
                "output_json": str(output_json.resolve()),
                "output_png": str(output_png.resolve()),
                "origin_rows": len(rows),
                "evaluated_event_ids": document["scope"]["evaluated_event_ids"],
                "excluded_event_ids": document["scope"]["excluded_event_ids"],
                "overall": document["aggregates"]["overall_event_equal"],
            },
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
