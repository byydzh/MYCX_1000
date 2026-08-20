"""Fixed-tier cutoff surfaces with an explicit historical as-of contract.

The public tracker APIs expose a sparse, fixed set of event cutoffs.  This
module deliberately never creates a fictional rank between those cutoffs.
"""
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from config import ALL_TRACKER_TIERS, canonicalize_tracker_tiers


_TIME_COLUMNS = ("time", "timestamp")
_VALUE_COLUMNS = ("value", "ep", "points", "score", "pt")
_AVAILABLE_AT_COLUMNS = ("available_at", "availableAt")


@dataclass
class TierSurfaceSnapshot:
    """A causal input surface as it was knowable at ``origin_as_of``.

    ``surface`` contains only scores timestamped no later than the origin.
    ``quality_report`` records data defects instead of silently repairing them;
    downstream feature code must use this object (or its ``surface``) rather
    than the unbounded raw tracker frames.
    """

    origin_as_of: int
    surface: pd.DataFrame
    tiers: tuple[int, ...]
    quality_report: dict[str, Any]
    provenance: dict[int, dict[str, Any]]


def _first_existing(columns, candidates):
    return next((name for name in candidates if name in columns), None)


def _provenance_for_tier(
    frame: pd.DataFrame,
    tier: int,
    source: Optional[str],
    fetched_at: Any,
    available_at: Any,
) -> dict[str, Any]:
    attrs = getattr(frame, "attrs", {}) or {}
    source_value = source if source is not None else attrs.get("source")
    fetched_value = fetched_at if fetched_at is not None else attrs.get("fetched_at")
    available_value = available_at if available_at is not None else attrs.get("available_at")
    try:
        available_value = int(available_value) if available_value is not None else None
    except (TypeError, ValueError):
        available_value = None
    try:
        fetched_value = int(fetched_value) if fetched_value is not None else None
    except (TypeError, ValueError):
        fetched_value = None
    return {
        "tier": int(tier),
        "source": source_value or "unknown",
        "fetched_at": fetched_value,
        "available_at": available_value,
        # fetched_at is audit provenance, never substituted for available_at.
        "availability_status": (
            "explicit" if available_value is not None
            else "unknown_degraded_no_available_at"
        ),
    }


def build_origin_as_of_tier_snapshot(
    tier_data: Mapping[int, Optional[pd.DataFrame]],
    meta: Mapping[str, Any],
    origin_as_of: int,
    *,
    tiers=None,
    source: Optional[str] = None,
    fetched_at: Any = None,
    available_at: Any = None,
) -> TierSurfaceSnapshot:
    """Build a fixed-tier snapshot with no data after ``origin_as_of``.

    An API record may be used only if ``time <= origin_as_of``.  If the input
    includes a row-level ``available_at`` it is also required to be no later
    than the origin.  Existing tracker archives lack that field, so their
    timestamp-only result is retained but visibly marked as a degraded
    availability guarantee; this function never invents an availability time.
    """
    try:
        origin = int(origin_as_of)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"origin_as_of must be a millisecond timestamp, got {origin_as_of!r}") from exc
    if origin < 0:
        raise ValueError("origin_as_of must be non-negative")

    expected_tiers = canonicalize_tracker_tiers(
        ALL_TRACKER_TIERS if tiers is None else tiers
    )
    if not isinstance(meta, Mapping) or "start_at" not in meta:
        raise ValueError("meta.start_at is required")
    try:
        start_at = int(meta["start_at"])
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("meta.start_at must be a millisecond timestamp") from exc
    if start_at < 0:
        raise ValueError("meta.start_at must be non-negative")

    frames = []
    provenance: dict[int, dict[str, Any]] = {}
    per_tier: dict[str, dict[str, Any]] = {}
    missing_tiers = []
    any_availability_degraded = False

    for tier in expected_tiers:
        raw_frame = tier_data.get(tier)
        if raw_frame is None:
            # JSON caches necessarily serialize rank keys as strings.
            raw_frame = tier_data.get(str(tier))
        if raw_frame is None or raw_frame.empty:
            empty_frame = raw_frame if isinstance(raw_frame, pd.DataFrame) else pd.DataFrame()
            prov = _provenance_for_tier(
                empty_frame, tier, source, fetched_at, available_at
            )
            # No row means there is no row-level availability evidence.  Keep
            # completeness and availability as separate defects, but never
            # claim that the whole requested snapshot has explicit as-of data.
            prov["availability_status"] = "unknown_degraded_missing_tier"
            provenance[int(tier)] = prov
            any_availability_degraded = True
            missing_tiers.append(int(tier))
            per_tier[str(tier)] = {
                "status": "missing",
                "raw_rows": 0 if raw_frame is None else int(len(raw_frame)),
                "usable_rows": 0,
                "post_origin_rows_excluded": 0,
                "duplicate_timestamps": 0,
                "non_monotonic_steps": 0,
                "invalid_timestamp_rows": 0,
                "invalid_value_rows": 0,
                "future_available_rows_excluded": 0,
                "unknown_available_rows_excluded": 0,
            }
            continue

        raw = raw_frame.copy()
        raw_count = int(len(raw))
        time_col = _first_existing(raw.columns, _TIME_COLUMNS)
        value_col = _first_existing(raw.columns, _VALUE_COLUMNS)
        prov = _provenance_for_tier(raw_frame, tier, source, fetched_at, available_at)
        provenance[int(tier)] = prov

        if time_col is None or value_col is None:
            prov["availability_status"] = "unknown_degraded_invalid_schema"
            any_availability_degraded = True
            missing_tiers.append(int(tier))
            per_tier[str(tier)] = {
                "status": "invalid_schema",
                "raw_rows": raw_count,
                "usable_rows": 0,
                "post_origin_rows_excluded": 0,
                "duplicate_timestamps": 0,
                "non_monotonic_steps": 0,
                "invalid_timestamp_rows": raw_count if time_col is None else 0,
                "invalid_value_rows": raw_count if value_col is None else 0,
                "future_available_rows_excluded": 0,
                "unknown_available_rows_excluded": 0,
            }
            continue

        normalized = pd.DataFrame({
            "time": pd.to_numeric(raw[time_col], errors="coerce"),
            "value": pd.to_numeric(raw[value_col], errors="coerce"),
        })
        valid_time = np.isfinite(normalized["time"]) & (normalized["time"] >= 0)
        valid_value = np.isfinite(normalized["value"]) & (normalized["value"] >= 0)
        invalid_timestamp_rows = int((~valid_time).sum())
        invalid_value_rows = int((~valid_value).sum())
        normalized = normalized[valid_time & valid_value].copy()
        normalized["time"] = normalized["time"].astype("int64")

        post_origin_rows = int((normalized["time"] > origin).sum())
        normalized = normalized[normalized["time"] <= origin].copy()

        available_col = _first_existing(raw.columns, _AVAILABLE_AT_COLUMNS)
        future_available_rows = 0
        unknown_available_rows = 0
        if available_col is not None:
            # Preserve row alignment with the numeric-normalized source.
            availability = pd.to_numeric(raw[available_col], errors="coerce")
            availability = availability.loc[valid_time & valid_value]
            # Rebuild through explicit index alignment after the time filter.
            normalized["_available_at"] = availability.reindex(normalized.index)
            unknown_available = normalized["_available_at"].isna()
            known_late = (
                normalized["_available_at"].notna()
                & (normalized["_available_at"] > origin)
            )
            unknown_available_rows = int(unknown_available.sum())
            future_available_rows = int(known_late.sum())
            # Once a response supplies row-level availability, a missing value
            # is not evidence that the row was knowable.  Exclude it rather
            # than silently falling back to tracker time.
            normalized = normalized[
                ~(known_late | unknown_available)
            ]
            if unknown_available_rows:
                prov["availability_status"] = (
                    "degraded_partial_row_level_available_at"
                )
                any_availability_degraded = True
            else:
                prov["availability_status"] = "explicit_row_level"
        elif prov["available_at"] is not None:
            if int(prov["available_at"]) > origin:
                future_available_rows = int(len(normalized))
                normalized = normalized.iloc[0:0].copy()
            prov["availability_status"] = "explicit_response_level"

        if prov["availability_status"].startswith("unknown"):
            any_availability_degraded = True

        duplicate_timestamps = int(normalized.duplicated("time", keep=False).sum())
        revision_sort = ["time"] + (
            ["_available_at"] if "_available_at" in normalized else []
        )
        normalized = (
            normalized.sort_values(revision_sort, kind="mergesort")
            .drop_duplicates("time", keep="last")
        )
        if "_available_at" in normalized:
            normalized = normalized.drop(columns=["_available_at"])
        non_monotonic_steps = int((normalized["value"].diff() < 0).sum())
        if normalized.empty:
            missing_tiers.append(int(tier))
            status = "no_usable_as_of_records"
        else:
            status = "ok"
            normalized["hours"] = (normalized["time"] - start_at) / 3600000.0
            normalized["tier"] = int(tier)
            frames.append(normalized[["hours", "tier", "value"]])

        per_tier[str(tier)] = {
            "status": status,
            "raw_rows": raw_count,
            "usable_rows": int(len(normalized)),
            "post_origin_rows_excluded": post_origin_rows,
            "duplicate_timestamps": duplicate_timestamps,
            "non_monotonic_steps": non_monotonic_steps,
            "invalid_timestamp_rows": invalid_timestamp_rows,
            "invalid_value_rows": invalid_value_rows,
            "future_available_rows_excluded": future_available_rows,
            "unknown_available_rows_excluded": unknown_available_rows,
        }

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        surface = combined.pivot(index="hours", columns="tier", values="value").sort_index()
        surface = surface.reindex(columns=[tier for tier in expected_tiers if tier in surface.columns])
    else:
        surface = pd.DataFrame()

    observed_tiers = tuple(int(tier) for tier in surface.columns)
    report = {
        "contract": "fixed_tier_origin_as_of_v2",
        "origin_as_of": origin,
        "expected_tiers": list(expected_tiers),
        "observed_tiers": list(observed_tiers),
        "missing_tiers": missing_tiers,
        "completeness": {
            "expected_count": len(expected_tiers),
            "observed_count": len(observed_tiers),
            "fraction": float(len(observed_tiers) / len(expected_tiers)) if expected_tiers else 1.0,
        },
        "availability": {
            "status": "degraded_timestamp_only" if any_availability_degraded else "explicit_available_at",
            "reason": (
                "one or more tracker inputs lack available_at; timestamp cutoff is still enforced"
                if any_availability_degraded else None
            ),
        },
        "quality_summary": {
            "post_origin_rows_excluded": sum(
                item["post_origin_rows_excluded"] for item in per_tier.values()
            ),
            "duplicate_timestamps": sum(
                item["duplicate_timestamps"] for item in per_tier.values()
            ),
            "non_monotonic_steps": sum(
                item["non_monotonic_steps"] for item in per_tier.values()
            ),
            "invalid_timestamp_rows": sum(
                item["invalid_timestamp_rows"] for item in per_tier.values()
            ),
            "invalid_value_rows": sum(
                item["invalid_value_rows"] for item in per_tier.values()
            ),
            "future_available_rows_excluded": sum(
                item["future_available_rows_excluded"]
                for item in per_tier.values()
            ),
            "unknown_available_rows_excluded": sum(
                item["unknown_available_rows_excluded"]
                for item in per_tier.values()
            ),
        },
        "tiers": per_tier,
    }
    return TierSurfaceSnapshot(
        origin_as_of=origin,
        surface=surface,
        tiers=observed_tiers,
        quality_report=report,
        provenance=provenance,
    )


def build_tier_surface(tier_data: dict, meta: dict):
    """
    从全量 tier tracker 数据构建 time × tier 观测矩阵。

    Args:
        tier_data: {tier: DataFrame | None}，每层 tracker 原始数据
        meta: 活动元数据，含 start_at (ms)

    Returns:
        pd.DataFrame: 行=hours_elapsed, 列=tier, 值=score。

    注意：
        这里刻意只保留 API 真实给出的档线，不在 tier 维度插值。
        奖励档线附近的竞争压力往往是离散断点，连续插值会抹平这些断点。
    """
    start_ts = meta.get('start_at', 0)
    available_tiers = [t for t, df in tier_data.items() if df is not None and not df.empty]

    if not available_tiers:
        return pd.DataFrame()

    frames = []
    for tier in available_tiers:
        df = tier_data[tier].copy()
        if 'ep' in df.columns:
            df = df.rename(columns={'ep': 'value'})
        if 'time' not in df.columns:
            continue
        df['hours'] = (df['time'] - start_ts) / 3600000.0
        df = df.sort_values('hours')
        df['tier'] = tier
        frames.append(df[['hours', 'tier', 'value']])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    surface = combined.pivot_table(
        index='hours', columns='tier', values='value', aggfunc='last'
    )
    surface = surface.sort_index()
    return surface


def align_observed_tier_surface(surface: pd.DataFrame, freq_hours: float | None = None):
    """
    只在时间轴对真实档线做对齐，不在档线轴插值。

    Args:
        surface: build_tier_surface 返回的真实档线观测矩阵。
        freq_hours: 可选时间网格步长；None 时沿用原始观测时间点。

    Returns:
        (aligned_surface, observed_tiers, hour_grid)
    """
    if surface.empty:
        return surface, np.array([]), np.array([])

    observed_tiers = np.array(sorted(surface.columns.tolist()), dtype=int)
    source = surface.sort_index()

    if freq_hours is not None and freq_hours > 0:
        start_h = float(source.index.min())
        end_h = float(source.index.max())
        hour_grid = np.arange(start_h, end_h + (freq_hours * 0.5), freq_hours)
        aligned = source.reindex(source.index.union(hour_grid)).sort_index()
        aligned = aligned.interpolate(method='index', limit_direction='forward')
        aligned = aligned.reindex(hour_grid)
    else:
        aligned = source.copy()

    # 分数只能随时间非递减；用 cummax 抑制 tracker 抖动。
    aligned = aligned.ffill().cummax()
    return aligned, observed_tiers, aligned.index.values


def interpolate_tier_surface(surface: pd.DataFrame):
    """
    Backward-compatible wrapper.

    旧实现会在 tier 维度做 PCHIP 插值；当前模型路线改为保留真实档线，
    所以这里仅做时间轴对齐并返回真实 observed_tiers。
    """
    return align_observed_tier_surface(surface)
