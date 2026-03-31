import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG
from domain_models import EventData, EventMeta
from math_models import CosineModeler, SeasonalityHandler
from prediction_engine import PredictionEngine
from tuner.data_cache import load_cached_events

logger = logging.getLogger("tuner.offline_evaluator")
DEFAULT_CURVE_PROGRESS_BUCKETS = [
    (0.15, 0.35),
    (0.35, 0.60),
    (0.60, 0.85),
]


def _wrap_cached_event(cached_event: dict) -> EventData:
    meta = cached_event["meta"]
    if isinstance(meta, dict):
        meta = EventMeta.from_dict(cached_event["event_id"], meta)

    return EventData(
        meta=meta,
        df=cached_event["dataframe"].copy(),
        scale=float(cached_event["scale"]),
    )


def _calculate_derived_columns(event_data: EventData) -> EventData:
    event_data.clean_data()
    df = event_data.df.copy()

    original_start = event_data.meta.start_at
    valid_points = df[df["value"] > 0]
    if not valid_points.empty:
        first_valid_ts = valid_points.iloc[0]["time"]
        if first_valid_ts > original_start and (first_valid_ts - original_start) < 86400000:
            corrected_start = int((first_valid_ts // 3600000) * 3600000)
            if corrected_start > original_start:
                event_data.meta.start_at = corrected_start

    start_ts = event_data.meta.start_at
    df["hours_elapsed"] = (df["time"] - start_ts) / 3600000.0

    if "speed" not in df.columns:
        diff_val = df["value"].diff()
        diff_time = df["time"].diff() / 60000.0
        speed = diff_val / diff_time
        df["speed"] = speed.fillna(0.0)
        df.loc[~np.isfinite(df["speed"]), "speed"] = 0.0
        df.loc[df["speed"] < 0, "speed"] = 0.0

    if "norm_speed" not in df.columns:
        df["norm_speed"] = df["speed"] / event_data.scale

    event_data.df = df
    return event_data


class OfflineEvaluator:
    def __init__(self, cached_events: Dict[int, dict]):
        self.cached_events = dict(sorted((int(k), v) for k, v in cached_events.items()))

    @classmethod
    def from_cache_dir(cls, cache_root=None, event_ids: Optional[Iterable[int]] = None):
        return cls(load_cached_events(cache_root=cache_root, event_ids=event_ids))

    def available_event_ids(self) -> List[int]:
        return list(self.cached_events.keys())

    def eligible_event_ids(
        self,
        similar_count: Optional[int] = None,
        ignore_ids: Optional[Iterable[int]] = None,
    ) -> List[int]:
        chosen_count = int(similar_count or DEFAULT_CONFIG["similar_count"])
        eligible = []

        for event_id in self.available_event_ids():
            cached_event = self.cached_events[event_id]
            event_type = cached_event.get("event_type") or cached_event.get("meta", {}).get("event_type", "unknown")
            history_ids = self._select_history_event_ids(
                target_event_id=event_id,
                event_type=event_type,
                count=chosen_count,
                ignore_ids=ignore_ids,
            )
            if len(history_ids) >= chosen_count:
                eligible.append(event_id)

        return eligible

    def _select_history_event_ids(
        self,
        target_event_id: int,
        event_type: str,
        count: int,
        ignore_ids: Optional[Iterable[int]] = None,
    ) -> List[int]:
        blocked = {int(item) for item in (ignore_ids or [])}
        selected = []

        for event_id in sorted(self.cached_events.keys(), reverse=True):
            if event_id >= int(target_event_id) or event_id in blocked:
                continue

            cached = self.cached_events[event_id]
            if float(cached.get("scale", 0.0) or 0.0) <= 0:
                continue

            cached_type = cached.get("event_type") or cached.get("meta", {}).get("event_type", "unknown")
            if str(cached_type).lower() != str(event_type).lower():
                continue

            selected.append(event_id)
            if len(selected) >= int(count):
                break

        return selected

    def _prepare_event_data(self, cached_event: dict, debug_hours: Optional[float] = None) -> EventData:
        event_data = _wrap_cached_event(cached_event)
        event_data = _calculate_derived_columns(event_data)
        event_data.full_df = event_data.df.copy()

        if debug_hours is not None:
            limit_ts = event_data.meta.start_at + (float(debug_hours) * 3600 * 1000)
            event_data.df = event_data.df[event_data.df["time"] <= limit_ts].copy()

        return event_data

    def _sample_curve_check_hours(
        self,
        event_id: int,
        observed_hours: float,
        total_hours: float,
        *,
        sample_count: int = 3,
        sample_seed: int = 42,
        progress_buckets: Optional[Iterable[tuple[float, float]]] = None,
    ) -> List[float]:
        if sample_count <= 0 or total_hours <= observed_hours + 1.0:
            return []

        remaining = total_hours - observed_hours
        buckets = list(progress_buckets) if progress_buckets is not None else list(DEFAULT_CURVE_PROGRESS_BUCKETS)
        rng_seed = (
            int(sample_seed) * 1_000_003
            + int(event_id) * 97
            + int(round(observed_hours * 100))
        ) % (2**32)
        rng = np.random.default_rng(rng_seed)

        sampled_hours = []
        bucket_idx = 0
        while len(sampled_hours) < int(sample_count):
            progress_low, progress_high = buckets[bucket_idx % len(buckets)]
            bucket_idx += 1

            hour_low = observed_hours + max(0.5, progress_low * remaining)
            hour_high = min(total_hours, observed_hours + progress_high * remaining)
            if hour_high <= hour_low:
                continue

            sampled = float(rng.uniform(hour_low, hour_high))
            sampled = float(round(sampled / 0.25) * 0.25)
            sampled = min(max(sampled, hour_low), hour_high)
            sampled_hours.append(sampled)

        return sorted({round(hour, 4) for hour in sampled_hours})

    def _build_engine(self, config: dict) -> PredictionEngine:
        runtime_config = DEFAULT_CONFIG.copy()
        runtime_config.update(config or {})

        seasonality = SeasonalityHandler(
            weekend_multiplier=float(runtime_config.get("weekend_multiplier", DEFAULT_CONFIG["weekend_multiplier"])),
            panic_scaler=float(runtime_config.get("panic_scaler", DEFAULT_CONFIG["panic_scaler"])),
            panic_ease_power=float(runtime_config.get("panic_ease_power", DEFAULT_CONFIG["panic_ease_power"])),
        )
        modeler = CosineModeler()
        return PredictionEngine(seasonality, modeler, config=runtime_config)

    def predict_event(
        self,
        event_id: int,
        config: dict,
        debug_hours: Optional[float] = None,
        similar_count: Optional[int] = None,
        curve_sample_count: int = 0,
        curve_sample_seed: int = 42,
        curve_progress_buckets: Optional[Iterable[tuple[float, float]]] = None,
    ) -> Optional[dict]:
        cached_event = self.cached_events.get(int(event_id))
        if cached_event is None:
            return None

        runtime_config = DEFAULT_CONFIG.copy()
        runtime_config.update(config or {})
        chosen_count = int(similar_count or runtime_config.get("similar_count", DEFAULT_CONFIG["similar_count"]))

        target_data = self._prepare_event_data(cached_event, debug_hours=debug_hours)
        if target_data.df.empty or len(target_data.df) < 3:
            return None

        history_ids = self._select_history_event_ids(
            target_event_id=event_id,
            event_type=target_data.meta.event_type,
            count=chosen_count,
            ignore_ids=runtime_config.get("ignore_event_ids", []),
        )
        if len(history_ids) < chosen_count:
            return None

        history_events = [self._prepare_event_data(self.cached_events[item]) for item in history_ids]
        engine = self._build_engine(runtime_config)
        result = engine.predict(target_data, history_events, debug_hours=debug_hours)

        actual_final = float(cached_event["actual_final_score"])
        abs_error = float(result.final_score - actual_final)
        rel_error = float(abs_error / max(actual_final, 1.0))
        log_error = float(np.log1p(result.final_score) - np.log1p(actual_final))

        observed_hours = float(target_data.df["hours_elapsed"].max()) if not target_data.df.empty else 0.0
        curve_check_hours = self._sample_curve_check_hours(
            event_id=event_id,
            observed_hours=observed_hours,
            total_hours=float(target_data.meta.total_hours),
            sample_count=curve_sample_count,
            sample_seed=curve_sample_seed,
            progress_buckets=curve_progress_buckets,
        )
        curve_relative_mse = None
        if curve_check_hours:
            pred_curve_scores = np.interp(curve_check_hours, result.full_t_score, result.full_score)
            actual_curve_scores = np.interp(curve_check_hours, target_data.full_df["hours_elapsed"], target_data.full_df["value"])
            curve_relative_errors = (pred_curve_scores - actual_curve_scores) / np.maximum(actual_curve_scores, 1.0)
            curve_relative_mse = float(np.mean(np.square(curve_relative_errors)))

        return {
            "event_id": int(event_id),
            "event_type": target_data.meta.event_type,
            "debug_hours": None if debug_hours is None else float(debug_hours),
            "observed_hours": observed_hours,
            "history_ids": history_ids,
            "predicted_final": float(result.final_score),
            "actual_final": actual_final,
            "abs_error": abs_error,
            "rel_error": rel_error,
            "log_error": log_error,
            "curve_check_hours": curve_check_hours,
            "curve_relative_mse": curve_relative_mse,
            "ratio": float(result.ratio),
            "scale_factor": float(result.scale_factor),
        }

    def evaluate(
        self,
        config: dict,
        event_ids: Optional[Iterable[int]] = None,
        debug_hours_list: Optional[Iterable[float]] = None,
        snapshot_plan: Optional[Dict[int, Iterable[float]]] = None,
        similar_count: Optional[int] = None,
        curve_sample_count: int = 0,
        curve_sample_seed: int = 42,
        curve_progress_buckets: Optional[Iterable[tuple[float, float]]] = None,
        curve_loss_weight: float = 0.0,
    ) -> dict:
        runtime_config = DEFAULT_CONFIG.copy()
        runtime_config.update(config or {})

        candidate_ids = list(event_ids) if event_ids is not None else self.available_event_ids()
        debug_hours_values = list(debug_hours_list) if debug_hours_list is not None else [24.0, 48.0, 72.0]
        normalized_snapshot_plan = {
            int(event_id): [float(hour) for hour in hours]
            for event_id, hours in (snapshot_plan or {}).items()
        }

        rows = []
        skipped = []
        for event_id in candidate_ids:
            event_debug_hours = normalized_snapshot_plan.get(int(event_id), debug_hours_values)
            for debug_hours in event_debug_hours:
                row = self.predict_event(
                    event_id=event_id,
                    config=runtime_config,
                    debug_hours=debug_hours,
                    similar_count=similar_count,
                    curve_sample_count=curve_sample_count,
                    curve_sample_seed=curve_sample_seed,
                    curve_progress_buckets=curve_progress_buckets,
                )
                if row is None:
                    skipped.append({"event_id": int(event_id), "debug_hours": float(debug_hours)})
                    continue
                rows.append(row)

        if not rows:
            return {
                "count": 0,
                "relative_mse": float("inf"),
                "log_mse": float("inf"),
                "rmse": float("inf"),
                "mape": float("inf"),
                "rows": [],
                "skipped": skipped,
            }

        frame = pd.DataFrame(rows)
        relative_mse = float(np.mean(np.square(frame["rel_error"])))
        log_mse = float(np.mean(np.square(frame["log_error"])))
        rmse = float(np.sqrt(np.mean(np.square(frame["abs_error"]))))
        mape = float(np.mean(np.abs(frame["rel_error"])))
        valid_curve_rows = frame["curve_relative_mse"].dropna() if "curve_relative_mse" in frame.columns else pd.Series(dtype=float)
        curve_relative_mse = float(valid_curve_rows.mean()) if len(valid_curve_rows) > 0 else None
        objective_loss = relative_mse if curve_relative_mse is None else float(relative_mse + float(curve_loss_weight) * curve_relative_mse)

        return {
            "count": int(len(frame)),
            "relative_mse": relative_mse,
            "log_mse": log_mse,
            "rmse": rmse,
            "mape": mape,
            "curve_relative_mse": curve_relative_mse,
            "objective_loss": objective_loss,
            "rows": frame.to_dict(orient="records"),
            "skipped": skipped,
        }
