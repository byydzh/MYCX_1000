import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import DEFAULT_CONFIG, DEFAULT_MODEL_ID, PROJECT_ROOT, list_models, list_presets, load_preset
from tuner.data_cache import CACHE_ROOT, cache_historical_events
from tuner.offline_evaluator import OfflineEvaluator
from tuner.train import (
    DEFAULT_CURVE_LOSS_WEIGHT,
    DEFAULT_CURVE_SAMPLE_COUNT,
    DEFAULT_FORMAL_HOLDOUT_MIN_EVENT_ID,
    DEFAULT_FORMAL_MIN_EVENT_ID,
    DEFAULT_FORMAL_TRAIN_MAX_EVENT_ID,
    DEFAULT_MIN_SNAPSHOT_HOURS,
    DEFAULT_SNAPSHOT_PROGRESS_BUCKETS,
    build_snapshot_plan,
    split_formal_event_ids,
)


def select_benchmark_event_ids(
    evaluator: OfflineEvaluator,
    *,
    similar_count: Optional[int] = None,
    ignore_ids: Optional[Iterable[int]] = None,
    min_event_id: int = DEFAULT_FORMAL_MIN_EVENT_ID,
    train_max_event_id: int = DEFAULT_FORMAL_TRAIN_MAX_EVENT_ID,
    holdout_min_event_id: int = DEFAULT_FORMAL_HOLDOUT_MIN_EVENT_ID,
) -> Tuple[List[int], List[int], List[int]]:
    runtime_config = DEFAULT_CONFIG.copy()
    runtime_config["similar_count"] = int(similar_count or runtime_config["similar_count"])
    runtime_config["ignore_event_ids"] = list(
        DEFAULT_CONFIG["ignore_event_ids"] if ignore_ids is None else ignore_ids
    )

    eligible_ids = evaluator.eligible_event_ids(
        similar_count=runtime_config["similar_count"],
        ignore_ids=runtime_config.get("ignore_event_ids", []),
    )
    eligible_ids = [int(event_id) for event_id in eligible_ids if int(event_id) >= int(min_event_id)]
    train_ids, holdout_ids = split_formal_event_ids(
        eligible_ids,
        min_event_id=min_event_id,
        train_max_event_id=train_max_event_id,
        holdout_min_event_id=holdout_min_event_id,
    )
    return eligible_ids, train_ids, holdout_ids


def summarize_rows_by_event(rows: Iterable[dict]) -> List[dict]:
    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["event_id"]), []).append(row)

    summaries = []
    for event_id, event_rows in sorted(grouped.items()):
        rel_errors = np.asarray([float(item["rel_error"]) for item in event_rows], dtype=float)
        abs_errors = np.asarray([float(item["abs_error"]) for item in event_rows], dtype=float)
        curve_mse_values = [
            float(item["curve_relative_mse"])
            for item in event_rows
            if item.get("curve_relative_mse") is not None
        ]
        summaries.append(
            {
                "event_id": event_id,
                "event_type": str(event_rows[0].get("event_type", "unknown")),
                "snapshot_count": len(event_rows),
                "mean_abs_rel_error": float(np.mean(np.abs(rel_errors))),
                "relative_mse": float(np.mean(np.square(rel_errors))),
                "rmse": float(np.sqrt(np.mean(np.square(abs_errors)))),
                "curve_relative_mse": None if not curve_mse_values else float(np.mean(curve_mse_values)),
                "history_ids": list(event_rows[0].get("history_ids", [])),
            }
        )

    summaries.sort(key=lambda item: (-item["mean_abs_rel_error"], item["event_id"]))
    return summaries


def _resolve_model_ids(model_ids: Optional[Iterable[str]] = None) -> List[str]:
    if model_ids is not None:
        return [str(model_id) for model_id in model_ids]
    return [str(model["id"]) for model in list_models()]


def _resolve_preset_ids_for_model(
    model_id: str,
    preset_ids_by_model: Optional[Dict[str, Iterable[str]]] = None,
) -> List[str]:
    if preset_ids_by_model and model_id in preset_ids_by_model:
        return [str(preset_id) for preset_id in preset_ids_by_model[model_id]]
    return [str(preset["id"]) for preset in list_presets(model_id)]


def run_global_benchmark(
    *,
    cache_root: Optional[Path] = None,
    prepare_cache: bool = False,
    refresh_cache: bool = False,
    history_count: int = 140,
    min_event_id: int = DEFAULT_FORMAL_MIN_EVENT_ID,
    train_max_event_id: int = DEFAULT_FORMAL_TRAIN_MAX_EVENT_ID,
    holdout_min_event_id: int = DEFAULT_FORMAL_HOLDOUT_MIN_EVENT_ID,
    model_ids: Optional[Iterable[str]] = None,
    preset_ids_by_model: Optional[Dict[str, Iterable[str]]] = None,
    benchmark_ignore_ids: Optional[Iterable[int]] = None,
    snapshot_seed: int = 42,
    snapshot_progress_buckets: Optional[Iterable[Tuple[float, float]]] = None,
    snapshot_samples_per_bucket: int = 1,
    min_snapshot_hours: float = DEFAULT_MIN_SNAPSHOT_HOURS,
    debug_hours_list: Optional[Iterable[float]] = None,
    curve_sample_count: int = DEFAULT_CURVE_SAMPLE_COUNT,
    curve_loss_weight: float = DEFAULT_CURVE_LOSS_WEIGHT,
    save_report: bool = True,
) -> dict:
    cache_root = Path(cache_root) if cache_root else CACHE_ROOT
    debug_hours = list(debug_hours_list) if debug_hours_list is not None else [24.0, 48.0, 72.0]
    effective_ignore_ids = list(
        DEFAULT_CONFIG["ignore_event_ids"] if benchmark_ignore_ids is None else benchmark_ignore_ids
    )

    cache_summary = None
    if prepare_cache:
        cache_summary = cache_historical_events(
            history_count=history_count,
            cache_root=cache_root,
            refresh=refresh_cache,
            min_event_id=min_event_id,
        )

    evaluator = OfflineEvaluator.from_cache_dir(cache_root=cache_root)
    eligible_ids, train_ids, holdout_ids = select_benchmark_event_ids(
        evaluator,
        similar_count=DEFAULT_CONFIG["similar_count"],
        ignore_ids=effective_ignore_ids,
        min_event_id=min_event_id,
        train_max_event_id=train_max_event_id,
        holdout_min_event_id=holdout_min_event_id,
    )

    train_snapshot_plan = build_snapshot_plan(
        evaluator,
        train_ids,
        seed=snapshot_seed,
        progress_buckets=snapshot_progress_buckets,
        samples_per_bucket=snapshot_samples_per_bucket,
        min_snapshot_hours=min_snapshot_hours,
    ) if train_ids else {}
    holdout_snapshot_plan = build_snapshot_plan(
        evaluator,
        holdout_ids,
        seed=snapshot_seed + 9973,
        progress_buckets=snapshot_progress_buckets,
        samples_per_bucket=snapshot_samples_per_bucket,
        min_snapshot_hours=min_snapshot_hours,
    ) if holdout_ids else {}

    results = []
    for model_id in _resolve_model_ids(model_ids):
        preset_ids = _resolve_preset_ids_for_model(model_id, preset_ids_by_model=preset_ids_by_model)
        for preset_id in preset_ids:
            runtime_config = load_preset(model_id, preset_id)
            runtime_config["ignore_event_ids"] = list(effective_ignore_ids)
            train_metrics = evaluator.evaluate(
                runtime_config,
                event_ids=train_ids,
                debug_hours_list=debug_hours,
                snapshot_plan=train_snapshot_plan,
                curve_sample_count=curve_sample_count,
                curve_sample_seed=snapshot_seed + 200_003,
                curve_loss_weight=curve_loss_weight,
            ) if train_ids else None
            holdout_metrics = evaluator.evaluate(
                runtime_config,
                event_ids=holdout_ids,
                debug_hours_list=debug_hours,
                snapshot_plan=holdout_snapshot_plan,
                curve_sample_count=curve_sample_count,
                curve_sample_seed=snapshot_seed + 400_003,
                curve_loss_weight=curve_loss_weight,
            ) if holdout_ids else None

            results.append(
                {
                    "model_id": model_id,
                    "preset_id": preset_id,
                    "train_metrics": train_metrics,
                    "holdout_metrics": holdout_metrics,
                    "train_event_summaries": [] if train_metrics is None else summarize_rows_by_event(train_metrics["rows"]),
                    "holdout_event_summaries": [] if holdout_metrics is None else summarize_rows_by_event(holdout_metrics["rows"]),
                }
            )

    results.sort(
        key=lambda item: (
            float("inf") if item["holdout_metrics"] is None else float(item["holdout_metrics"]["objective_loss"]),
            item["model_id"],
            item["preset_id"],
        )
    )

    leaderboard = []
    for item in results:
        train_metrics = item["train_metrics"] or {}
        holdout_metrics = item["holdout_metrics"] or {}
        leaderboard.append(
            {
                "model_id": item["model_id"],
                "preset_id": item["preset_id"],
                "train_relative_mse": train_metrics.get("relative_mse"),
                "train_curve_relative_mse": train_metrics.get("curve_relative_mse"),
                "train_objective_loss": train_metrics.get("objective_loss"),
                "holdout_relative_mse": holdout_metrics.get("relative_mse"),
                "holdout_curve_relative_mse": holdout_metrics.get("curve_relative_mse"),
                "holdout_objective_loss": holdout_metrics.get("objective_loss"),
            }
        )

    report_payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cache_root": str(cache_root),
        "cache_summary": cache_summary,
        "available_ids": evaluator.available_event_ids(),
        "eligible_ids": eligible_ids,
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
        "train_range": [int(min_event_id), int(train_max_event_id)],
        "holdout_range": [int(holdout_min_event_id), None],
        "evaluation_mode": "snapshot_plan",
        "debug_hours_list": debug_hours,
        "snapshot_progress_buckets": [
            [float(low), float(high)]
            for low, high in (
                list(snapshot_progress_buckets)
                if snapshot_progress_buckets is not None
                else list(DEFAULT_SNAPSHOT_PROGRESS_BUCKETS)
            )
        ],
        "snapshot_samples_per_bucket": int(snapshot_samples_per_bucket),
        "min_snapshot_hours": float(min_snapshot_hours),
        "curve_sample_count": int(curve_sample_count),
        "curve_loss_weight": float(curve_loss_weight),
        "benchmark_ignore_ids": effective_ignore_ids,
        "leaderboard": leaderboard,
        "results": results,
    }

    report_path = None
    if save_report:
        report_dir = PROJECT_ROOT / "tuner" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)

    return {
        "leaderboard": leaderboard,
        "results": results,
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
        "report_payload": report_payload,
        "report_path": None if report_path is None else str(report_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Global cross-model benchmark for mycx_1000 presets")
    parser.add_argument("--history-count", type=int, default=140)
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--min-event-id", type=int, default=DEFAULT_FORMAL_MIN_EVENT_ID)
    parser.add_argument("--train-max-event-id", type=int, default=DEFAULT_FORMAL_TRAIN_MAX_EVENT_ID)
    parser.add_argument("--holdout-min-event-id", type=int, default=DEFAULT_FORMAL_HOLDOUT_MIN_EVENT_ID)
    parser.add_argument("--curve-sample-count", type=int, default=DEFAULT_CURVE_SAMPLE_COUNT)
    parser.add_argument("--curve-loss-weight", type=float, default=DEFAULT_CURVE_LOSS_WEIGHT)
    parser.add_argument("--snapshot-seed", type=int, default=42)
    parser.add_argument("--save-report", action="store_true")
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--preset-id", action="append", dest="preset_ids")
    parser.add_argument("--ignore-event-id", action="append", dest="ignore_event_ids", type=int)
    args = parser.parse_args()

    preset_ids_by_model = None
    if args.preset_ids:
        chosen_model_id = args.model_ids[0] if args.model_ids else DEFAULT_MODEL_ID
        preset_ids_by_model = {chosen_model_id: args.preset_ids}

    result = run_global_benchmark(
        prepare_cache=args.prepare_cache,
        refresh_cache=args.refresh_cache,
        history_count=args.history_count,
        min_event_id=args.min_event_id,
        train_max_event_id=args.train_max_event_id,
        holdout_min_event_id=args.holdout_min_event_id,
        model_ids=args.model_ids,
        preset_ids_by_model=preset_ids_by_model,
        benchmark_ignore_ids=args.ignore_event_ids,
        curve_sample_count=args.curve_sample_count,
        curve_loss_weight=args.curve_loss_weight,
        snapshot_seed=args.snapshot_seed,
        save_report=args.save_report,
    )
    print(json.dumps(
        {
            "train_ids": result["train_ids"],
            "holdout_ids": result["holdout_ids"],
            "leaderboard": result["leaderboard"],
            "report_path": result["report_path"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
