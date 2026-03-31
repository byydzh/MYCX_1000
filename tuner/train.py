import argparse
import json
import logging
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from config import DEFAULT_CONFIG, DEFAULT_MODEL_ID, PROJECT_ROOT
from domain_models import EventMeta
from tuner.data_cache import CACHE_ROOT, cache_historical_events
from tuner.offline_evaluator import OfflineEvaluator

logger = logging.getLogger("tuner.train")

DEFAULT_SNAPSHOT_PROGRESS_BUCKETS = [
    (0.15, 0.25),
    (0.25, 0.40),
    (0.40, 0.55),
    (0.55, 0.70),
    (0.70, 0.85),
]
DEFAULT_MIN_TEST_EVENTS = 4
DEFAULT_MIN_SNAPSHOT_HOURS = 18.0
DEFAULT_SNAPSHOT_ROUND_TO_HOURS = 0.25
DEFAULT_CURVE_SAMPLE_COUNT = 3
DEFAULT_CURVE_SAMPLE_SEED_OFFSET = 200_003
DEFAULT_CURVE_LOSS_WEIGHT = 0.10
DEFAULT_FORMAL_MIN_EVENT_ID = 200
DEFAULT_FORMAL_TRAIN_MAX_EVENT_ID = 300
DEFAULT_FORMAL_HOLDOUT_MIN_EVENT_ID = 301

TIER1_SEARCH_SPACE = {
    "weekend_multiplier": (0.9, 1.35),
    "panic_ease_power": (0.5, 1.8),
    "panic_scaler": (0.8, 1.5),
    "t_start_cmp": (3.0, 10.0),
    "refit_weight_scale": (1.0, 30.0),
    "duration_align_log_full_weight": (0.08, 0.35),
    "duration_window_align_max_weight": (0.2, 0.9),
    "duration_param_align_max_weight": (0.2, 0.95),
    "kf_Q_scale": (1e-6, 1e-4),
    "kf_base_R": (0.03, 0.5),
    "kf_adaptive_R_numerator": (500.0, 5000.0),
    "kf_trend_half_life_hours": (1.0, 8.0),
}


def get_tier_search_space(tier: int = 1) -> Dict[str, Tuple[float, float]]:
    if int(tier) != 1:
        raise ValueError("Only tier 1 is implemented in this first notebook-oriented training pass.")
    return TIER1_SEARCH_SPACE.copy()


def _event_type_for(evaluator: Optional[OfflineEvaluator], event_id: int) -> str:
    if evaluator is None:
        return "unknown"

    cached_event = evaluator.cached_events.get(int(event_id), {})
    return str(
        cached_event.get("event_type")
        or cached_event.get("meta", {}).get("event_type")
        or "unknown"
    ).lower()


def split_event_ids(
    event_ids: Iterable[int],
    train_ratio: float = 0.8,
    evaluator: Optional[OfflineEvaluator] = None,
    min_test_events: int = 1,
) -> Tuple[List[int], List[int]]:
    ordered = sorted(int(event_id) for event_id in event_ids)
    if len(ordered) < 2:
        return ordered, []

    split_idx = max(1, min(len(ordered) - 1, int(len(ordered) * train_ratio)))
    target_test_size = max(int(min_test_events), len(ordered) - split_idx)

    candidate_splits = []
    for candidate_idx in range(1, len(ordered)):
        candidate_test = ordered[candidate_idx:]
        if len(candidate_test) < target_test_size:
            continue

        if evaluator is not None:
            test_types = {_event_type_for(evaluator, event_id) for event_id in candidate_test}
            all_types = {_event_type_for(evaluator, event_id) for event_id in ordered}
            missing_type_count = len(all_types - test_types)
            type_coverage = len(test_types)
        else:
            missing_type_count = 0
            type_coverage = 0

        candidate_splits.append((
            missing_type_count,
            -type_coverage,
            abs(len(candidate_test) - target_test_size),
            -candidate_idx,
            candidate_idx,
        ))

    if candidate_splits:
        split_idx = min(candidate_splits)[-1]

    return ordered[:split_idx], ordered[split_idx:]


def split_formal_event_ids(
    event_ids: Iterable[int],
    *,
    min_event_id: int = DEFAULT_FORMAL_MIN_EVENT_ID,
    train_max_event_id: int = DEFAULT_FORMAL_TRAIN_MAX_EVENT_ID,
    holdout_min_event_id: int = DEFAULT_FORMAL_HOLDOUT_MIN_EVENT_ID,
) -> Tuple[List[int], List[int]]:
    ordered = sorted(
        int(event_id)
        for event_id in event_ids
        if int(event_id) >= int(min_event_id)
    )
    train_ids = [event_id for event_id in ordered if event_id <= int(train_max_event_id)]
    test_ids = [event_id for event_id in ordered if event_id >= int(holdout_min_event_id)]
    return train_ids, test_ids


def split_formal_training_event_ids(
    event_ids: Iterable[int],
    *,
    train_ratio: float = 0.8,
    evaluator: Optional[OfflineEvaluator] = None,
    min_test_events: int = DEFAULT_MIN_TEST_EVENTS,
    min_event_id: int = DEFAULT_FORMAL_MIN_EVENT_ID,
    train_max_event_id: int = DEFAULT_FORMAL_TRAIN_MAX_EVENT_ID,
    holdout_min_event_id: int = DEFAULT_FORMAL_HOLDOUT_MIN_EVENT_ID,
) -> Tuple[List[int], List[int], List[int]]:
    formal_pool_ids, external_holdout_ids = split_formal_event_ids(
        event_ids,
        min_event_id=min_event_id,
        train_max_event_id=train_max_event_id,
        holdout_min_event_id=holdout_min_event_id,
    )
    train_ids, validation_ids = split_event_ids(
        formal_pool_ids,
        train_ratio=train_ratio,
        evaluator=evaluator,
        min_test_events=min_test_events,
    )
    return train_ids, validation_ids, external_holdout_ids


def _event_total_hours(evaluator: OfflineEvaluator, event_id: int) -> float:
    cached_event = evaluator.cached_events[int(event_id)]
    meta = cached_event.get("meta", {})
    meta_obj = EventMeta.from_dict(int(event_id), meta) if isinstance(meta, dict) else meta
    return float(meta_obj.total_hours)


def normalize_snapshot_plan(snapshot_plan: Optional[Dict[int, Iterable[float]]]) -> Dict[int, List[float]]:
    normalized = {}
    for event_id, hours in (snapshot_plan or {}).items():
        normalized[int(event_id)] = sorted({float(hour) for hour in hours})
    return normalized


def build_snapshot_plan(
    evaluator: OfflineEvaluator,
    event_ids: Iterable[int],
    *,
    seed: int = 42,
    progress_buckets: Optional[Iterable[Tuple[float, float]]] = None,
    samples_per_bucket: int = 1,
    min_snapshot_hours: float = DEFAULT_MIN_SNAPSHOT_HOURS,
    round_to_hours: float = DEFAULT_SNAPSHOT_ROUND_TO_HOURS,
) -> Dict[int, List[float]]:
    rng = np.random.default_rng(int(seed))
    buckets = list(progress_buckets) if progress_buckets is not None else list(DEFAULT_SNAPSHOT_PROGRESS_BUCKETS)
    snapshot_plan = {}

    for event_id in sorted(int(item) for item in event_ids):
        total_hours = _event_total_hours(evaluator, event_id)
        max_snapshot_hours = max(min_snapshot_hours, total_hours * 0.90)
        sampled_hours = []

        for progress_low, progress_high in buckets:
            for _ in range(max(1, int(samples_per_bucket))):
                hour_low = max(float(min_snapshot_hours), float(progress_low) * total_hours)
                hour_high = min(float(progress_high) * total_hours, max_snapshot_hours)
                if hour_high <= hour_low:
                    continue

                sampled = float(rng.uniform(hour_low, hour_high))
                if round_to_hours > 0:
                    sampled = float(round(sampled / round_to_hours) * round_to_hours)
                sampled = min(max(sampled, hour_low), hour_high)
                sampled_hours.append(sampled)

        if not sampled_hours:
            fallback = min(max_snapshot_hours, max(min_snapshot_hours, total_hours * 0.5))
            sampled_hours = [fallback]

        snapshot_plan[event_id] = sorted({round(float(hour), 4) for hour in sampled_hours})

    return snapshot_plan


def build_runtime_config(vector: np.ndarray, param_names: List[str], base_config: Optional[dict] = None) -> dict:
    runtime_config = DEFAULT_CONFIG.copy()
    if base_config:
        runtime_config.update(base_config)
    for idx, name in enumerate(param_names):
        runtime_config[name] = float(vector[idx])
    return runtime_config


def validate_constraints(config: dict) -> List[str]:
    violations = []
    if not (config["smooth_thresh1"] < config["smooth_thresh2"] < config["smooth_hard_cap"]):
        violations.append("smooth thresholds must satisfy thresh1 < thresh2 < hard_cap")
    if (config["smooth_thresh2"] - config["smooth_thresh1"]) < 0.05:
        violations.append("smooth thresholds must keep a minimum margin of 0.05")
    if not (config["ratio_min"] < config["ratio_max"]):
        violations.append("ratio_min must be smaller than ratio_max")
    if not (config["t_start_cmp"] < config["t_end_cap"]):
        violations.append("t_start_cmp must be smaller than t_end_cap")
    if config.get("kf_Q_scale", 0.0) <= 0.0:
        violations.append("kf_Q_scale must be positive")
    if config.get("kf_base_R", 0.0) <= 0.0:
        violations.append("kf_base_R must be positive")
    if config.get("kf_adaptive_R_numerator", 0.0) <= 0.0:
        violations.append("kf_adaptive_R_numerator must be positive")
    if config.get("kf_trend_half_life_hours", 0.0) <= 0.0:
        violations.append("kf_trend_half_life_hours must be positive")
    return violations


def _objective_with_history(
    vector: np.ndarray,
    *,
    param_names: List[str],
    evaluator: OfflineEvaluator,
    train_ids: List[int],
    debug_hours_list: Iterable[float],
    snapshot_plan: Optional[Dict[int, Iterable[float]]],
    curve_sample_count: int,
    curve_sample_seed: int,
    curve_loss_weight: float,
    base_config: dict,
    history: List[dict],
    verbose: bool = False,
    log_every_n_evals: int = 25,
) -> float:
    runtime_config = build_runtime_config(vector, param_names, base_config=base_config)
    violations = validate_constraints(runtime_config)
    if violations:
        loss = 1e6 + (1000.0 * len(violations))
    else:
        metrics = evaluator.evaluate(
            runtime_config,
            event_ids=train_ids,
            debug_hours_list=debug_hours_list,
            snapshot_plan=snapshot_plan,
            curve_sample_count=curve_sample_count,
            curve_sample_seed=curve_sample_seed,
            curve_loss_weight=curve_loss_weight,
        )
        loss = float(metrics["objective_loss"])

    history.append({
        "iteration": len(history) + 1,
        "loss": loss,
        "final_relative_mse": None if violations else float(metrics["relative_mse"]),
        "curve_relative_mse": None if violations else metrics.get("curve_relative_mse"),
        "params": {name: float(runtime_config[name]) for name in param_names},
    })

    if verbose and (len(history) % max(1, int(log_every_n_evals)) == 0):
        print(
            f"[train] eval={len(history)} loss={loss:.6f} "
            f"best={min(item['loss'] for item in history):.6f}",
            flush=True,
        )
    return loss


def save_learned_preset(
    output_path: Path,
    params: dict,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    train_metrics: dict,
    test_metrics: dict,
    baseline_test_metrics: Optional[dict],
    train_ids: List[int],
    test_ids: List[int],
    external_holdout_metrics: Optional[dict] = None,
    external_holdout_ids: Optional[List[int]] = None,
    description: str = "",
) -> Path:
    payload = {
        "_meta": {
            "name": output_path.stem,
            "description": description or "ML learned preset generated by tuner.train",
            "author": "tuner/train.py",
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "model_id": model_id,
            "training_stats": {
                "train_events": len(train_ids),
                "test_events": len(test_ids),
                "train_relative_mse": train_metrics.get("relative_mse"),
                "test_relative_mse": test_metrics.get("relative_mse"),
                "test_rmse": test_metrics.get("rmse"),
                "baseline_test_relative_mse": None if baseline_test_metrics is None else baseline_test_metrics.get("relative_mse"),
                "baseline_test_rmse": None if baseline_test_metrics is None else baseline_test_metrics.get("rmse"),
                "external_holdout_events": len(external_holdout_ids or []),
                "external_holdout_relative_mse": None if external_holdout_metrics is None else external_holdout_metrics.get("relative_mse"),
                "external_holdout_rmse": None if external_holdout_metrics is None else external_holdout_metrics.get("rmse"),
                "improvement_pct": None if baseline_test_metrics is None or not np.isfinite(baseline_test_metrics.get("relative_mse", np.nan)) else (
                    100.0 * (baseline_test_metrics["relative_mse"] - test_metrics["relative_mse"]) / max(baseline_test_metrics["relative_mse"], 1e-12)
                ),
            },
        },
        "params": params,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def save_training_report(report_dir: Path, report_payload: dict) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    return report_path


def run_training(
    *,
    cache_root: Optional[Path] = None,
    train_ids: Optional[Iterable[int]] = None,
    test_ids: Optional[Iterable[int]] = None,
    debug_hours_list: Optional[Iterable[float]] = None,
    snapshot_plan: Optional[Dict[int, Iterable[float]]] = None,
    snapshot_progress_buckets: Optional[Iterable[Tuple[float, float]]] = None,
    snapshot_samples_per_bucket: int = 1,
    min_snapshot_hours: float = DEFAULT_MIN_SNAPSHOT_HOURS,
    curve_sample_count: int = DEFAULT_CURVE_SAMPLE_COUNT,
    curve_loss_weight: float = DEFAULT_CURVE_LOSS_WEIGHT,
    tier: int = 1,
    maxiter: int = 10,
    popsize: int = 8,
    seed: int = 42,
    refresh_cache: bool = False,
    prepare_cache: bool = True,
    history_count: int = 80,
    min_test_events: int = DEFAULT_MIN_TEST_EVENTS,
    use_formal_holdout_split: bool = False,
    formal_min_event_id: int = DEFAULT_FORMAL_MIN_EVENT_ID,
    formal_train_max_event_id: int = DEFAULT_FORMAL_TRAIN_MAX_EVENT_ID,
    formal_holdout_min_event_id: int = DEFAULT_FORMAL_HOLDOUT_MIN_EVENT_ID,
    external_holdout_ids: Optional[Iterable[int]] = None,
    external_holdout_snapshot_plan: Optional[Dict[int, Iterable[float]]] = None,
    output_preset_path: Optional[Path] = None,
    save_report: bool = True,
    base_config: Optional[dict] = None,
    verbose: bool = False,
    log_every_n_evals: int = 25,
    workers: int = 1,
    use_gpu: bool = False,
) -> dict:
    started_at = time.time()
    cache_root = Path(cache_root) if cache_root else CACHE_ROOT
    debug_hours_list = list(debug_hours_list) if debug_hours_list is not None else [24.0, 48.0, 72.0]
    runtime_base_config = DEFAULT_CONFIG.copy()
    if base_config:
        runtime_base_config.update(base_config)

    if use_gpu and verbose:
        print(
            "[train] use_gpu=True was requested, but current offline evaluator / scipy DE path is CPU-only. "
            "Falling back to CPU.",
            flush=True,
        )

    de_workers = int(workers)
    if de_workers == 0:
        raise ValueError("workers must be -1 (all cores) or a positive integer")

    if verbose:
        print(
            f"[train] start tier={tier} seed={seed} maxiter={maxiter} popsize={popsize} "
            f"workers={de_workers} fallback_debug_hours={list(debug_hours_list)} "
            f"curve_samples={curve_sample_count} curve_weight={curve_loss_weight}",
            flush=True,
        )

    if prepare_cache:
        cache_summary = cache_historical_events(
            history_count=history_count,
            cache_root=cache_root,
            refresh=refresh_cache,
            api_source=runtime_base_config.get("api_source", DEFAULT_CONFIG["api_source"]),
        )
    else:
        cache_summary = {
            "cache_root": str(cache_root),
            "requested": history_count,
            "candidate_ids": [],
            "fetched": 0,
            "reused": 0,
            "failed": [],
            "cached_total_before": None,
            "cached_total_after": None,
        }
    evaluator = OfflineEvaluator.from_cache_dir(cache_root=cache_root)

    available_ids = evaluator.available_event_ids()
    eligible_ids = evaluator.eligible_event_ids(
        similar_count=runtime_base_config.get("similar_count", DEFAULT_CONFIG["similar_count"]),
        ignore_ids=runtime_base_config.get("ignore_event_ids", []),
    )
    external_holdout_ids = list(external_holdout_ids or [])
    if train_ids is None and test_ids is None:
        if use_formal_holdout_split:
            train_ids, test_ids, external_holdout_ids = split_formal_training_event_ids(
                eligible_ids,
                train_ratio=0.8,
                evaluator=evaluator,
                min_test_events=min_test_events,
                min_event_id=formal_min_event_id,
                train_max_event_id=formal_train_max_event_id,
                holdout_min_event_id=formal_holdout_min_event_id,
            )
        else:
            train_ids, test_ids = split_event_ids(
                eligible_ids,
                train_ratio=0.8,
                evaluator=evaluator,
                min_test_events=min_test_events,
            )
    else:
        train_ids = list(train_ids or [])
        test_ids = list(test_ids or [])

    if not train_ids:
        raise ValueError(
            "No eligible train event ids available for offline training. "
            f"cached={len(available_ids)}, eligible={len(eligible_ids)}. "
            "Try prepare_cache=True, increase history_count, lower similar_count, or clear ignore_event_ids for offline learning."
        )

    param_space = get_tier_search_space(tier=tier)
    param_names = list(param_space.keys())
    bounds = [param_space[name] for name in param_names]
    objective_history = []
    generation_history = []

    normalized_snapshot_plan = normalize_snapshot_plan(snapshot_plan)
    normalized_external_holdout_snapshot_plan = normalize_snapshot_plan(external_holdout_snapshot_plan)
    if normalized_snapshot_plan:
        train_snapshot_plan = {
            int(event_id): normalized_snapshot_plan[int(event_id)]
            for event_id in train_ids
            if int(event_id) in normalized_snapshot_plan
        }
        test_snapshot_plan = {
            int(event_id): normalized_snapshot_plan[int(event_id)]
            for event_id in test_ids
            if int(event_id) in normalized_snapshot_plan
        }
    else:
        train_snapshot_plan = build_snapshot_plan(
            evaluator,
            train_ids,
            seed=seed,
            progress_buckets=snapshot_progress_buckets,
            samples_per_bucket=snapshot_samples_per_bucket,
            min_snapshot_hours=min_snapshot_hours,
        )
        test_snapshot_plan = build_snapshot_plan(
            evaluator,
            test_ids,
            seed=seed + 9973,
            progress_buckets=snapshot_progress_buckets,
            samples_per_bucket=snapshot_samples_per_bucket,
            min_snapshot_hours=min_snapshot_hours,
        )

    if normalized_external_holdout_snapshot_plan:
        benchmark_holdout_snapshot_plan = {
            int(event_id): normalized_external_holdout_snapshot_plan[int(event_id)]
            for event_id in external_holdout_ids
            if int(event_id) in normalized_external_holdout_snapshot_plan
        }
    else:
        benchmark_holdout_snapshot_plan = build_snapshot_plan(
            evaluator,
            external_holdout_ids,
            seed=seed + 19973,
            progress_buckets=snapshot_progress_buckets,
            samples_per_bucket=snapshot_samples_per_bucket,
            min_snapshot_hours=min_snapshot_hours,
        ) if external_holdout_ids else {}

    curve_sample_seed = int(seed) + DEFAULT_CURVE_SAMPLE_SEED_OFFSET
    baseline_train = evaluator.evaluate(
        runtime_base_config,
        event_ids=train_ids,
        debug_hours_list=debug_hours_list,
        snapshot_plan=train_snapshot_plan,
        curve_sample_count=curve_sample_count,
        curve_sample_seed=curve_sample_seed,
        curve_loss_weight=curve_loss_weight,
    )
    baseline_test = evaluator.evaluate(
        runtime_base_config,
        event_ids=test_ids,
        debug_hours_list=debug_hours_list,
        snapshot_plan=test_snapshot_plan,
        curve_sample_count=curve_sample_count,
        curve_sample_seed=curve_sample_seed,
        curve_loss_weight=curve_loss_weight,
    ) if test_ids else None
    baseline_external_holdout = evaluator.evaluate(
        runtime_base_config,
        event_ids=external_holdout_ids,
        debug_hours_list=debug_hours_list,
        snapshot_plan=benchmark_holdout_snapshot_plan,
        curve_sample_count=curve_sample_count,
        curve_sample_seed=curve_sample_seed,
        curve_loss_weight=curve_loss_weight,
    ) if external_holdout_ids else None

    if verbose:
        eval_mode = "snapshot_plan" if train_snapshot_plan else "debug_hours"
        print(
            f"[train] eval_mode={eval_mode} baseline train_mse={baseline_train['relative_mse']:.6f} "
            f"test_mse={None if baseline_test is None else round(baseline_test['relative_mse'], 6)} "
            f"baseline_curve={baseline_train.get('curve_relative_mse')} "
            f"train_snapshots={sum(len(v) for v in train_snapshot_plan.values())}",
            flush=True,
        )

    objective = partial(
        _objective_with_history,
        param_names=param_names,
        evaluator=evaluator,
        train_ids=train_ids,
        debug_hours_list=debug_hours_list,
        snapshot_plan=train_snapshot_plan,
        curve_sample_count=curve_sample_count,
        curve_sample_seed=curve_sample_seed,
        curve_loss_weight=curve_loss_weight,
        base_config=runtime_base_config,
        history=objective_history,
        verbose=verbose,
        log_every_n_evals=log_every_n_evals,
    )

    de_updating = "immediate" if de_workers == 1 else "deferred"
    de_population = max(1, int(popsize) * len(param_names))

    def _callback(_xk, convergence):
        gen_idx = len(generation_history) + 1
        est_evals = len(objective_history) if objective_history else de_population * gen_idx

        runtime_config = build_runtime_config(_xk, param_names, base_config=runtime_base_config)
        violations = validate_constraints(runtime_config)
        if violations:
            candidate_loss = 1e6 + (1000.0 * len(violations))
        else:
            candidate_metrics = evaluator.evaluate(
                runtime_config,
                event_ids=train_ids,
                debug_hours_list=debug_hours_list,
                snapshot_plan=train_snapshot_plan,
                curve_sample_count=curve_sample_count,
                curve_sample_seed=curve_sample_seed,
                curve_loss_weight=curve_loss_weight,
            )
            candidate_loss = float(candidate_metrics["objective_loss"])

        generation_history.append({
            "iteration": gen_idx,
            "loss": candidate_loss,
            "params": {name: float(runtime_config[name]) for name in param_names},
            "estimated_evals": int(est_evals),
            "convergence": float(convergence),
        })

        if verbose:
            if objective_history:
                best_loss = min(item["loss"] for item in objective_history)
            else:
                best_loss = min(item["loss"] for item in generation_history)
            print(
                f"[train] generation_done evals={est_evals} "
                f"best_loss={best_loss:.6f} convergence={float(convergence):.6f}",
                flush=True,
            )
        return False

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=False,
        updating=de_updating,
        workers=de_workers,
        disp=False,
        callback=_callback,
        atol=0.0,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        init="latinhypercube",
        constraints=(),
        x0=None,
        integrality=None,
        vectorized=False,
    )

    # With multiprocessing workers, per-eval history lives in child processes and does not
    # come back to the parent process. Fall back to generation-level history for plotting/reporting.
    if not objective_history and generation_history:
        objective_history = [
            {
                "iteration": item["iteration"],
                "loss": item["loss"],
                "params": item["params"],
            }
            for item in generation_history
        ]

    best_config = build_runtime_config(result.x, param_names, base_config=runtime_base_config)
    train_metrics = evaluator.evaluate(
        best_config,
        event_ids=train_ids,
        debug_hours_list=debug_hours_list,
        snapshot_plan=train_snapshot_plan,
        curve_sample_count=curve_sample_count,
        curve_sample_seed=curve_sample_seed,
        curve_loss_weight=curve_loss_weight,
    )
    test_metrics = evaluator.evaluate(
        best_config,
        event_ids=test_ids,
        debug_hours_list=debug_hours_list,
        snapshot_plan=test_snapshot_plan,
        curve_sample_count=curve_sample_count,
        curve_sample_seed=curve_sample_seed,
        curve_loss_weight=curve_loss_weight,
    ) if test_ids else None
    external_holdout_metrics = evaluator.evaluate(
        best_config,
        event_ids=external_holdout_ids,
        debug_hours_list=debug_hours_list,
        snapshot_plan=benchmark_holdout_snapshot_plan,
        curve_sample_count=curve_sample_count,
        curve_sample_seed=curve_sample_seed,
        curve_loss_weight=curve_loss_weight,
    ) if external_holdout_ids else None

    runtime_sec = time.time() - started_at
    if verbose:
        print(
            f"[train] done in {runtime_sec:.1f}s | train_mse={train_metrics['relative_mse']:.6f} "
            f"test_mse={None if test_metrics is None else round(test_metrics['relative_mse'], 6)} "
            f"train_curve={train_metrics.get('curve_relative_mse')} "
            f"test_curve={None if test_metrics is None else test_metrics.get('curve_relative_mse')}",
            flush=True,
        )

    preset_path = None
    if output_preset_path is not None:
        preset_path = save_learned_preset(
            Path(output_preset_path),
            best_config,
            train_metrics=train_metrics,
            test_metrics=test_metrics or {"relative_mse": None, "rmse": None},
            baseline_test_metrics=baseline_test,
            train_ids=train_ids,
            test_ids=test_ids,
            external_holdout_metrics=external_holdout_metrics,
            external_holdout_ids=external_holdout_ids,
            description="Notebook-oriented differential evolution tier1 preset.",
        )

    report_payload = {
        "cache_summary": cache_summary,
        "available_ids": available_ids,
        "eligible_ids": eligible_ids,
        "tier": tier,
        "seed": seed,
        "debug_hours_list": list(debug_hours_list),
        "evaluation_mode": "snapshot_plan" if train_snapshot_plan else "debug_hours",
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
        "curve_sample_seed": int(curve_sample_seed),
        "curve_loss_weight": float(curve_loss_weight),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "external_holdout_ids": external_holdout_ids,
        "use_formal_holdout_split": bool(use_formal_holdout_split),
        "formal_min_event_id": int(formal_min_event_id),
        "formal_train_max_event_id": int(formal_train_max_event_id),
        "formal_holdout_min_event_id": int(formal_holdout_min_event_id),
        "train_snapshot_plan": train_snapshot_plan,
        "test_snapshot_plan": test_snapshot_plan,
        "external_holdout_snapshot_plan": benchmark_holdout_snapshot_plan,
        "search_space": {name: list(bounds[idx]) for idx, name in enumerate(param_names)},
        "objective_history": objective_history,
        "generation_history": generation_history,
        "best_params": {name: float(best_config[name]) for name in param_names},
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "baseline_external_holdout": baseline_external_holdout,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "external_holdout_metrics": external_holdout_metrics,
        "optimizer_result": {
            "fun": float(result.fun),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "success": bool(result.success),
            "message": str(result.message),
        },
        "runtime_sec": runtime_sec,
        "preset_path": None if preset_path is None else str(preset_path),
    }

    report_path = None
    if save_report:
        report_path = save_training_report(PROJECT_ROOT / "tuner" / "reports", report_payload)

    return {
        "best_config": best_config,
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "external_holdout_metrics": external_holdout_metrics,
        "objective_history": objective_history,
        "generation_history": generation_history,
        "report_payload": report_payload,
        "report_path": None if report_path is None else str(report_path),
        "preset_path": None if preset_path is None else str(preset_path),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "external_holdout_ids": external_holdout_ids,
        "runtime_sec": runtime_sec,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Offline preset training for mycx_1000")
    parser.add_argument("--history-count", type=int, default=80)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "configs" / "models" / DEFAULT_MODEL_ID / "learned_v1.json"))
    parser.add_argument("--debug-hours", nargs="*", type=float, default=[24.0, 48.0, 72.0])
    parser.add_argument("--min-test-events", type=int, default=DEFAULT_MIN_TEST_EVENTS)
    parser.add_argument("--use-formal-holdout-split", action="store_true")
    parser.add_argument("--formal-min-event-id", type=int, default=DEFAULT_FORMAL_MIN_EVENT_ID)
    parser.add_argument("--formal-train-max-event-id", type=int, default=DEFAULT_FORMAL_TRAIN_MAX_EVENT_ID)
    parser.add_argument("--formal-holdout-min-event-id", type=int, default=DEFAULT_FORMAL_HOLDOUT_MIN_EVENT_ID)
    parser.add_argument("--curve-sample-count", type=int, default=DEFAULT_CURVE_SAMPLE_COUNT)
    parser.add_argument("--curve-loss-weight", type=float, default=DEFAULT_CURVE_LOSS_WEIGHT)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1, help="-1 uses all CPU cores")
    parser.add_argument("--use-gpu", action="store_true", help="Reserved flag. Current training path is CPU-only.")
    args = parser.parse_args()

    result = run_training(
        history_count=args.history_count,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        tier=args.tier,
        refresh_cache=args.refresh_cache,
        debug_hours_list=args.debug_hours,
        min_test_events=args.min_test_events,
        use_formal_holdout_split=args.use_formal_holdout_split,
        formal_min_event_id=args.formal_min_event_id,
        formal_train_max_event_id=args.formal_train_max_event_id,
        formal_holdout_min_event_id=args.formal_holdout_min_event_id,
        curve_sample_count=args.curve_sample_count,
        curve_loss_weight=args.curve_loss_weight,
        output_preset_path=Path(args.output),
        verbose=args.verbose,
        log_every_n_evals=args.log_every,
        workers=args.workers,
        use_gpu=args.use_gpu,
    )

    print(json.dumps({
        "preset_path": result["preset_path"],
        "report_path": result["report_path"],
        "train_relative_mse": result["train_metrics"]["relative_mse"],
        "test_relative_mse": None if result["test_metrics"] is None else result["test_metrics"]["relative_mse"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
