"""Bounded chronological experiment for behavior_cohort_model; no network I/O.

screen: train 192..239 and select on 240..263 (12 fixed candidates).
repair: one reward-pressure transfer on the selected config, still on 240..263.
confirm: selected config, original 192..239 prior, score 264..283.
replay: refit selected basis on 192..283; compare 284..319 to saved baseline.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from behavior_cohort_model import CohortConfig, CohortTemplate, fit_mass, predict_cohort_curve
from behavior_pace_prior import decode_training_event, TRAINING_TIERS
from behavior_pace_model import MS_PER_HOUR, PaceWeights, predict_tier_curve, reward_pressure
from scripts.evaluate_reward_tiers_318_319 import (
    load_hhwx_event, LoadedEvaluationEvent, origin_hours, _visible_prefix,
    _target_prefix, _actual_final_score, _baseline_predictions,
)

CACHE = ROOT / "event_data/tier_surface_cache"
OUT = ROOT / "event_data/cohort_experiment_20260908"
REFERENCE = ROOT / "event_data/reward_tier_evaluation_284_319_36events_224cea9f.json"
OLD_PRIOR = ROOT / "configs/behavior_model/pace_prior_train192_283.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def build_templates(event_ids, basis):
    templates, inputs, quality = [], [], []
    for event_id in event_ids:
        path = CACHE / f"{event_id}.json"
        raw = path.read_bytes()
        parsed = decode_training_event(raw, expected_event_id=event_id)
        meta = json.loads(raw)["meta"]
        for tier in TRAINING_TIERS:
            series = parsed.tier_observations[tier]
            diag = parsed.data_quality["tiers"][str(tier)]
            if diag["post_end_rows_mapped"] != 1:
                quality.append({"event_id":event_id, "tier":tier,
                    "excluded":"missing_canonical_terminal_bucket"})
                continue
            if series[-1, 1] != diag["terminal_mapping_score"]:
                quality.append({"event_id":event_id, "tier":tier,
                    "excluded":"terminal_score_conflicts_with_cummax"})
                continue
            mass, error = fit_mass(series[:, 0], series[:, 1],
                duration_hours=parsed.duration_hours,
                start_local_hour=(parsed.start_at / MS_PER_HOUR + 8) % 24,
                tier=tier, reward_tiers=parsed.reward_tiers, basis=basis)
            templates.append(CohortTemplate(event_id, parsed.end_at, meta["event_type"],
                tier, mass, error, reward_pressure(tier, parsed.reward_tiers)))
            quality.append({"event_id":event_id, "tier":tier, "fit_rmse":error,
                "repaired_steps":diag["non_monotonic_steps_repaired"]})
        inputs.append({"event_id":event_id, "sha256":hashlib.sha256(raw).hexdigest()})
        if (event_id - min(event_ids) + 1) % 12 == 0:
            print(f"fit {basis}: {event_id}", flush=True)
    return templates, inputs, quality


def get_templates(train_max, basis):
    """One bounded fit cache per basis, bound to exact sources and input bytes."""
    path = OUT / f"fits_{basis}.json"
    source = {n:sha(ROOT/n) for n in ("behavior_cohort_model.py", "behavior_pace_prior.py",
                                      "scripts/experiment_behavior_cohorts.py")}
    inputs = [{"event_id":e, "sha256":sha(CACHE/f"{e}.json")} for e in range(192,train_max+1)]
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior["sources"] == source and prior["inputs"] == inputs:
            return [CohortTemplate(**t) for t in prior["templates"]], inputs, prior["quality"]
    templates, _, quality = build_templates(range(192,train_max+1), basis)
    write(path, {"sources":source, "inputs":inputs,
        "templates":[asdict(t) for t in templates], "quality":quality})
    return templates, inputs, quality


def old_weights(train_max, before_start):
    prior = json.loads(OLD_PRIOR.read_text(encoding="utf-8"))
    fits = [f for f in prior["event_fits"] if f["event_id"] <= train_max]
    fits = [f for f in fits if json.loads((CACHE/f'{f["event_id"]}.json').read_bytes())["meta"]["end_at"] < before_start]
    for f in fits:
        if sha(CACHE / f'{f["event_id"]}.json') != f["sha256"]:
            raise ValueError("original fit input changed")
    names = ("sustain", "launch", "deadline")
    return PaceWeights.from_sequence(np.mean([[f["weights"][n] for n in names] for f in fits], axis=0))


def aggregate(rows, method, *, event_type=None, tier=None, phase=None, support=None):
    records = []
    for row in rows:
        if method not in row["predictions"]:
            continue
        if support and support not in row["predictions"]:
            continue
        if event_type and row["event_type"] != event_type:
            continue
        if tier and row["tier"] != tier:
            continue
        if phase and row["phase"] != phase:
            continue
        y = row["actual"]
        pred = row["predictions"][method]
        records.append({"event_id":row["event_id"], "tier":row["tier"],
            "mape":100 * abs(pred / y - 1), "bias":100 * (pred / y - 1),
            "mae":abs(pred - y), "smape":200 * abs(pred - y) / (pred + y)})
    if not records:
        return None
    frame = pd.DataFrame(records)
    by_event = frame.groupby(["event_id", "tier"])[["mape","bias","mae","smape"]].mean().groupby("event_id").mean()
    return {**{k:float(v) for k,v in by_event.mean().items()}, "origins":len(records),
        "events":len(by_event), "event_mape":{str(k):float(v) for k,v in by_event.mape.items()}}


def evaluate(event_ids, methods, templates_by_basis, train_max, reference=None):
    rows, inputs = [], []
    reference_rows = {}
    if reference:
        reference_rows = {(r["event_id"],r["tier"],r["origin_at"]):r for r in reference["per_origin"]}
    for event_id in event_ids:
        try:
            loaded = load_hhwx_event(CACHE, event_id)
        except (ValueError, FileNotFoundError) as exc:
            inputs.append({"event_id":event_id, "excluded_reason":str(exc),
                "sha256":sha(CACHE/f"{event_id}.json") if (CACHE/f"{event_id}.json").exists() else None})
            print(f"event {event_id} unavailable: {exc}",flush=True)
            continue
        if not isinstance(loaded, LoadedEvaluationEvent):
            raise ValueError(f"event {event_id}: no reward targets")
        e = loaded.event
        weights = old_weights(train_max, e.start_at)
        legal_templates = {basis:[t for t in ts if t.end_at < e.start_at]
                           for basis,ts in templates_by_basis.items()}
        inputs.append({"event_id":event_id, "sha256":loaded.cache_sha256})
        for hour in origin_hours(e):
            origin = e.start_at + hour * MS_PER_HOUR
            # An absent early prefix is a scheduled input failure, not a reason
            # to drop the event or stop scoring later origins.
            prefix = e.frame.loc[e.frame["time"] <= origin].copy()
            for tier in e.reward_tiers:
                r = {"event_id":event_id, "tier":tier, "event_type":e.event_type,
                    "origin_hours":hour, "origin_at":origin, "predictions":{}, "failures":{},
                    "phase": "early" if hour * MS_PER_HOUR / (e.end_at-e.start_at) < 1/3 else
                             "middle" if hour * MS_PER_HOUR / (e.end_at-e.start_at) < 2/3 else "late"}
                try:
                    target = _target_prefix(prefix, tier)
                except ValueError as exc:
                    r["failures"]["input"] = str(exc)
                    r["actual"] = _actual_final_score(loaded, tier)
                    rows.append(r)
                    continue
                obs = target[["time","score"]].to_numpy()
                # All model predictions are made before opening terminal truth.
                for name, config in methods.items():
                    try:
                        pred, diagnostic = predict_cohort_curve(obs, tier=tier,
                            start_at=e.start_at, end_at=e.end_at, origin_at=origin,
                            event_type=e.event_type, reward_tiers=e.reward_tiers,
                            templates=legal_templates[config.basis], config=config)
                        r["predictions"][name] = float(pred[-1])
                        if len(methods) == 1:
                            r["diagnostics"] = diagnostic
                    except (ValueError, RuntimeError) as exc:
                        r["failures"][name] = str(exc)
                try:
                    old = predict_tier_curve(target, tier, e.start_at, e.end_at,
                        np.array([origin, e.end_at]), e.reward_tiers, weights)
                    r["predictions"]["pace_v1"] = float(old.scores[-1])
                except (ValueError, RuntimeError) as exc:
                    r["failures"]["pace_v1"] = str(exc)
                r["predictions"].update(_baseline_predictions(e, prefix, tier))
                r["actual"] = _actual_final_score(loaded, tier)
                if reference:
                    previous = reference_rows[(event_id,tier,origin)]
                    if previous["scoring_status"] != "evaluable":
                        raise ValueError("reference scoring support mismatch")
                    if not np.isclose(r["actual"], previous["actual_final_score"], rtol=0, atol=0):
                        raise ValueError("reference final truth mismatch")
                    if not np.isclose(r["predictions"]["pace_v1"], previous["predictions"]["behavior_pace_model"], rtol=1e-10):
                        raise ValueError("pace reproduction mismatch")
                    for key in ("skeleton_kf_same_rank_history", "skeleton_kf_reward_behavior_history"):
                        if previous["method_status"][key]["success"]:
                            r["predictions"][key] = previous["predictions"][key]
                        else:
                            r["failures"][key] = previous["method_status"][key]["failure_reason"]
                rows.append(r)
        print(f"evaluated {event_id}: {len(rows)} scheduled origins", flush=True)
    return rows, inputs


def summary(rows):
    methods = sorted({k for r in rows for k in r["predictions"]})
    return {method:aggregate(rows, method) for method in methods}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["screen","repair","confirm","replay"])
    args = parser.parse_args()
    t0 = time.monotonic()
    if args.stage == "screen":
        methods = {f"{basis}_{'type' if same else 'all'}_k{k}": CohortConfig(basis, same, k)
            for basis in ("pace3","cohort6") for same in (False,True) for k in (0,16,64)}
        train_max, evaluation = 239, range(240,264)
    else:
        selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
        if args.stage == "repair":
            selection = selection.get("unrepaired_selection", selection)
            config = {**selection["config"], "reward_transfer":True}
            methods = {selection["name"]+"_reward":CohortConfig(**config)}
            train_max, evaluation = 239, range(240,264)
        else:
            methods = {selection["name"]:CohortConfig(**selection["config"])}
            train_max = 239 if args.stage == "confirm" else 283
            evaluation = range(264,284) if args.stage == "confirm" else range(284,320)
    templates, training_inputs, fit_quality = {}, [], []
    for basis in sorted({m.basis for m in methods.values()}):
        ts, inp, quality = get_templates(train_max, basis)
        templates[basis] = ts
        training_inputs = inp
        fit_quality.extend({**q, "basis":basis} for q in quality)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8")) if args.stage == "replay" else None
    if reference:
        for name, digest in reference["production_baseline_model"]["implementation"]["files"].items():
            if sha(ROOT/name) != digest:
                raise ValueError(f"saved baseline implementation changed: {name}")
        preset = reference["production_baseline_model"]["preset"]
        if sha(ROOT/"configs/models/skeleton_kf/learned_notebook.json") != preset["sha256"]:
            raise ValueError("saved baseline preset changed")
        # Match each evaluated frozen target to the original benchmark receipts.
        receipts = reference["inputs"]["evaluated"]
        by_id = {r["event_id"]:r for r in receipts}
        for event_id in evaluation:
            if sha(CACHE / f"{event_id}.json") != by_id[event_id]["cache_sha256"]:
                raise ValueError("benchmark target cache changed")
    rows, eval_inputs = evaluate(evaluation, methods, templates, train_max, reference)
    metrics = summary(rows)
    output = {"stage":args.stage, "configs":{n:asdict(c) for n,c in methods.items()},
        "training_inputs":training_inputs, "evaluation_inputs":eval_inputs,
        "excluded_events":[r for r in eval_inputs if "excluded_reason" in r],
        "source_sha256":{n:sha(ROOT/n) for n in ("behavior_cohort_model.py","scripts/experiment_behavior_cohorts.py")},
        "fit_quality":fit_quality, "metrics":metrics, "rows":rows,
        "runtime_seconds":time.monotonic()-t0}
    if reference:
        output["reference_sha256"] = sha(REFERENCE)
    write(OUT / f"{args.stage}.json", output)
    if args.stage == "screen":
        eligible = [n for n in methods if metrics[n] and metrics[n]["origins"] == metrics["pace_v1"]["origins"]]
        if not eligible:
            raise RuntimeError("no candidate has full common input coverage")
        selected = min(eligible, key=lambda n:metrics[n]["mape"])
        write(OUT / "selection.json", {"name":selected, "config":asdict(methods[selected]),
            "selected_on":"240..263 reward-tier event-equal MAPE", "training":"192..239",
            "selection_mape":metrics[selected]["mape"]})
        print("SELECTED",selected,flush=True)
    if args.stage == "repair":
        selected = next(iter(methods))
        if (metrics[selected]["origins"] == metrics["pace_v1"]["origins"]
                and metrics[selected]["mape"] < selection["selection_mape"]):
            write(OUT / "selection.json", {"name":selected, "config":asdict(methods[selected]),
                "selected_on":"240..263 reward-tier event-equal MAPE; one mechanism repair",
                "training":"192..239", "selection_mape":metrics[selected]["mape"],
                "unrepaired_selection":selection})
            print("REWARD TRANSFER SELECTED",selected,flush=True)
    if args.stage == "replay":
        write(OUT / "prior.json", {"config":asdict(next(iter(methods.values()))),
            "training_inputs":training_inputs, "templates":[asdict(t) for ts in templates.values() for t in ts]})
    print(json.dumps({m:{k:v for k,v in v.items() if k!='event_mape'} for m,v in metrics.items() if v},indent=2))


if __name__ == "__main__":
    main()
