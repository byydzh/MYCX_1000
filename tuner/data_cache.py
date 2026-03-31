import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from config import DEFAULT_API_SOURCE, DEFAULT_SERVER, PROJECT_ROOT
from data_source import create_data_source

logger = logging.getLogger("tuner.data_cache")

CACHE_ROOT = PROJECT_ROOT / "tuner" / "data"
EVENTS_DIR = CACHE_ROOT / "events"
INDEX_PATH = CACHE_ROOT / "index.json"


def _ensure_cache_dirs(cache_root: Optional[Path] = None) -> Path:
    root = Path(cache_root) if cache_root else CACHE_ROOT
    (root / "events").mkdir(parents=True, exist_ok=True)
    return root


def _event_file_path(event_id: int, cache_root: Optional[Path] = None) -> Path:
    root = _ensure_cache_dirs(cache_root)
    return root / "events" / f"{int(event_id)}.json"


def _normalize_tracker_frame(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ep": "value",
        "points": "value",
        "score": "value",
        "pt": "value",
        "timestamp": "time",
    }
    normalized = df.rename(columns=rename_map).copy()
    if "time" not in normalized.columns or "value" not in normalized.columns:
        raise ValueError(f"tracker data missing required columns: {list(normalized.columns)}")
    normalized = normalized.sort_values("time").reset_index(drop=True)
    return normalized


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cache_index(cache_root: Optional[Path] = None) -> Dict[str, dict]:
    root = _ensure_cache_dirs(cache_root)
    index_path = root / INDEX_PATH.name
    if not index_path.exists():
        return {}
    try:
        data = _load_json(index_path)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def save_cache_index(index_payload: Dict[str, dict], cache_root: Optional[Path] = None) -> Path:
    root = _ensure_cache_dirs(cache_root)
    index_path = root / INDEX_PATH.name
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)
    return index_path


def serialize_event_pack(data_pack: dict, api_source: str) -> dict:
    meta = dict(data_pack["meta"])
    tracker_df = _normalize_tracker_frame(data_pack["dataframe"])
    actual_final_score = float(tracker_df["value"].iloc[-1])

    return {
        "event_id": int(data_pack["event_id"]),
        "meta": meta,
        "event_type": meta.get("event_type", "unknown"),
        "scale": float(data_pack["scale"]),
        "actual_final_score": actual_final_score,
        "df_records": tracker_df.to_dict(orient="records"),
        "record_count": int(len(tracker_df)),
        "fetched_at_ms": int(time.time() * 1000),
        "api_source": api_source,
    }


def save_event_payload(event_payload: dict, cache_root: Optional[Path] = None) -> Path:
    path = _event_file_path(int(event_payload["event_id"]), cache_root=cache_root)
    with path.open("w", encoding="utf-8") as f:
        json.dump(event_payload, f, ensure_ascii=False, indent=2)
    return path


def load_cached_event(event_id: int, cache_root: Optional[Path] = None) -> Optional[dict]:
    path = _event_file_path(event_id, cache_root=cache_root)
    if not path.exists():
        return None
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    payload = payload.copy()
    payload["event_id"] = int(payload["event_id"])
    payload["scale"] = float(payload["scale"])
    payload["actual_final_score"] = float(payload["actual_final_score"])
    payload["dataframe"] = pd.DataFrame(payload.get("df_records", []))
    return payload


def load_cached_events(
    cache_root: Optional[Path] = None,
    event_ids: Optional[Iterable[int]] = None,
) -> Dict[int, dict]:
    root = _ensure_cache_dirs(cache_root)
    if event_ids is None:
        paths = sorted((root / "events").glob("*.json"))
    else:
        paths = [_event_file_path(int(event_id), cache_root=root) for event_id in event_ids]

    loaded = {}
    for path in paths:
        if not path.exists():
            continue
        payload = load_cached_event(int(path.stem), cache_root=root)
        if payload is not None:
            loaded[int(payload["event_id"])] = payload
    return dict(sorted(loaded.items()))


def _iter_ended_event_ids(
    api_source: str,
    server_index: int,
    limit: int,
    min_event_id: Optional[int] = None,
    max_event_id: Optional[int] = None,
    event_types: Optional[Iterable[str]] = None,
) -> List[int]:
    ds = create_data_source(api_source=api_source, server_index=server_index)
    try:
        index_payload = ds.fetch_events_index() or {}
    finally:
        ds.close()

    wanted_types = {str(item).lower() for item in event_types} if event_types else None
    now_ms = int(time.time() * 1000)
    ended_ids = []

    for eid_str, meta in index_payload.items():
        try:
            event_id = int(eid_str)
        except (TypeError, ValueError):
            continue

        if min_event_id is not None and event_id < int(min_event_id):
            continue
        if max_event_id is not None and event_id > int(max_event_id):
            continue

        event_type = str(meta.get("eventType") or meta.get("event_type") or "unknown").lower()
        if wanted_types and event_type not in wanted_types:
            continue

        end_at = None
        for field_name in ("aggregateAt", "aggregate_at", "aggregateEndAt", "aggregate_end_at", "endAt", "end_at"):
            value = meta.get(field_name)
            if isinstance(value, list):
                if server_index >= len(value):
                    continue
                value = value[server_index]
            if value is None:
                continue
            try:
                end_at = int(value)
                break
            except (TypeError, ValueError):
                continue

        if end_at is None or end_at >= now_ms:
            continue

        ended_ids.append(event_id)

    ended_ids.sort(reverse=True)
    return ended_ids[:limit]


def cache_historical_events(
    history_count: int = 100,
    api_source: str = DEFAULT_API_SOURCE,
    server_index: int = DEFAULT_SERVER,
    cache_root: Optional[Path] = None,
    refresh: bool = False,
    min_event_id: Optional[int] = None,
    max_event_id: Optional[int] = None,
    event_types: Optional[Iterable[str]] = None,
) -> dict:
    root = _ensure_cache_dirs(cache_root)
    index_payload = load_cache_index(root)
    cached_before = len(index_payload)

    candidate_ids = _iter_ended_event_ids(
        api_source=api_source,
        server_index=server_index,
        limit=history_count,
        min_event_id=min_event_id,
        max_event_id=max_event_id,
        event_types=event_types,
    )

    ds = create_data_source(api_source=api_source, server_index=server_index)
    fetched = 0
    reused = 0
    failed = []

    try:
        for event_id in candidate_ids:
            event_file = _event_file_path(event_id, cache_root=root)
            if event_file.exists() and not refresh:
                existing = load_cached_event(event_id, cache_root=root)
                if existing is not None:
                    index_payload[str(event_id)] = {
                        "event_id": event_id,
                        "event_type": existing.get("event_type") or existing.get("meta", {}).get("event_type", "unknown"),
                        "actual_final_score": existing.get("actual_final_score"),
                        "scale": existing.get("scale"),
                        "record_count": existing.get("record_count", len(existing["dataframe"])),
                    }
                    reused += 1
                    continue

            data_pack = ds.fetch_event_data_pack(event_id)
            if not data_pack or not data_pack.get("scale"):
                failed.append(event_id)
                continue

            try:
                payload = serialize_event_pack(data_pack, api_source=api_source)
                save_event_payload(payload, cache_root=root)
            except Exception as exc:
                logger.warning("failed to cache event %s: %s", event_id, exc)
                failed.append(event_id)
                continue

            index_payload[str(event_id)] = {
                "event_id": int(payload["event_id"]),
                "event_type": payload["event_type"],
                "actual_final_score": payload["actual_final_score"],
                "scale": payload["scale"],
                "record_count": payload["record_count"],
            }
            fetched += 1
    finally:
        ds.close()

    save_cache_index(index_payload, cache_root=root)
    summary = {
        "cache_root": str(root),
        "requested": history_count,
        "candidate_ids": candidate_ids,
        "fetched": fetched,
        "reused": reused,
        "failed": failed,
        "cached_total_before": cached_before,
        "cached_total_after": len(index_payload),
    }
    logger.info("cache_historical_events summary: %s", summary)
    return summary
