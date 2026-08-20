"""Incrementally collect public fixed-tier cutoffs with first-seen versions.

``tier_records`` remains a latest-value compatibility view.
``tier_record_versions`` keeps each distinct revision plus the millisecond
when this collector first observed it.  Legacy rows without such versions stay
timestamp-only; the collector never invents a historical availability time.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ALL_TRACKER_TIERS,
    DEFAULT_API_SOURCE,
    DEFAULT_SERVER,
    canonicalize_tracker_tiers,
    validate_tracker_tier,
)
from data_source import create_data_source

DEFAULT_CACHE_DIR = PROJECT_ROOT / "event_data" / "tier_surface_cache"
VERSION_AVAILABILITY_STATUS = "explicit_row_level_first_seen_at"
FINAL_REFRESH_GRACE_MS = 20 * 60 * 1000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _event_file_lock(path: Path):
    """Cross-process lock for one cache read/merge/replace transaction."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def resolve_event_ids(event_ids: Optional[Iterable[int]], min_event_id: Optional[int], max_event_id: Optional[int]) -> list[int]:
    """Resolve an explicit list or a closed event-id range without guessing scope."""
    if event_ids:
        return sorted(set(int(event_id) for event_id in event_ids))
    if min_event_id is None or max_event_id is None:
        raise ValueError("Provide --event-ids or both --min-event-id and --max-event-id")
    low, high = int(min_event_id), int(max_event_id)
    if low < 0 or high < low:
        raise ValueError("Event-id range must satisfy 0 <= min-event-id <= max-event-id")
    return list(range(low, high + 1))


def _cache_path(cache_dir: Path, event_id: int) -> Path:
    return cache_dir / f"{int(event_id)}.json"


def read_cache(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot safely update existing cache {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Cannot safely update existing cache {path}: root must be an object")
    return payload


def _validate_cache_identity(
    payload: dict[str, Any],
    *,
    requested_event_id: int,
    path: Path,
) -> None:
    if not payload:
        return
    try:
        payload_event_id = int(payload["event_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot safely update {path}: missing/invalid event_id"
        ) from exc
    if payload_event_id != int(requested_event_id):
        raise ValueError(
            f"Cannot safely update {path}: payload event_id "
            f"{payload_event_id} != requested {requested_event_id}"
        )
    meta = payload.get("meta")
    if isinstance(meta, dict) and "event_id" in meta:
        try:
            meta_event_id = int(meta["event_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot safely update {path}: invalid meta.event_id"
            ) from exc
        if meta_event_id != int(requested_event_id):
            raise ValueError(
                f"Cannot safely update {path}: meta.event_id "
                f"{meta_event_id} != requested {requested_event_id}"
            )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one event cache; an exception leaves the old file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
        ) as tmp:
            tmp_name = tmp.name
            json.dump(payload, tmp, ensure_ascii=False, separators=(",", ":"))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _records_from_frame(frame: Optional[pd.DataFrame]) -> Optional[list[dict[str, Any]]]:
    if frame is None or frame.empty:
        return None
    # DataFrame.to_dict preserves the public response field names (normally
    # time/ep), and deliberately does not add collection timestamps to rows.
    records = frame.to_dict(orient="records")
    return records if records else None


def _normalize_reward_result(
    result: Any,
    *,
    event_id: int,
    requested_tiers: list[int],
) -> tuple[list[int], dict[str, int]]:
    """Validate one reward response and return its canonical cache fields."""
    if not isinstance(result, dict):
        raise ValueError(f"Event {event_id}: reward metadata response is missing")
    if "target_tiers" not in result or "last_appearance" not in result:
        raise ValueError(
            f"Event {event_id}: reward metadata needs target_tiers and last_appearance"
        )
    raw_tiers = result["target_tiers"]
    if not isinstance(raw_tiers, list):
        raise ValueError(f"Event {event_id}: target_tiers must be a list")
    try:
        reward_tiers = canonicalize_tracker_tiers(raw_tiers)
    except ValueError as exc:
        raise ValueError(f"Event {event_id}: invalid reward target tier: {exc}") from exc
    raw_last_appearance = result["last_appearance"]
    if not isinstance(raw_last_appearance, dict):
        raise ValueError(f"Event {event_id}: last_appearance must be an object")
    last_appearance: dict[str, int] = {}
    for reward_key, raw_tier in raw_last_appearance.items():
        if not isinstance(reward_key, str) or not reward_key:
            raise ValueError(
                f"Event {event_id}: last_appearance keys must be non-empty strings"
            )
        try:
            last_appearance[reward_key] = validate_tracker_tier(raw_tier)
        except ValueError as exc:
            raise ValueError(
                f"Event {event_id}: invalid last_appearance tier for {reward_key}: {exc}"
            ) from exc
    last_appearance = dict(sorted(last_appearance.items()))
    derived_tiers = canonicalize_tracker_tiers(
        sorted(set(last_appearance.values()))
    )
    if reward_tiers != derived_tiers:
        raise ValueError(
            f"Event {event_id}: target_tiers {reward_tiers} disagree with "
            f"last_appearance tiers {derived_tiers}"
        )

    requested = {validate_tracker_tier(tier) for tier in requested_tiers}
    unrequested = [tier for tier in reward_tiers if tier not in requested]
    if unrequested:
        raise ValueError(
            f"Event {event_id}: reward target tiers {unrequested} were not "
            f"requested from the tracker"
        )
    return reward_tiers, last_appearance


def _existing_reward_metadata_state(
    payload: dict[str, Any],
    *,
    event_id: int,
    requested_tiers: list[int],
    expected_reward_tiers: list[int],
    expected_last_appearance: dict[str, int],
    source: str,
    server: int,
    replace_legacy_reward_metadata: bool,
) -> str:
    """Classify cached reward metadata; malformed/conflicting data fails."""
    has_tiers = "reward_tiers" in payload
    has_provenance = "reward_tier_provenance" in payload
    if not has_tiers and not has_provenance:
        return "missing"
    if has_tiers and not has_provenance:
        if not replace_legacy_reward_metadata:
            raise ValueError(
                f"Event {event_id}: cached reward metadata is partial; both "
                "reward_tiers and reward_tier_provenance are required"
            )
        raw_legacy_tiers = payload["reward_tiers"]
        if not isinstance(raw_legacy_tiers, list):
            raise ValueError(
                f"Event {event_id}: legacy reward_tiers must be a list"
            )
        # Old caches predate reward provenance and some were populated by the
        # former Graph extractor, which mistook title-effect rows for actual
        # tracker cutoffs (for example T2/T3).  The explicit replacement flag
        # means the fetched, fully validated HHWX payload is authoritative;
        # requiring the untrusted legacy value itself to satisfy the current
        # tracker schema would make that corruption impossible to repair.
        return "replace_legacy"
    if not has_tiers and has_provenance:
        raise ValueError(
            f"Event {event_id}: cached reward metadata is partial; both "
            "reward_tiers and reward_tier_provenance are required"
        )

    provenance = payload["reward_tier_provenance"]
    if not isinstance(provenance, dict):
        raise ValueError(
            f"Event {event_id}: cached reward_tier_provenance must be an object"
        )
    cached_tiers, cached_last = _normalize_reward_result(
        {
            "target_tiers": payload["reward_tiers"],
            "last_appearance": provenance.get("last_appearance"),
        },
        event_id=event_id,
        requested_tiers=requested_tiers,
    )
    if payload["reward_tiers"] != cached_tiers:
        raise ValueError(f"Event {event_id}: cached reward_tiers are not normalized")
    if provenance.get("last_appearance") != cached_last:
        raise ValueError(
            f"Event {event_id}: cached reward last_appearance is not normalized"
        )
    observed_at = provenance.get("observed_at")
    if (
        isinstance(observed_at, bool)
        or not isinstance(observed_at, int)
        or observed_at <= 0
    ):
        raise ValueError(
            f"Event {event_id}: cached reward observed_at must be a positive integer"
        )
    if provenance.get("source") != str(source):
        raise ValueError(f"Event {event_id}: cached reward source conflicts")
    if (
        isinstance(provenance.get("server"), bool)
        or not isinstance(provenance.get("server"), int)
        or provenance.get("server") != int(server)
    ):
        raise ValueError(f"Event {event_id}: cached reward server conflicts")
    if (
        cached_tiers != expected_reward_tiers
        or cached_last != expected_last_appearance
    ):
        raise ValueError(f"Event {event_id}: cached reward metadata conflicts")
    return "matching"


def _fetch_tiers(data_source, event_id: int, tiers: list[int], workers: int, wait_ms: int) -> dict[int, Optional[list[dict[str, Any]]]]:
    """Fetch selected fixed tiers with bounded request concurrency."""
    results: dict[int, Optional[list[dict[str, Any]]]] = {tier: None for tier in tiers}
    if not tiers:
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_tier = {}
        for tier in tiers:
            future_to_tier[executor.submit(data_source.fetch_tier_data, int(event_id), int(tier))] = tier
            if wait_ms > 0:
                time.sleep(wait_ms / 1000.0)
        for future in as_completed(future_to_tier):
            tier = future_to_tier[future]
            try:
                results[tier] = _records_from_frame(future.result())
            except Exception:
                results[tier] = None
    return results


def _event_is_live(meta: dict[str, Any], now_ms: Optional[int] = None) -> bool:
    """Refresh existing rows until the event's last advertised finalization time."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    candidates = []
    for key in ("aggregate_at", "aggregateAt", "end_at", "endAt"):
        try:
            value = int(meta.get(key))
        except (AttributeError, TypeError, ValueError):
            continue
        if value > 0:
            candidates.append(value)
    return bool(candidates) and int(now_ms) <= (
        max(candidates) + FINAL_REFRESH_GRACE_MS
    )


def _record_key(record: dict[str, Any], ordinal: int) -> tuple[str, Any]:
    for field in ("time", "timestamp"):
        if field in record:
            try:
                return field, int(record[field])
            except (TypeError, ValueError):
                return field, str(record[field])
    # Rows without a tracker timestamp cannot safely replace one another.
    return "row", ordinal


def _merge_tracker_records(
    existing: Optional[list[dict[str, Any]]],
    fetched: Optional[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Append new observations and retain the newest revision at each timestamp."""
    merged: dict[tuple[str, Any], dict[str, Any]] = {}
    ordinal = 0
    for collection in (existing or [], fetched or []):
        if not isinstance(collection, list):
            continue
        for record in collection:
            if not isinstance(record, dict):
                continue
            merged[_record_key(record, ordinal)] = dict(record)
            ordinal += 1
    rows = list(merged.values())
    rows.sort(
        key=lambda row: (
            int(row.get("time", row.get("timestamp", 0)))
            if str(row.get("time", row.get("timestamp", 0))).lstrip("-").isdigit()
            else 0
        )
    )
    return rows


def _raw_version_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"available_at", "observed_at", "source"}
    }


def _merge_version_records(
    existing: Any,
    fetched: Optional[list[dict[str, Any]]],
    *,
    observed_at: int,
    source: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Keep every distinct revision and its first observed time."""
    versions = [
        dict(record)
        for record in (existing if isinstance(existing, list) else [])
        if isinstance(record, dict)
    ]
    identities = {
        json.dumps(
            _raw_version_record(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in versions
    }
    changed = False
    for record in fetched or []:
        if not isinstance(record, dict):
            continue
        raw_record = _raw_version_record(record)
        identity = json.dumps(
            raw_record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if identity in identities:
            continue
        versions.append({
            **raw_record,
            "available_at": int(observed_at),
            "source": str(source),
        })
        identities.add(identity)
        changed = True
    versions.sort(
        key=lambda row: (
            int(row.get("time", row.get("timestamp", 0)))
            if str(row.get("time", row.get("timestamp", 0))).lstrip("-").isdigit()
            else 0,
            int(row.get("available_at", -1))
            if str(row.get("available_at", -1)).lstrip("-").isdigit()
            else -1,
        )
    )
    return versions, changed


def _collection_metadata(
    source: str,
    server: int,
    requested_tiers: list[int],
    added_tiers: list[int],
    refreshed_tiers: list[int],
    changed_tiers: list[int],
    failed_tiers: list[int],
) -> dict[str, Any]:
    # fetched_at describes the run.  Per-revision first-seen millisecond
    # timestamps live in tier_record_versions[*].available_at.
    return {
        "fetched_at": _utc_now(),
        "source": str(source),
        "server": int(server),
        "requested_tiers": [int(tier) for tier in requested_tiers],
        "added_tiers": [int(tier) for tier in added_tiers],
        "refreshed_tiers": [int(tier) for tier in refreshed_tiers],
        "changed_tiers": [int(tier) for tier in changed_tiers],
        "failed_tiers": [int(tier) for tier in failed_tiers],
        "availability_status": VERSION_AVAILABILITY_STATUS,
        "available_at": None,
    }


def collect_event(
    *,
    event_id: int,
    cache_dir: Path,
    tiers: list[int],
    api_source: str,
    server: int,
    workers: int,
    wait_ms: int,
    dry_run: bool,
    refresh_existing: bool,
    replace_legacy_reward_metadata: bool,
    data_source,
) -> dict[str, Any]:
    """Fill missing tiers and append fresh rows while an event is still live."""
    tiers = canonicalize_tracker_tiers(tiers)
    path = _cache_path(cache_dir, event_id)
    existing = read_cache(path)
    _validate_cache_identity(
        existing, requested_event_id=event_id, path=path
    )
    existing_records = existing.get("tier_records") if isinstance(existing.get("tier_records"), dict) else {}
    missing_tiers = [tier for tier in tiers if not existing_records.get(str(tier))]
    meta = existing.get("meta") if isinstance(existing.get("meta"), dict) else None
    live = bool(meta and _event_is_live(meta))
    should_refresh = live or bool(refresh_existing)
    refresh_tiers = [
        tier for tier in tiers
        if should_refresh and existing_records.get(str(tier))
    ]
    fetch_tiers = missing_tiers + [
        tier for tier in refresh_tiers if tier not in missing_tiers
    ]

    if dry_run:
        return {
            "event_id": int(event_id),
            "status": "dry_run",
            "path": str(path),
            "existing_tiers": sorted(int(key) for key in existing_records if str(key).isdigit()),
            "event_is_live": live,
            "forced_refresh": bool(refresh_existing),
            "replace_legacy_reward_metadata": bool(
                replace_legacy_reward_metadata
            ),
            "would_fetch_reward_metadata": True,
            "would_fetch_tiers": fetch_tiers,
            "would_refresh_tiers": refresh_tiers,
        }

    if meta is None:
        meta = data_source.fetch_event_meta(event_id)
        if not meta:
            return {"event_id": int(event_id), "status": "failed_meta", "path": str(path)}
        live = _event_is_live(meta)
        should_refresh = live or bool(refresh_existing)
        refresh_tiers = [
            tier for tier in tiers
            if should_refresh and existing_records.get(str(tier))
        ]
        fetch_tiers = missing_tiers + [
            tier for tier in refresh_tiers if tier not in missing_tiers
        ]

    reward_result = data_source.fetch_event_rewards(event_id)
    reward_observed_at = int(time.time() * 1000)
    reward_tiers, reward_last_appearance = _normalize_reward_result(
        reward_result,
        event_id=event_id,
        requested_tiers=tiers,
    )
    existing_reward_state = _existing_reward_metadata_state(
        existing,
        event_id=event_id,
        requested_tiers=tiers,
        expected_reward_tiers=reward_tiers,
        expected_last_appearance=reward_last_appearance,
        source=api_source,
        server=server,
        replace_legacy_reward_metadata=replace_legacy_reward_metadata,
    )

    if not fetch_tiers and existing_reward_state == "matching":
        return {
            "event_id": int(event_id),
            "status": "already_complete",
            "path": str(path),
            "added_tiers": [],
            "refreshed_tiers": [],
            "changed_tiers": [],
            "failed_tiers": [],
            "reward_metadata_changed": False,
        }

    fetched = _fetch_tiers(data_source, event_id, fetch_tiers, workers, wait_ms)
    successful_tiers = [tier for tier, records in fetched.items() if records]
    failed_tiers = [tier for tier in fetch_tiers if tier not in successful_tiers]

    # A fully failed incremental attempt should not even rewrite metadata: the
    # previous cache remains byte-for-byte usable.
    if fetch_tiers and not successful_tiers:
        return {
            "event_id": int(event_id),
            "status": "failed_tiers",
            "path": str(path),
            "failed_tiers": failed_tiers,
        }

    observed_at = int(time.time() * 1000)
    with _event_file_lock(path):
        # Another collector may have completed while requests were in flight.
        # Re-read under the lock and merge against that latest state.
        latest = read_cache(path)
        _validate_cache_identity(
            latest, requested_event_id=event_id, path=path
        )
        latest_records = (
            latest.get("tier_records")
            if isinstance(latest.get("tier_records"), dict)
            else {}
        )
        latest_versions = (
            latest.get("tier_record_versions")
            if isinstance(latest.get("tier_record_versions"), dict)
            else {}
        )
        latest_reward_state = _existing_reward_metadata_state(
            latest,
            event_id=event_id,
            requested_tiers=tiers,
            expected_reward_tiers=reward_tiers,
            expected_last_appearance=reward_last_appearance,
            source=api_source,
            server=server,
            replace_legacy_reward_metadata=replace_legacy_reward_metadata,
        )
        reward_metadata_changed = latest_reward_state in {
            "missing", "replace_legacy"
        }
        reward_metadata_replaced_legacy = (
            latest_reward_state == "replace_legacy"
        )
        updated = dict(latest)
        updated["event_id"] = int(event_id)
        if not isinstance(updated.get("meta"), dict):
            updated["meta"] = meta
        if reward_metadata_changed:
            updated["reward_tiers"] = reward_tiers
            updated["reward_tier_provenance"] = {
                "source": str(api_source),
                "server": int(server),
                "last_appearance": reward_last_appearance,
                "observed_at": reward_observed_at,
            }
        updated_records = dict(latest_records)
        updated_versions = dict(latest_versions)
        added_tiers = []
        refreshed_tiers = []
        changed_tiers = []
        for tier in successful_tiers:
            old_records = latest_records.get(str(tier))
            if old_records:
                refreshed_tiers.append(tier)
            else:
                added_tiers.append(tier)
            merged_records = _merge_tracker_records(
                old_records, fetched[tier]
            )
            merged_versions, versions_changed = _merge_version_records(
                latest_versions.get(str(tier)),
                list(old_records or []) + list(fetched[tier] or []),
                observed_at=observed_at,
                source=api_source,
            )
            updated_records[str(tier)] = merged_records
            updated_versions[str(tier)] = merged_versions
            if (
                merged_records != (old_records or [])
                or versions_changed
            ):
                changed_tiers.append(tier)
        updated["tier_records"] = updated_records
        updated["tier_record_versions"] = updated_versions
        updated["schema_version"] = max(
            int(updated.get("schema_version", 1) or 1), 3
        )

        prior_metadata = updated.get("collection_metadata")
        history = []
        if isinstance(prior_metadata, dict):
            history = list(prior_metadata.get("history") or [])
        run_metadata = _collection_metadata(
            api_source,
            server,
            tiers,
            added_tiers,
            refreshed_tiers,
            changed_tiers,
            failed_tiers,
        )
        updated["collection_metadata"] = {
            **run_metadata,
            "history": history + [run_metadata],
        }
        atomic_write_json(path, updated)
    return {
        "event_id": int(event_id),
        "status": (
            "updated"
            if changed_tiers or reward_metadata_changed
            else "refreshed_no_change"
        ),
        "path": str(path),
        "added_tiers": added_tiers,
        "refreshed_tiers": refreshed_tiers,
        "changed_tiers": changed_tiers,
        "failed_tiers": failed_tiers,
        "reward_metadata_changed": reward_metadata_changed,
        "reward_metadata_replaced_legacy": reward_metadata_replaced_legacy,
    }


def collect_events(
    args: argparse.Namespace,
    data_source_factory: Callable[..., Any] = create_data_source,
) -> list[dict[str, Any]]:
    if not 1 <= int(args.workers) <= 4:
        raise ValueError("--workers must be between 1 and 4")
    if int(args.wait_ms) < 0:
        raise ValueError("--wait-ms must be non-negative")
    event_ids = resolve_event_ids(args.event_ids, args.min_event_id, args.max_event_id)
    tiers = canonicalize_tracker_tiers(args.tiers)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    if args.dry_run:
        return [
            collect_event(
                event_id=event_id, cache_dir=cache_dir, tiers=tiers,
                api_source=args.api_source, server=args.server, workers=args.workers,
                wait_ms=args.wait_ms, dry_run=True,
                refresh_existing=bool(getattr(args, "refresh_existing", False)),
                replace_legacy_reward_metadata=bool(
                    getattr(args, "replace_legacy_reward_metadata", False)
                ),
                data_source=None,
            )
            for event_id in event_ids
        ]

    data_source = data_source_factory(args.api_source, server_index=args.server)
    try:
        return [
            collect_event(
                event_id=event_id, cache_dir=cache_dir, tiers=tiers,
                api_source=args.api_source, server=args.server, workers=args.workers,
                wait_ms=args.wait_ms, dry_run=False,
                refresh_existing=bool(getattr(args, "refresh_existing", False)),
                replace_legacy_reward_metadata=bool(
                    getattr(args, "replace_legacy_reward_metadata", False)
                ),
                data_source=data_source,
            )
            for event_id in event_ids
        ]
    finally:
        close = getattr(data_source, "close", None)
        if callable(close):
            close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-source", choices=["bestdori", "hhwx"], default=DEFAULT_API_SOURCE)
    parser.add_argument("--server", type=int, default=DEFAULT_SERVER)
    parser.add_argument("--event-ids", type=int, nargs="+", default=None)
    parser.add_argument("--min-event-id", type=int, default=None)
    parser.add_argument("--max-event-id", type=int, default=None)
    parser.add_argument("--tiers", type=int, nargs="+", default=list(ALL_TRACKER_TIERS))
    parser.add_argument("--wait-ms", type=int, default=250)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help=(
            "re-fetch existing tiers even after the automatic end+20m "
            "refresh window"
        ),
    )
    parser.add_argument(
        "--replace-legacy-reward-metadata",
        action="store_true",
        help=(
            "replace a normalized legacy reward_tiers field only when the "
            "cache has no reward_tier_provenance; complete conflicting "
            "metadata still fails closed"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = collect_events(args)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
