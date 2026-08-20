import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from config import (
    API_SOURCE_CONFIGS,
    DEFAULT_API_SOURCE,
    DEFAULT_FALLBACK_API_SOURCE,
    DEFAULT_SERVER,
    EVENTS_INDEX_CACHE_TTL_SECONDS,
    LIVE_SCALE_CACHE_TTL_SECONDS,
    ALL_TRACKER_TIERS,
    canonicalize_tracker_tiers,
    validate_tracker_tier,
)

logger = logging.getLogger('predictor.datasource')

SERVER_KEYS_BY_INDEX = ['jp', 'tw', 'en', 'cn', 'kr']
_GLOBAL_SCALE_CACHE = {}
_GLOBAL_SCALE_CACHE_LOCK = threading.RLock()
_GLOBAL_SCALE_KEY_LOCKS = {}
_GLOBAL_EVENTS_INDEX_CACHE = {}
_GLOBAL_EVENTS_INDEX_LOCK = threading.RLock()


@dataclass(frozen=True)
class ScaleObservation:
    """A scale value together with the provenance of the bytes that made it."""

    value: float | None
    source: str
    fetched_at: int
    available_at: int | None
    availability_status: str
    origin_as_of: int | None
    fallback_used: bool
    cache_hit: bool = False
    cache_expires_at: int | None = None
    primary_error: str | None = None
    fallback_error: str | None = None


@dataclass(frozen=True)
class _TimedCacheEntry:
    value: object
    expires_monotonic: float | None

    def is_fresh(self, now_monotonic: float) -> bool:
        return (
            self.expires_monotonic is None
            or now_monotonic < self.expires_monotonic
        )


def _get_scale_key_lock(cache_key):
    with _GLOBAL_SCALE_CACHE_LOCK:
        lock = _GLOBAL_SCALE_KEY_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _GLOBAL_SCALE_KEY_LOCKS[cache_key] = lock
        return lock


def _normalize_hhwx_events_payload(payload):
    """Unwrap the HHWX master/events response into an event-id mapping."""
    if not isinstance(payload, dict) or payload.get('success') is not True:
        raise ValueError("HHWX events response must contain success=true")
    data = payload.get('data')
    if not isinstance(data, dict):
        raise ValueError("HHWX events response data must be an object")

    # Keep parsing the previously supported HHWX list representation for
    # callers holding an in-memory response from before the endpoint move.
    events = data.get('events')
    if not isinstance(events, list):
        normalized = {}
        for event_id, metadata in data.items():
            if not str(event_id).isdigit() or not isinstance(metadata, dict):
                raise ValueError("HHWX events data must map numeric ids to objects")
            normalized[str(int(event_id))] = metadata
        if not normalized:
            raise ValueError("HHWX events data is empty")
        return normalized

    normalized = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        eid = ev.get('eventId')
        if eid is None:
            continue
        timeline = ev.get('timeline') or {}
        starts = [None] * len(SERVER_KEYS_BY_INDEX)
        ends = [None] * len(SERVER_KEYS_BY_INDEX)
        for idx, key in enumerate(SERVER_KEYS_BY_INDEX):
            slot = timeline.get(key)
            if isinstance(slot, dict):
                starts[idx] = slot.get('startAt')
                ends[idx] = slot.get('endAt')
        normalized[str(eid)] = {
            'startAt': starts,
            'endAt': ends,
            'eventType': ev.get('eventType'),
        }
    return normalized


def _unwrap_hhwx_event_detail(payload, event_id):
    """Return the current HHWX detail body, rejecting wrapper mismatches."""
    if not isinstance(payload, dict) or payload.get('success') is not True:
        raise ValueError(
            f"HHWX event {int(event_id)} detail must contain success=true"
        )
    detail = payload.get('data')
    if not isinstance(detail, dict):
        raise ValueError(f"HHWX event {int(event_id)} detail data must be an object")
    return detail


def resolve_api_source_config(api_source):
    source_key = (api_source or DEFAULT_API_SOURCE).lower()
    if source_key not in API_SOURCE_CONFIGS:
        raise ValueError(f"Unknown api_source '{api_source}'. Available: {list(API_SOURCE_CONFIGS.keys())}")
    return source_key, API_SOURCE_CONFIGS[source_key]


class BandoriDataSource:
    """
    按 API profile 驱动的数据源客户端。
    统一封装 event meta / tracker / scale / similar event 扫描逻辑。
    """
    def __init__(
        self,
        api_source=None,
        server_index=DEFAULT_SERVER,
        allow_fallback=False,
        fallback_api_source=DEFAULT_FALLBACK_API_SOURCE,
    ):
        self.api_source, self.api_config = resolve_api_source_config(api_source)
        self.server_index = server_index
        self.allow_fallback = bool(allow_fallback)
        self.fallback_api_source = str(fallback_api_source).lower()
        if self.fallback_api_source not in API_SOURCE_CONFIGS:
            raise ValueError(
                f"Unknown fallback_api_source '{fallback_api_source}'. "
                f"Available: {list(API_SOURCE_CONFIGS.keys())}"
            )
        self.session = requests.Session()
        try:
            adapter = HTTPAdapter(pool_connections=10, pool_maxsize=30, max_retries=3)
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
        except Exception as e:
            logger.warning(f"Failed to mount adapter: {e}")
        self._events_index_cache_by_source: dict = {}
        self._meta_cache: dict = {}
        self._scale_cache: dict = {}
        self._operation_provenance: dict = {}

    def close(self):
        """关闭 Session 资源"""
        if self.session:
            self.session.close()

    def _fallback_enabled(self, allow_fallback=None):
        enabled = self.allow_fallback if allow_fallback is None else bool(allow_fallback)
        return (
            enabled
            and self.api_source != self.fallback_api_source
            and self.fallback_api_source in API_SOURCE_CONFIGS
        )

    def _source_candidates(self, allow_fallback=None):
        yield self.api_source, self.api_config, False
        if self._fallback_enabled(allow_fallback):
            yield (
                self.fallback_api_source,
                API_SOURCE_CONFIGS[self.fallback_api_source],
                True,
            )

    def _record_provenance(
        self,
        operation,
        *,
        source,
        fallback_used,
        primary_error=None,
        cache_hit=False,
    ):
        record = {
            "operation": str(operation),
            "requested_source": self.api_source,
            "source": str(source),
            "fallback_used": bool(fallback_used),
            "primary_error": str(primary_error) if primary_error else None,
            "cache_hit": bool(cache_hit),
            "fetched_at": int(time.time() * 1000),
        }
        self._operation_provenance[str(operation)] = record
        return record

    def get_provenance(self, operation=None):
        """Return a copy of the latest source-routing evidence."""
        if operation is None:
            return {
                key: value.copy()
                for key, value in self._operation_provenance.items()
            }
        record = self._operation_provenance.get(str(operation))
        return record.copy() if record else None

    def _get_json(self, url, timeout=10, suppress_log=False, retry=True):
        try:
            if retry:
                response = self.session.get(url, timeout=timeout)
            else:
                response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log = logger.debug if suppress_log else logger.warning
            log(f"[{self.api_source}] Request failed: {url} | {e}")
            return None

    def _extract_server_timestamp(self, meta, *field_names):
        for field_name in field_names:
            value = meta.get(field_name)
            if value is None:
                continue

            if isinstance(value, list):
                if self.server_index >= len(value):
                    continue
                value = value[self.server_index]

            if value is None:
                continue

            try:
                return int(value)
            except (TypeError, ValueError):
                continue

        return None

    def _fetch_events_index_from_source(self, source_key, api_config, timeout=8):
        """Fetch one concrete source without recursively invoking fallback."""
        now_monotonic = time.monotonic()
        local_entry = self._events_index_cache_by_source.get(source_key)
        if isinstance(local_entry, _TimedCacheEntry) and local_entry.is_fresh(now_monotonic):
            return local_entry.value, True
        self._events_index_cache_by_source.pop(source_key, None)

        cache_key = (str(source_key), int(self.server_index))
        with _GLOBAL_EVENTS_INDEX_LOCK:
            cache_entry = _GLOBAL_EVENTS_INDEX_CACHE.get(cache_key)
            if (
                isinstance(cache_entry, _TimedCacheEntry)
                and cache_entry.is_fresh(now_monotonic)
            ):
                self._events_index_cache_by_source[source_key] = cache_entry
                return cache_entry.value, True
            _GLOBAL_EVENTS_INDEX_CACHE.pop(cache_key, None)

        url = api_config['event_index_url']
        payload = self._get_json(url, timeout=timeout)
        if source_key == 'hhwx':
            if payload is None:
                raise RuntimeError(f"HHWX events request failed: {url}")
            result = _normalize_hhwx_events_payload(payload)
        else:
            result = payload
        if not isinstance(result, dict) or not result:
            raise ValueError(f"{str(source_key).upper()} events index is empty or invalid: {url}")

        expires_monotonic = time.monotonic() + EVENTS_INDEX_CACHE_TTL_SECONDS
        cache_entry = _TimedCacheEntry(
            value=result,
            expires_monotonic=expires_monotonic,
        )
        self._events_index_cache_by_source[source_key] = cache_entry
        with _GLOBAL_EVENTS_INDEX_LOCK:
            _GLOBAL_EVENTS_INDEX_CACHE[cache_key] = cache_entry
        return result, False

    def fetch_events_index(self, timeout=8, allow_fallback=None):
        primary_error = None
        attempts = []
        for source_key, api_config, fallback_used in self._source_candidates(allow_fallback):
            try:
                result, cache_hit = self._fetch_events_index_from_source(
                    source_key,
                    api_config,
                    timeout=timeout,
                )
            except Exception as exc:
                detail = f"{str(source_key).upper()}: {type(exc).__name__}: {exc}"
                attempts.append(detail)
                if not fallback_used:
                    primary_error = detail
                continue

            self._record_provenance(
                "events_index",
                source=source_key,
                fallback_used=fallback_used,
                primary_error=primary_error,
                cache_hit=cache_hit,
            )
            return result

        raise RuntimeError("events index unavailable; " + "; ".join(attempts))

    def _select_current_event_id(self, all_events, server_index):
        now_ms = int(time.time() * 1000)
        active_candidates = []
        nearest_candidate = None
        nearest_dist = float('inf')

        for eid_str, meta in all_events.items():
            if not isinstance(meta, dict):
                raise ValueError(f"event {eid_str} metadata must be an object")
            starts = meta.get('startAt') or meta.get('start_at')
            ends = meta.get('endAt') or meta.get('end_at')
            if not isinstance(starts, list) or not isinstance(ends, list):
                raise ValueError(f"event {eid_str} metadata needs timestamp arrays")
            if server_index >= len(starts) or server_index >= len(ends):
                continue

            start = starts[server_index]
            end = ends[server_index]
            if start is None or end is None:
                continue

            try:
                start = int(start)
                end = int(end)
                event_id = int(eid_str)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"event {eid_str} has invalid timestamps") from exc

            if start <= now_ms <= end:
                active_candidates.append((event_id, start))
            else:
                dist = min(abs(start - now_ms), abs(end - now_ms))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_candidate = event_id

        if active_candidates:
            active_candidates.sort(key=lambda item: item[1], reverse=True)
            return active_candidates[0][0]

        if nearest_candidate is None:
            raise RuntimeError(
                f"events index has no timestamps for server {server_index}"
            )
        return nearest_candidate

    def get_current_event_id(self, server_index=None, allow_fallback=None):
        """
        从 events index 推断当前活动 ID。
        请求成功但目标服时间轴不可用也属于主源失败。
        """
        if server_index is None:
            server_index = self.server_index

        primary_error = None
        attempts = []
        for source_key, api_config, fallback_used in self._source_candidates(allow_fallback):
            try:
                all_events, cache_hit = self._fetch_events_index_from_source(
                    source_key,
                    api_config,
                    timeout=8,
                )
                selected = self._select_current_event_id(all_events, server_index)
            except Exception as exc:
                detail = f"{str(source_key).upper()}: {type(exc).__name__}: {exc}"
                attempts.append(detail)
                if not fallback_used:
                    primary_error = detail
                continue

            for operation in ("events_index", "current_event"):
                self._record_provenance(
                    operation,
                    source=source_key,
                    fallback_used=fallback_used,
                    primary_error=primary_error,
                    cache_hit=cache_hit,
                )
            return selected

        raise RuntimeError("current event unavailable; " + "; ".join(attempts))

    def _fetch_event_metadata_from_source(self, source_key, api_config, event_id):
        if source_key == 'hhwx':
            all_events, _ = self._fetch_events_index_from_source(
                source_key,
                api_config,
                timeout=8,
            )
            metadata = all_events.get(str(int(event_id)))
            endpoint = api_config['event_index_url']
        else:
            url_template = api_config.get('event_meta_url')
            if not url_template:
                raise RuntimeError(f"{str(source_key).upper()} event detail URL is not configured")
            endpoint = url_template.format(event_id=event_id)
            metadata = self._get_json(endpoint, timeout=5)

        if not isinstance(metadata, dict) or not metadata:
            raise ValueError(
                f"{str(source_key).upper()} event {int(event_id)} metadata is empty: {endpoint}"
            )
        return metadata

    def fetch_event_meta(self, event_id, allow_fallback=None):
        primary_error = None
        attempts = []
        for source_key, api_config, fallback_used in self._source_candidates(allow_fallback):
            cache_key = (str(source_key), int(event_id))
            cached = self._meta_cache.get(cache_key)
            try:
                metadata = (
                    cached
                    if isinstance(cached, dict)
                    else self._fetch_event_metadata_from_source(
                        source_key,
                        api_config,
                        event_id,
                    )
                )

                start_at = self._extract_server_timestamp(metadata, 'startAt', 'start_at')
                end_at = self._extract_server_timestamp(metadata, 'endAt', 'end_at')
                aggregate_at = self._extract_server_timestamp(
                    metadata,
                    'aggregateAt',
                    'aggregate_at',
                    'aggregateEndAt',
                    'aggregate_end_at',
                )
                if start_at is None or end_at is None:
                    raise ValueError(
                        f"{str(source_key).upper()} event {int(event_id)} metadata "
                        "has no usable server timestamps"
                    )
            except Exception as exc:
                detail = f"{str(source_key).upper()}: {type(exc).__name__}: {exc}"
                attempts.append(detail)
                if not fallback_used:
                    primary_error = detail
                continue

            result = {
                'event_id': int(event_id),
                'start_at': start_at,
                'end_at': end_at,
                'aggregate_at': aggregate_at or end_at,
                'event_type': metadata.get('eventType') or metadata.get('event_type') or 'unknown',
                'source': str(source_key),
                'fallback_used': bool(fallback_used),
            }
            if cached is None:
                self._meta_cache[cache_key] = metadata
            self._record_provenance(
                "event_meta",
                source=source_key,
                fallback_used=fallback_used,
                primary_error=primary_error,
                cache_hit=cached is not None,
            )
            return result

        if self._fallback_enabled(allow_fallback):
            raise RuntimeError(
                f"event {int(event_id)} metadata unavailable; " + "; ".join(attempts)
            )
        logger.warning("; ".join(attempts))
        return None

    def _fetch_event_rewards_from_source(self, source_key, api_config, event_id):
        url_template = api_config.get('event_meta_url')
        if not url_template:
            raise RuntimeError(f"{str(source_key).upper()} event detail URL is not configured")

        url = url_template.format(event_id=event_id)
        event_json = self._get_json(url, timeout=8)
        if not event_json:
            raise RuntimeError(
                f"{str(source_key).upper()} event {int(event_id)} detail request failed: {url}"
            )

        if source_key == 'hhwx':
            event_json = _unwrap_hhwx_event_detail(event_json, event_id)

        ranking_rewards = event_json.get('rankingRewards')
        if not isinstance(ranking_rewards, list):
            raise ValueError(
                f"{str(source_key).upper()} event {int(event_id)} rankingRewards must be a list"
            )

        server_idx = self.server_index
        if not 0 <= server_idx < len(ranking_rewards):
            raise ValueError(
                f"{str(source_key).upper()} event {int(event_id)} has no rankingRewards "
                f"for server {server_idx}"
            )

        entries = ranking_rewards[server_idx]
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"{str(source_key).upper()} event {int(event_id)} "
                f"rankingRewards[{server_idx}] is empty or invalid"
            )
        return entries

    def fetch_event_rewards(self, event_id, allow_fallback=None):
        """
        分析 rankingRewards 中非普通奖励的差分截止点，返回玩家真正在乎的目标档线。
        普通奖励（item, practice_ticket, star, degree）被排除，
        其余特殊奖励必须按 rewardType + rewardId 区分，不能只按类型合并。
        例如同一活动里可能有多个 voice_stamp，不同 stamp 的尾端不同。
        返回 {"target_tiers": [toRank, ...], "last_appearance": {type:id: rank}}
        """
        primary_error = None
        attempts = []
        for source_key, api_config, fallback_used in self._source_candidates(allow_fallback):
            try:
                entries = self._fetch_event_rewards_from_source(
                    source_key,
                    api_config,
                    event_id,
                )
                target_tiers, last_appearance = self._extract_special_reward_targets(entries)
            except Exception as exc:
                detail = f"{str(source_key).upper()}: {type(exc).__name__}: {exc}"
                attempts.append(detail)
                if not fallback_used:
                    primary_error = detail
                continue

            break
        else:
            raise RuntimeError(
                f"event {int(event_id)} rewards unavailable; " + "; ".join(attempts)
            )

        if len(target_tiers) > 3:
            logger.warning(
                f"Event {event_id}: 特殊档线数量 ({len(target_tiers)}) 超过 3，"
                f"新增特殊奖励类型可能导致进一步增加。target_tiers={target_tiers}"
            )

        self._record_provenance(
            "event_rewards",
            source=source_key,
            fallback_used=fallback_used,
            primary_error=primary_error,
        )
        return {
            'target_tiers': target_tiers,
            'last_appearance': last_appearance,
            'source': str(source_key),
            'fallback_used': bool(fallback_used),
        }

    @staticmethod
    def _extract_special_reward_targets(entries):
        ordinary_types = {
            'item', 'practice_ticket', 'star', 'degree', 'bili_degree_effect'
        }
        last_appearance = {}

        for entry in entries:
            reward_type = entry.get('rewardType', '')
            if not reward_type or reward_type in ordinary_types:
                continue
            try:
                rank = int(entry['toRank'])
            except (KeyError, TypeError, ValueError):
                continue

            reward_id = entry.get('rewardId')
            reward_key = f"{reward_type}:{reward_id}" if reward_id is not None else str(reward_type)
            last_appearance[reward_key] = max(last_appearance.get(reward_key, 0), rank)

        target_tiers = sorted(set(last_appearance.values()))
        return target_tiers, last_appearance

    def _fetch_tier_data_from_source(self, source_key, api_config, event_id, tier):
        url = api_config['tracker_url'].format(
            server=self.server_index,
            event_id=event_id,
            tier=tier,
        )
        tracker_data = self._get_json(url, timeout=10)
        if not isinstance(tracker_data, dict):
            raise RuntimeError(
                f"{str(source_key).upper()} event {int(event_id)} T{int(tier)} "
                f"tracker request failed: {url}"
            )
        if tracker_data.get("result") is not True:
            raise ValueError(
                f"{str(source_key).upper()} event {int(event_id)} T{int(tier)} "
                "tracker response must contain result=true"
            )

        cutoffs = tracker_data.get("cutoffs")
        if not isinstance(cutoffs, list) or not cutoffs:
            raise ValueError(
                f"{str(source_key).upper()} event {int(event_id)} T{int(tier)} "
                "tracker cutoffs are empty or invalid"
            )

        frame = pd.DataFrame(cutoffs)
        if frame.empty or "time" not in frame.columns or "ep" not in frame.columns:
            raise ValueError(
                f"{str(source_key).upper()} event {int(event_id)} T{int(tier)} "
                "tracker cutoffs need time and ep"
            )
        # attrs survive the in-memory pipeline and make the provenance gap
        # explicit.  Tracker payloads do not expose per-record available_at.
        frame.attrs.update({
            "source": str(source_key),
            "fetched_at": int(time.time() * 1000),
            "available_at": None,
            "availability_status": "unknown_degraded_no_available_at",
            "event_id": int(event_id),
            "tier": int(tier),
        })
        return frame

    def fetch_tier_data(self, event_id, tier=1000, allow_fallback=None):
        tier = validate_tracker_tier(tier)
        primary_error = None
        attempts = []
        for source_key, api_config, fallback_used in self._source_candidates(allow_fallback):
            try:
                frame = self._fetch_tier_data_from_source(
                    source_key,
                    api_config,
                    event_id,
                    tier,
                )
            except Exception as exc:
                detail = f"{str(source_key).upper()}: {type(exc).__name__}: {exc}"
                attempts.append(detail)
                if not fallback_used:
                    primary_error = detail
                continue

            frame.attrs.update({
                "requested_source": self.api_source,
                "fallback_used": bool(fallback_used),
                "primary_error": primary_error,
            })
            operation = f"tier_data:T{int(tier)}"
            self._record_provenance(
                operation,
                source=source_key,
                fallback_used=fallback_used,
                primary_error=primary_error,
            )
            self._operation_provenance["tier_data"] = self._operation_provenance[operation].copy()
            return frame

        message = (
            f"event {int(event_id)} T{int(tier)} tracker unavailable; "
            + "; ".join(attempts)
        )
        if self._fallback_enabled(allow_fallback):
            raise RuntimeError(message)
        logger.info(message)
        return None

    def fetch_tier_1000_data(self, event_id, tier=1000):
        return self.fetch_tier_data(event_id, tier=tier)

    def _get_adjacent_tracker_tiers(self, tier):
        tier = int(tier)
        tiers = sorted(int(t) for t in ALL_TRACKER_TIERS)
        lower = [t for t in tiers if t < tier]
        upper = [t for t in tiers if t > tier]
        if not lower or not upper:
            return None
        return max(lower), min(upper)

    @staticmethod
    def _score_column(df):
        for column in ("value", "ep", "points", "score", "pt"):
            if column in df.columns:
                return column
        return None

    def _interpolate_tier_dataframe(self, lower_df, upper_df, lower_tier, target_tier, upper_tier):
        """
        用相邻实际榜线构造缺失 tier 的历史曲线。
        这是 baseline 模型的显式历史 fallback，不是固定参数兜底。
        """
        if lower_df is None or upper_df is None or lower_df.empty or upper_df.empty:
            return None

        lower_col = self._score_column(lower_df)
        upper_col = self._score_column(upper_df)
        if lower_col is None or upper_col is None or "time" not in lower_df.columns or "time" not in upper_df.columns:
            return None

        left = lower_df[["time", lower_col]].rename(columns={lower_col: "lower_value"}).copy()
        right = upper_df[["time", upper_col]].rename(columns={upper_col: "upper_value"}).copy()
        left["time"] = pd.to_numeric(left["time"], errors="coerce")
        right["time"] = pd.to_numeric(right["time"], errors="coerce")
        left["lower_value"] = pd.to_numeric(left["lower_value"], errors="coerce")
        right["upper_value"] = pd.to_numeric(right["upper_value"], errors="coerce")
        left = left.dropna().sort_values("time")
        right = right.dropna().sort_values("time")
        if left.empty or right.empty:
            return None

        merged = pd.merge_asof(
            left,
            right,
            on="time",
            direction="nearest",
            tolerance=3600000,
        ).dropna()
        if merged.empty:
            return None

        lower_tier = float(lower_tier)
        target_tier = float(target_tier)
        upper_tier = float(upper_tier)
        if lower_tier <= 0 or target_tier <= lower_tier or upper_tier <= target_tier:
            return None

        rank_weight = float(np.log(target_tier / lower_tier) / np.log(upper_tier / lower_tier))
        merged["value"] = (
            merged["lower_value"] * (1.0 - rank_weight)
            + merged["upper_value"] * rank_weight
        )
        merged["value"] = merged["value"].clip(lower=0.0)
        return merged[["time", "value"]].copy()

    def fetch_interpolated_tier_data_pack(
        self,
        event_id,
        tier,
        allow_scale_fallback=False,
        scale_primary_timeout=2,
        scale_fallback_timeout=2,
        scale_primary_retry=False,
        scale_fallback_retry=False,
    ):
        neighbors = self._get_adjacent_tracker_tiers(tier)
        if neighbors is None:
            return None
        lower_tier, upper_tier = neighbors

        meta = self.fetch_event_meta(event_id)
        if not meta:
            return None

        lower_df = self.fetch_tier_data(event_id, lower_tier)
        upper_df = self.fetch_tier_data(event_id, upper_tier)
        df = self._interpolate_tier_dataframe(lower_df, upper_df, lower_tier, tier, upper_tier)
        if df is None or df.empty:
            return None

        scale = self.fetch_top10_max_speed(
            event_id,
            allow_fallback=allow_scale_fallback,
            primary_timeout=scale_primary_timeout,
            fallback_timeout=scale_fallback_timeout,
            primary_retry=scale_primary_retry,
            fallback_retry=scale_fallback_retry,
            suppress_fallback_log=True,
        )
        if scale is None or scale <= 0:
            return None

        return {
            "event_id": event_id,
            "meta": meta,
            "dataframe": df,
            "scale": scale,
            "tier": int(tier),
            "is_interpolated_tier": True,
            "interpolated_from_tiers": [int(lower_tier), int(upper_tier)],
        }

    def fetch_all_tier_data(self, event_id, tiers=None):
        """
        批量获取全量 tracker 档线数据。
        返回 {tier: DataFrame | None}，缺失的 tier 值为 None。
        """
        tier_list = canonicalize_tracker_tiers(ALL_TRACKER_TIERS if tiers is None else tiers)
        results = {tier: None for tier in tier_list}
        max_workers = min(6, max(len(tier_list), 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tier = {
                executor.submit(self.fetch_tier_data, event_id, tier=tier): tier
                for tier in tier_list
            }
            for future in as_completed(future_to_tier):
                tier = future_to_tier[future]
                try:
                    results[tier] = future.result()
                except Exception as exc:
                    logger.debug(f"[{self.api_source}] fetch_all_tier_data tier={tier} failed: {exc}")
        return results

    def fetch_origin_as_of_tier_snapshot(self, event_id, origin_as_of, tiers=None):
        """Fetch a causal fixed-tier input snapshot for a prediction origin.

        The surface and the T10-derived scale share the same timestamp cutoff.
        Tracker archives do not publish ``available_at``; the returned contract
        exposes that limitation instead of pretending a later fetch was known
        at the origin.
        """
        try:
            origin = int(origin_as_of)
        except (TypeError, ValueError) as exc:
            raise ValueError("origin_as_of must be a millisecond timestamp") from exc
        tier_list = canonicalize_tracker_tiers(ALL_TRACKER_TIERS if tiers is None else tiers)
        meta = self.fetch_event_meta(event_id)
        if not meta:
            return None

        tier_data = self.fetch_all_tier_data(event_id, tier_list)
        from tier_surface import build_origin_as_of_tier_snapshot

        snapshot = build_origin_as_of_tier_snapshot(
            tier_data,
            meta,
            origin,
            tiers=tier_list,
            source=self.api_source,
        )
        scale_observation = self.fetch_top10_max_speed_observation(
            event_id, origin_as_of=origin
        )
        return {
            "event_id": int(event_id),
            "meta": meta,
            "snapshot": snapshot,
            "surface": snapshot.surface,
            "scale": scale_observation.value,
            "scale_provenance": asdict(scale_observation),
            "input_contract": snapshot.quality_report,
        }

    def _calculate_scale_from_points(self, points, debug_limit_ts=None):
        if not points:
            return None

        df = pd.DataFrame(points)
        if debug_limit_ts is not None:
            df = df[df["time"] <= debug_limit_ts]
        if df.empty:
            return None

        df = df.sort_values(["uid", "time"])
        df["speed"] = df.groupby("uid")["value"].diff() / (df.groupby("uid")["time"].diff() / 60000)
        valid = df[(df["speed"] > 0) & (df["speed"] < 1000000)]["speed"]
        if valid.empty:
            return None

        return float(np.mean(valid.nlargest(3).values))

    def _calculate_scale_from_cutoffs(self, cutoffs, debug_limit_ts=None):
        if not cutoffs:
            return None

        df = pd.DataFrame(cutoffs)
        if debug_limit_ts is not None:
            df = df[df["time"] <= debug_limit_ts]
        if df.empty or "ep" not in df.columns:
            return None

        df = df.sort_values("time").copy()
        df["speed"] = df["ep"].diff() / (df["time"].diff() / 60000)
        valid = df[(df["speed"] > 0) & (df["speed"] < 1000000)]["speed"]
        if valid.empty:
            return None

        return float(np.mean(valid.nlargest(3).values))

    def _fetch_top10_max_speed_from_config(
        self,
        api_config,
        event_id,
        debug_limit_ts=None,
        suppress_log=False,
        timeout=10,
        retry=True,
    ):
        url = api_config['top10_url'].format(
            server=self.server_index,
            event_id=event_id,
        )
        data = self._get_json(url, timeout=timeout, suppress_log=suppress_log, retry=retry)
        if not data:
            return None

        try:
            if "points" in data:
                return self._calculate_scale_from_points(
                    data["points"],
                    debug_limit_ts=debug_limit_ts,
                )
            if "cutoffs" in data:
                return self._calculate_scale_from_cutoffs(
                    data["cutoffs"],
                    debug_limit_ts=debug_limit_ts,
                )
        except (KeyError, TypeError, ValueError) as exc:
            log = logger.debug if suppress_log else logger.warning
            log(
                f"[{self.api_source}] Invalid T10 payload for event "
                f"{int(event_id)}: {type(exc).__name__}: {exc}"
            )
        return None

    def fetch_top10_max_speed_observation(
        self,
        event_id,
        debug_limit_ts=None,
        origin_as_of=None,
        allow_fallback=None,
        primary_timeout=10,
        fallback_timeout=10,
        primary_retry=True,
        fallback_retry=True,
        suppress_fallback_log=False,
    ) -> ScaleObservation:
        allow_fallback = self._fallback_enabled(allow_fallback)
        if origin_as_of is not None:
            origin_as_of = int(origin_as_of)
            if debug_limit_ts is not None and int(debug_limit_ts) != origin_as_of:
                raise ValueError("debug_limit_ts and origin_as_of disagree")
            debug_limit_ts = origin_as_of
        debug_key = None if debug_limit_ts is None else int(debug_limit_ts)
        cache_key = (
            self.api_source,
            int(self.server_index),
            int(event_id),
            debug_key,
            bool(allow_fallback),
        )

        def read_cached_entry(cache, key, now_monotonic):
            entry = cache.get(key)
            if (
                isinstance(entry, _TimedCacheEntry)
                and entry.is_fresh(now_monotonic)
                and isinstance(entry.value, ScaleObservation)
                and entry.value.value is not None
            ):
                return replace(entry.value, cache_hit=True)
            cache.pop(key, None)
            return None

        def cache_success(observation):
            if observation.value is None:
                raise ValueError("Cannot cache an empty scale observation")

            # A frozen origin is a deterministic prefix and can be memoized for
            # the process lifetime.  A live result must expire with the tracker
            # refresh window.
            expires_monotonic = None
            if debug_key is None:
                expires_monotonic = (
                    time.monotonic() + LIVE_SCALE_CACHE_TTL_SECONDS
                )
            entry = _TimedCacheEntry(
                value=observation,
                expires_monotonic=expires_monotonic,
            )
            self._scale_cache[cache_key] = entry
            with _GLOBAL_SCALE_CACHE_LOCK:
                _GLOBAL_SCALE_CACHE[cache_key] = entry

        now_monotonic = time.monotonic()
        with _GLOBAL_SCALE_CACHE_LOCK:
            cached = read_cached_entry(
                _GLOBAL_SCALE_CACHE, cache_key, now_monotonic
            )
        if cached is not None:
            self._record_provenance(
                "top10_scale",
                source=cached.source,
                fallback_used=cached.fallback_used,
                primary_error=cached.primary_error,
                cache_hit=True,
            )
            return cached

        cached = read_cached_entry(
            self._scale_cache, cache_key, now_monotonic
        )
        if cached is not None:
            self._record_provenance(
                "top10_scale",
                source=cached.source,
                fallback_used=cached.fallback_used,
                primary_error=cached.primary_error,
                cache_hit=True,
            )
            return cached

        key_lock = _get_scale_key_lock(cache_key)
        with key_lock:
            now_monotonic = time.monotonic()
            with _GLOBAL_SCALE_CACHE_LOCK:
                cached = read_cached_entry(
                    _GLOBAL_SCALE_CACHE, cache_key, now_monotonic
                )
            if cached is not None:
                self._record_provenance(
                    "top10_scale",
                    source=cached.source,
                    fallback_used=cached.fallback_used,
                    primary_error=cached.primary_error,
                    cache_hit=True,
                )
                return cached

            cached = read_cached_entry(
                self._scale_cache, cache_key, now_monotonic
            )
            if cached is not None:
                self._record_provenance(
                    "top10_scale",
                    source=cached.source,
                    fallback_used=cached.fallback_used,
                    primary_error=cached.primary_error,
                    cache_hit=True,
                )
                return cached

            has_bestdori_fallback = bool(allow_fallback)
            primary_url = self.api_config['top10_url'].format(
                server=self.server_index,
                event_id=event_id,
            )
            scale = self._fetch_top10_max_speed_from_config(
                self.api_config,
                event_id,
                debug_limit_ts=debug_limit_ts,
                suppress_log=has_bestdori_fallback,
                timeout=primary_timeout,
                retry=primary_retry,
            )
            if (
                scale is not None
                and np.isfinite(float(scale))
                and float(scale) > 0
            ):
                fetched_at = int(time.time() * 1000)
                observation = ScaleObservation(
                    value=float(scale),
                    source=self.api_source,
                    fetched_at=fetched_at,
                    available_at=None,
                    availability_status=(
                        "unknown_degraded_no_available_at"
                    ),
                    origin_as_of=debug_key,
                    fallback_used=False,
                    cache_expires_at=(
                        None
                        if debug_key is not None
                        else fetched_at
                        + int(LIVE_SCALE_CACHE_TTL_SECONDS * 1000)
                    ),
                )
                cache_success(observation)
                self._record_provenance(
                    "top10_scale",
                    source=self.api_source,
                    fallback_used=False,
                )
                return observation

            primary_error = (
                f"{str(self.api_source).upper()} returned no valid T10 scale: "
                f"{primary_url}"
            )
            if has_bestdori_fallback:
                fallback_source = self.fallback_api_source
                fallback_config = API_SOURCE_CONFIGS[fallback_source]
                logger.info(
                    f"[{self.api_source}] Scale missing for event {event_id}, "
                    f"fallback to {fallback_source} eventtop"
                )
                scale = self._fetch_top10_max_speed_from_config(
                    fallback_config,
                    event_id,
                    debug_limit_ts=debug_limit_ts,
                    suppress_log=suppress_fallback_log,
                    timeout=fallback_timeout,
                    retry=fallback_retry,
                )
                fallback_valid = (
                    scale is not None
                    and np.isfinite(float(scale))
                    and float(scale) > 0
                )
                fetched_at = int(time.time() * 1000)
                fallback_url = fallback_config['top10_url'].format(
                    server=self.server_index,
                    event_id=event_id,
                )
                fallback_error = None if fallback_valid else (
                    f"{str(fallback_source).upper()} returned no valid T10 scale: "
                    f"{fallback_url}"
                )
                observation = ScaleObservation(
                    value=float(scale) if fallback_valid else None,
                    source=fallback_source,
                    fetched_at=fetched_at,
                    available_at=None,
                    availability_status=(
                        "unknown_degraded_no_available_at"
                    ),
                    origin_as_of=debug_key,
                    fallback_used=True,
                    cache_expires_at=(
                        None
                        if debug_key is not None or not fallback_valid
                        else fetched_at
                        + int(LIVE_SCALE_CACHE_TTL_SECONDS * 1000)
                    ),
                    primary_error=primary_error,
                    fallback_error=fallback_error,
                )
                if fallback_valid:
                    cache_success(observation)
                    self._record_provenance(
                        "top10_scale",
                        source=fallback_source,
                        fallback_used=True,
                        primary_error=primary_error,
                    )
                return observation

            observation = ScaleObservation(
                value=None,
                source=self.api_source,
                fetched_at=int(time.time() * 1000),
                available_at=None,
                availability_status="unknown_degraded_no_available_at",
                origin_as_of=debug_key,
                fallback_used=False,
                primary_error=primary_error,
            )
            return observation

    def fetch_top10_max_speed(
        self,
        event_id,
        debug_limit_ts=None,
        origin_as_of=None,
        allow_fallback=None,
        primary_timeout=10,
        fallback_timeout=10,
        primary_retry=True,
        fallback_retry=True,
        suppress_fallback_log=False,
    ):
        """Backward-compatible numeric scale accessor."""
        return self.fetch_top10_max_speed_observation(
            event_id,
            debug_limit_ts=debug_limit_ts,
            origin_as_of=origin_as_of,
            allow_fallback=allow_fallback,
            primary_timeout=primary_timeout,
            fallback_timeout=fallback_timeout,
            primary_retry=primary_retry,
            fallback_retry=fallback_retry,
            suppress_fallback_log=suppress_fallback_log,
        ).value

    def fetch_event_data_pack(
        self,
        event_id,
        tier=1000,
        meta=None,
        scale=None,
        allow_scale_fallback=None,
        scale_primary_timeout=None,
        scale_fallback_timeout=None,
        scale_primary_retry=None,
        scale_fallback_retry=None,
        suppress_scale_fallback_log=None,
        suppress_scale_failure_log=None,
    ):
        """
        获取单个活动的完整数据包。
        meta/scale 为 None 时会自动请求并缓存；可传入预取值跳过重复请求。
        """
        if meta is None:
            meta = self.fetch_event_meta(event_id)
        if not meta:
            return None

        df = None
        if tier is not None:
            df = self.fetch_tier_data(event_id, tier=tier)
            if df is None or df.empty:
                logger.info(f"[{self.api_source}] Event {event_id} tier={tier}: 无可用数据")
                return None

        if scale is None:
            if allow_scale_fallback is None:
                allow_scale_fallback = getattr(
                    self,
                    '_default_allow_scale_fallback',
                    self.allow_fallback,
                )
            if scale_primary_timeout is None:
                scale_primary_timeout = getattr(self, '_default_scale_primary_timeout', 10)
            if scale_fallback_timeout is None:
                scale_fallback_timeout = getattr(self, '_default_scale_fallback_timeout', 10)
            if scale_primary_retry is None:
                scale_primary_retry = getattr(self, '_default_scale_primary_retry', True)
            if scale_fallback_retry is None:
                scale_fallback_retry = getattr(self, '_default_scale_fallback_retry', True)
            if suppress_scale_fallback_log is None:
                suppress_scale_fallback_log = getattr(self, '_default_suppress_scale_fallback_log', False)
            if suppress_scale_failure_log is None:
                suppress_scale_failure_log = getattr(self, '_default_suppress_scale_failure_log', False)
            scale = self.fetch_top10_max_speed(
                event_id,
                allow_fallback=allow_scale_fallback,
                primary_timeout=scale_primary_timeout,
                fallback_timeout=scale_fallback_timeout,
                primary_retry=scale_primary_retry,
                fallback_retry=scale_fallback_retry,
                suppress_fallback_log=suppress_scale_fallback_log or not allow_scale_fallback,
            )
        if scale is None:
            log = logger.debug if suppress_scale_failure_log or not allow_scale_fallback else logger.warning
            log(f"[{self.api_source}] Failed to fetch scale for event {event_id}")

        tracker_provenance = None
        if df is not None:
            tracker_provenance = {
                key: df.attrs.get(key)
                for key in (
                    "source",
                    "requested_source",
                    "fallback_used",
                    "primary_error",
                    "fetched_at",
                )
            }
        scale_provenance = self.get_provenance("top10_scale") if scale is not None else None
        source_provenance = {
            "meta": {
                "source": meta.get("source", self.api_source),
                "fallback_used": bool(meta.get("fallback_used", False)),
            },
            "tracker": tracker_provenance,
            "scale": scale_provenance,
        }
        fallback_used = any(
            bool(item and item.get("fallback_used"))
            for item in source_provenance.values()
        )

        return {
            'event_id': event_id,
            'meta': meta,
            'dataframe': df,
            'scale': scale,
            'tier': tier,
            'source': (
                tracker_provenance.get("source")
                if tracker_provenance
                else meta.get("source", self.api_source)
            ),
            'fallback_used': fallback_used,
            'source_provenance': source_provenance,
        }

    def fetch_current_scale(self, event_id, debug_limit_ts=None):
        """
        获取当前活动的实时 T10 速度 (Scale)。
        """
        try:
            return self.fetch_top10_max_speed(event_id, debug_limit_ts=debug_limit_ts)
        except Exception as e:
            logger.warning(f"Error fetching current scale: {e}")
            return None

    def find_similar_events(
        self,
        target_event_id,
        event_type,
        count=5,
        ignore_ids=None,
        tier=1000,
        allow_scale_fallback=False,
        allow_tier_interpolation=False,
        scale_primary_timeout=2,
        scale_fallback_timeout=2,
        scale_primary_retry=False,
        scale_fallback_retry=False,
    ):
        """
        并发扫描同类活动。按 tier 获取对应榜线数据。
        """
        if ignore_ids is None:
            ignore_ids = []

        candidates = []
        all_idx = self.fetch_events_index()
        if all_idx:
            for eid_s, meta in all_idx.items():
                try:
                    eid = int(eid_s)
                    if eid >= target_event_id:
                        continue
                    if eid in ignore_ids:
                        continue

                    candidate_type = meta.get('eventType') or meta.get('event_type')
                    if candidate_type and str(candidate_type).lower() == str(event_type).lower():
                        candidates.append(eid)
                except Exception:
                    continue
            candidates.sort(reverse=True)

        scan_limit = min(len(candidates), count + 3)
        results = []

        old_timeout = getattr(self, '_default_scale_fallback_timeout', 10)
        old_primary_timeout = getattr(self, '_default_scale_primary_timeout', 10)
        old_primary_retry = getattr(self, '_default_scale_primary_retry', True)
        old_fallback_retry = getattr(self, '_default_scale_fallback_retry', True)
        old_suppress = getattr(self, '_default_suppress_scale_fallback_log', False)
        old_suppress_failure = getattr(self, '_default_suppress_scale_failure_log', False)
        old_allow_fallback = getattr(self, '_default_allow_scale_fallback', False)
        self._default_allow_scale_fallback = allow_scale_fallback
        self._default_scale_primary_timeout = scale_primary_timeout
        self._default_scale_fallback_timeout = scale_fallback_timeout
        self._default_scale_primary_retry = scale_primary_retry
        self._default_scale_fallback_retry = scale_fallback_retry
        self._default_suppress_scale_fallback_log = True
        self._default_suppress_scale_failure_log = True
        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_eid = {
                    executor.submit(self.fetch_event_data_pack, eid, tier): eid
                    for eid in candidates[:scan_limit]
                }

                for future in as_completed(future_to_eid):
                    try:
                        data_pack = future.result()
                        if data_pack and data_pack['scale'] and data_pack['scale'] > 0:
                            if data_pack['meta'].get('event_type') == event_type:
                                results.append(data_pack)
                    except Exception:
                        pass
        finally:
            self._default_allow_scale_fallback = old_allow_fallback
            self._default_scale_primary_timeout = old_primary_timeout
            self._default_scale_fallback_timeout = old_timeout
            self._default_scale_primary_retry = old_primary_retry
            self._default_scale_fallback_retry = old_fallback_retry
            self._default_suppress_scale_fallback_log = old_suppress
            self._default_suppress_scale_failure_log = old_suppress_failure

        results.sort(key=lambda item: item['event_id'], reverse=True)
        if allow_tier_interpolation and len(results) < count:
            existing_event_ids = {int(item['event_id']) for item in results}
            fallback_candidates = [
                eid for eid in candidates[:scan_limit]
                if int(eid) not in existing_event_ids
            ]
            fallback_results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_eid = {
                    executor.submit(
                        self.fetch_interpolated_tier_data_pack,
                        eid,
                        tier,
                        allow_scale_fallback=allow_scale_fallback,
                        scale_primary_timeout=scale_primary_timeout,
                        scale_fallback_timeout=scale_fallback_timeout,
                        scale_primary_retry=scale_primary_retry,
                        scale_fallback_retry=scale_fallback_retry,
                    ): eid
                    for eid in fallback_candidates
                }
                for future in as_completed(future_to_eid):
                    try:
                        data_pack = future.result()
                        if data_pack and data_pack['scale'] and data_pack['scale'] > 0:
                            if data_pack['meta'].get('event_type') == event_type:
                                fallback_results.append(data_pack)
                    except Exception:
                        pass
            if fallback_results:
                fallback_results.sort(key=lambda item: item['event_id'], reverse=True)
                results.extend(fallback_results[: max(0, count - len(results))])
                results.sort(key=lambda item: item['event_id'], reverse=True)
        return results[:count]


class BestdoriDataSource(BandoriDataSource):
    def __init__(self, server_index=DEFAULT_SERVER, allow_fallback=False):
        super().__init__(
            api_source='bestdori',
            server_index=server_index,
            allow_fallback=allow_fallback,
        )


class HHWXDataSource(BandoriDataSource):
    def __init__(self, server_index=DEFAULT_SERVER, allow_fallback=False):
        super().__init__(
            api_source='hhwx',
            server_index=server_index,
            allow_fallback=allow_fallback,
        )


def create_data_source(
    api_source=None,
    server_index=DEFAULT_SERVER,
    allow_fallback=False,
):
    return BandoriDataSource(
        api_source=api_source,
        server_index=server_index,
        allow_fallback=allow_fallback,
    )
