import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from config import API_SOURCE_CONFIGS, DEFAULT_API_SOURCE, DEFAULT_SERVER

logger = logging.getLogger('predictor.datasource')

SERVER_KEYS_BY_INDEX = ['jp', 'tw', 'en', 'cn', 'kr']
_GLOBAL_SCALE_CACHE = {}
_GLOBAL_SCALE_CACHE_LOCK = threading.RLock()
_GLOBAL_SCALE_KEY_LOCKS = {}
_GLOBAL_EVENTS_INDEX_CACHE = {}
_GLOBAL_EVENTS_INDEX_LOCK = threading.RLock()


def _get_scale_key_lock(cache_key):
    with _GLOBAL_SCALE_CACHE_LOCK:
        lock = _GLOBAL_SCALE_KEY_LOCKS.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _GLOBAL_SCALE_KEY_LOCKS[cache_key] = lock
        return lock


def _normalize_hhwx_events_payload(payload):
    """把 /api/bandori/events 的新结构摊平成旧 bestdori 代理风格的字典。"""
    events = payload.get('data', {}).get('events') if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return None

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
    def __init__(self, api_source=None, server_index=DEFAULT_SERVER):
        self.api_source, self.api_config = resolve_api_source_config(api_source)
        self.server_index = server_index
        self.session = requests.Session()
        try:
            adapter = HTTPAdapter(pool_connections=10, pool_maxsize=30, max_retries=3)
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
        except Exception as e:
            logger.warning(f"Failed to mount adapter: {e}")
        self._events_index_cache = None
        self._meta_cache: dict = {}
        self._scale_cache: dict = {}

    def close(self):
        """关闭 Session 资源"""
        if self.session:
            self.session.close()

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

    def fetch_events_index(self, timeout=8):
        if self._events_index_cache is not None:
            return self._events_index_cache
        cache_key = (self.api_source, int(self.server_index))
        with _GLOBAL_EVENTS_INDEX_LOCK:
            if cache_key in _GLOBAL_EVENTS_INDEX_CACHE:
                self._events_index_cache = _GLOBAL_EVENTS_INDEX_CACHE[cache_key]
                return self._events_index_cache

        url = self.api_config['event_index_url']
        payload = self._get_json(url, timeout=timeout)
        if isinstance(payload, dict) and isinstance(payload.get('data'), dict) \
                and 'events' in payload['data']:
            result = _normalize_hhwx_events_payload(payload)
        else:
            result = payload
        self._events_index_cache = result
        if result is not None:
            with _GLOBAL_EVENTS_INDEX_LOCK:
                _GLOBAL_EVENTS_INDEX_CACHE[cache_key] = result
        return result

    def get_current_event_id(self, server_index=None):
        """
        从 events index 推断当前活动 ID。
        兼容 Bestdori 原始 events/all.3.json 以及 HHWX 的代理接口。
        """
        if server_index is None:
            server_index = self.server_index

        all_events = self.fetch_events_index()
        if not all_events:
            return None

        now_ms = int(time.time() * 1000)
        active_candidates = []
        nearest_candidate = None
        nearest_dist = float('inf')

        for eid_str, meta in all_events.items():
            try:
                starts = meta.get('startAt') or meta.get('start_at')
                ends = meta.get('endAt') or meta.get('end_at')
                if not isinstance(starts, list) or not isinstance(ends, list):
                    continue
                if server_index >= len(starts) or server_index >= len(ends):
                    continue

                start = starts[server_index]
                end = ends[server_index]
                if start is None or end is None:
                    continue

                start = int(start)
                end = int(end)
                event_id = int(eid_str)

                if start <= now_ms <= end:
                    active_candidates.append((event_id, start))
                else:
                    dist = min(abs(start - now_ms), abs(end - now_ms))
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_candidate = event_id
            except Exception:
                continue

        if active_candidates:
            active_candidates.sort(key=lambda item: item[1], reverse=True)
            return active_candidates[0][0]

        return nearest_candidate

    def fetch_event_meta(self, event_id):
        if event_id in self._meta_cache:
            return self._meta_cache[event_id]

        url_template = self.api_config.get('event_meta_url')
        metadata = None
        if url_template:
            url = url_template.format(event_id=event_id)
            metadata = self._get_json(url, timeout=5)
        if not metadata:
            all_events = self.fetch_events_index()
            if isinstance(all_events, dict):
                metadata = all_events.get(str(int(event_id)))
        if not metadata:
            return None

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
            return None

        result = {
            'event_id': int(event_id),
            'start_at': start_at,
            'end_at': end_at,
            'aggregate_at': aggregate_at or end_at,
            'event_type': metadata.get('eventType') or metadata.get('event_type') or 'unknown',
        }
        self._meta_cache[event_id] = result
        return result

    def fetch_event_rewards(self, event_id):
        """
        分析 rankingRewards 中非普通奖励的差分截止点，返回玩家真正在乎的目标档线。
        普通奖励（item, practice_ticket, star, degree）被排除，
        其余类型（voice_stamp, deco_pins 及未来新增特殊奖励）自动视为特殊奖励。
        返回 {"target_tiers": [toRank, ...], "last_appearance": {type: rank}}
        """
        ORDINARY_TYPES = {'item', 'practice_ticket', 'star', 'degree'}

        url_template = self.api_config.get('event_meta_url')
        if not url_template:
            return {'target_tiers': [], 'last_appearance': {}}

        url = url_template.format(event_id=event_id)
        event_json = self._get_json(url, timeout=8)
        if not event_json:
            return {'target_tiers': [], 'last_appearance': {}}

        ranking_rewards = event_json.get('rankingRewards')
        if not isinstance(ranking_rewards, list):
            return {'target_tiers': [], 'last_appearance': {}}

        server_idx = self.server_index
        if server_idx >= len(ranking_rewards):
            return {'target_tiers': [], 'last_appearance': {}}

        entries = ranking_rewards[server_idx]
        if not isinstance(entries, list):
            return {'target_tiers': [], 'last_appearance': {}}

        from collections import defaultdict
        all_ranks = set()
        by_rank = defaultdict(set)
        for e in entries:
            r = int(e['toRank'])
            all_ranks.add(r)
            rt = e.get('rewardType', '')
            if rt and rt not in ORDINARY_TYPES:
                by_rank[r].add(rt)
        for r in all_ranks:
            if r not in by_rank:
                by_rank[r] = set()

        last_appearance = {}
        sorted_ranks = sorted(by_rank.keys())
        for rank in sorted_ranks:
            for rt in by_rank[rank]:
                last_appearance[rt] = max(last_appearance.get(rt, 0), rank)

        target_tiers = sorted(set(last_appearance.values()))

        if len(target_tiers) > 3:
            logger.warning(
                f"Event {event_id}: 特殊档线数量 ({len(target_tiers)}) 超过 3，"
                f"新增特殊奖励类型可能导致进一步增加。target_tiers={target_tiers}"
            )

        return {
            'target_tiers': target_tiers,
            'last_appearance': last_appearance,
        }

    def fetch_tier_data(self, event_id, tier=1000):
        url = self.api_config['tracker_url'].format(
            server=self.server_index,
            event_id=event_id,
            tier=tier,
        )
        tracker_data = self._get_json(url, timeout=10)
        if not tracker_data or not tracker_data.get("result"):
            return None

        cutoffs = tracker_data.get("cutoffs") or []
        if not cutoffs:
            return None

        return pd.DataFrame(cutoffs)

    def fetch_tier_1000_data(self, event_id, tier=1000):
        return self.fetch_tier_data(event_id, tier=tier)

    def fetch_all_tier_data(self, event_id, tiers=None):
        """
        批量获取全量 tracker 档线数据。
        返回 {tier: DataFrame | None}，缺失的 tier 值为 None。
        """
        from config import ALL_TRACKER_TIERS
        if tiers is None:
            tiers = ALL_TRACKER_TIERS

        tier_list = [int(tier) for tier in tiers]
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

        if "points" in data:
            return self._calculate_scale_from_points(data["points"], debug_limit_ts=debug_limit_ts)
        if "cutoffs" in data:
            return self._calculate_scale_from_cutoffs(data["cutoffs"], debug_limit_ts=debug_limit_ts)
        return None

    def fetch_top10_max_speed(
        self,
        event_id,
        debug_limit_ts=None,
        allow_fallback=True,
        primary_timeout=10,
        fallback_timeout=10,
        primary_retry=True,
        fallback_retry=True,
        suppress_fallback_log=False,
    ):
        debug_key = None if debug_limit_ts is None else int(debug_limit_ts)
        cache_key = (
            self.api_source,
            int(self.server_index),
            int(event_id),
            debug_key,
            bool(allow_fallback),
        )
        with _GLOBAL_SCALE_CACHE_LOCK:
            if cache_key in _GLOBAL_SCALE_CACHE:
                return _GLOBAL_SCALE_CACHE[cache_key]

        if debug_limit_ts is None and event_id in self._scale_cache:
            return self._scale_cache[event_id]

        key_lock = _get_scale_key_lock(cache_key)
        with key_lock:
            with _GLOBAL_SCALE_CACHE_LOCK:
                if cache_key in _GLOBAL_SCALE_CACHE:
                    return _GLOBAL_SCALE_CACHE[cache_key]

            if debug_limit_ts is None and event_id in self._scale_cache:
                scale = self._scale_cache[event_id]
                with _GLOBAL_SCALE_CACHE_LOCK:
                    _GLOBAL_SCALE_CACHE[cache_key] = scale
                return scale

            has_bestdori_fallback = allow_fallback and self.api_source != 'bestdori' and 'bestdori' in API_SOURCE_CONFIGS
            scale = self._fetch_top10_max_speed_from_config(
                self.api_config,
                event_id,
                debug_limit_ts=debug_limit_ts,
                suppress_log=has_bestdori_fallback,
                timeout=primary_timeout,
                retry=primary_retry,
            )
            if scale is not None:
                if debug_limit_ts is None:
                    self._scale_cache[event_id] = scale
                with _GLOBAL_SCALE_CACHE_LOCK:
                    _GLOBAL_SCALE_CACHE[cache_key] = scale
                return scale

            if has_bestdori_fallback:
                fallback_config = API_SOURCE_CONFIGS['bestdori']
                logger.info(
                    f"[{self.api_source}] Scale missing for event {event_id}, fallback to bestdori eventtop"
                )
                scale = self._fetch_top10_max_speed_from_config(
                    fallback_config,
                    event_id,
                    debug_limit_ts=debug_limit_ts,
                    suppress_log=suppress_fallback_log,
                    timeout=fallback_timeout,
                    retry=fallback_retry,
                )
                if scale is not None and debug_limit_ts is None:
                    self._scale_cache[event_id] = scale
                with _GLOBAL_SCALE_CACHE_LOCK:
                    _GLOBAL_SCALE_CACHE[cache_key] = scale
                return scale

            with _GLOBAL_SCALE_CACHE_LOCK:
                _GLOBAL_SCALE_CACHE[cache_key] = scale
            return None

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
                allow_scale_fallback = getattr(self, '_default_allow_scale_fallback', True)
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

        return {
            'event_id': event_id,
            'meta': meta,
            'dataframe': df,
            'scale': scale,
            'tier': tier,
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
        allow_scale_fallback=True,
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

        if not candidates:
            candidates = list(range(target_event_id - 1, max(0, target_event_id - 100), -1))
            candidates = [eid for eid in candidates if eid not in ignore_ids]

        scan_limit = min(len(candidates), count + 3)
        results = []

        old_timeout = getattr(self, '_default_scale_fallback_timeout', 10)
        old_primary_timeout = getattr(self, '_default_scale_primary_timeout', 10)
        old_primary_retry = getattr(self, '_default_scale_primary_retry', True)
        old_fallback_retry = getattr(self, '_default_scale_fallback_retry', True)
        old_suppress = getattr(self, '_default_suppress_scale_fallback_log', False)
        old_suppress_failure = getattr(self, '_default_suppress_scale_failure_log', False)
        old_allow_fallback = getattr(self, '_default_allow_scale_fallback', True)
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
        return results[:count]


class BestdoriDataSource(BandoriDataSource):
    def __init__(self, server_index=DEFAULT_SERVER):
        super().__init__(api_source='bestdori', server_index=server_index)


class HHWXDataSource(BandoriDataSource):
    def __init__(self, server_index=DEFAULT_SERVER):
        super().__init__(api_source='hhwx', server_index=server_index)


def create_data_source(api_source=None, server_index=DEFAULT_SERVER):
    return BandoriDataSource(api_source=api_source, server_index=server_index)
