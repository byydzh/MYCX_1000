import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from config import API_SOURCE_CONFIGS, DEFAULT_API_SOURCE, DEFAULT_SERVER

logger = logging.getLogger('predictor.datasource')

SERVER_KEYS_BY_INDEX = ['jp', 'tw', 'en', 'cn', 'kr']


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

    def _get_json(self, url, timeout=10, suppress_log=False):
        try:
            response = self.session.get(url, timeout=timeout)
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
        url = self.api_config['event_index_url']
        payload = self._get_json(url, timeout=timeout)
        if isinstance(payload, dict) and isinstance(payload.get('data'), dict) \
                and 'events' in payload['data']:
            result = _normalize_hhwx_events_payload(payload)
        else:
            result = payload
        self._events_index_cache = result
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

    def _fetch_top10_max_speed_from_config(self, api_config, event_id, debug_limit_ts=None, suppress_log=False):
        url = api_config['top10_url'].format(
            server=self.server_index,
            event_id=event_id,
        )
        data = self._get_json(url, timeout=10, suppress_log=suppress_log)
        if not data:
            return None

        if "points" in data:
            return self._calculate_scale_from_points(data["points"], debug_limit_ts=debug_limit_ts)
        if "cutoffs" in data:
            return self._calculate_scale_from_cutoffs(data["cutoffs"], debug_limit_ts=debug_limit_ts)
        return None

    def fetch_top10_max_speed(self, event_id, debug_limit_ts=None):
        if event_id in self._scale_cache:
            return self._scale_cache[event_id]

        has_bestdori_fallback = self.api_source != 'bestdori' and 'bestdori' in API_SOURCE_CONFIGS
        scale = self._fetch_top10_max_speed_from_config(
            self.api_config,
            event_id,
            debug_limit_ts=debug_limit_ts,
            suppress_log=has_bestdori_fallback,
        )
        if scale is not None:
            self._scale_cache[event_id] = scale
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
            )
            if scale is not None:
                self._scale_cache[event_id] = scale
            return scale

        return None

    def fetch_event_data_pack(self, event_id, tier=1000, meta=None, scale=None):
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
            scale = self.fetch_top10_max_speed(event_id)
        if scale is None:
            logger.warning(f"[{self.api_source}] Failed to fetch scale for event {event_id}")

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

    def find_similar_events(self, target_event_id, event_type, count=5, ignore_ids=None, tier=1000):
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

        scan_limit = count + 3
        results = []

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
