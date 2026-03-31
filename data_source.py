import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from config import API_SOURCE_CONFIGS, DEFAULT_API_SOURCE, DEFAULT_SERVER

logger = logging.getLogger('predictor.datasource')


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
            adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=3)
            self.session.mount('http://', adapter)
            self.session.mount('https://', adapter)
        except Exception as e:
            logger.warning(f"Failed to mount adapter: {e}")

    def close(self):
        """关闭 Session 资源"""
        if self.session:
            self.session.close()

    def _get_json(self, url, timeout=10):
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"[{self.api_source}] Request failed: {url} | {e}")
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
        url = self.api_config['event_index_url']
        return self._get_json(url, timeout=timeout)

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
        url = self.api_config['event_meta_url'].format(event_id=event_id)
        metadata = self._get_json(url, timeout=5)
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

        return {
            'event_id': int(event_id),
            'start_at': start_at,
            'end_at': end_at,
            'aggregate_at': aggregate_at or end_at,
            'event_type': metadata.get('eventType') or metadata.get('event_type') or 'unknown',
        }

    def fetch_tier_1000_data(self, event_id, tier=1000):
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

    def fetch_top10_max_speed(self, event_id, debug_limit_ts=None):
        url = self.api_config['top10_url'].format(
            server=self.server_index,
            event_id=event_id,
        )
        data = self._get_json(url, timeout=10)
        if not data:
            return None

        if "points" in data:
            return self._calculate_scale_from_points(data["points"], debug_limit_ts=debug_limit_ts)
        if "cutoffs" in data:
            return self._calculate_scale_from_cutoffs(data["cutoffs"], debug_limit_ts=debug_limit_ts)
        return None

    def fetch_event_data_pack(self, event_id):
        """
        获取单个活动的完整数据包（Meta, T1000, Scale）。
        """
        meta = self.fetch_event_meta(event_id)
        if not meta:
            return None

        df = self.fetch_tier_1000_data(event_id)
        if df is None or df.empty:
            return None

        scale = self.fetch_top10_max_speed(event_id)
        if scale is None:
            logger.warning(f"[{self.api_source}] Failed to fetch scale for event {event_id}")

        return {
            'event_id': event_id,
            'meta': meta,
            'dataframe': df,
            'scale': scale
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

    def find_similar_events(self, target_event_id, event_type, count=5, ignore_ids=None):
        """
        并发扫描同类活动。
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
                executor.submit(self.fetch_event_data_pack, eid): eid
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
