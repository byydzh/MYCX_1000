# data_source.py
import requests
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter

# 假设 base_distribution 还在原来的位置，我们需要用到它里面的基础抓取函数
# 如果您打算彻底重构，以后这些也可以移进来，但现在先保持兼容
from base_distribution import (
    fetch_event_meta,
    fetch_tier_1000_data,
    fetch_top10_max_speed,
    BASE_URL,
    SERVER
)

logger = logging.getLogger('predictor.datasource')

class BestdoriDataSource:
    """
    负责与 Bestdori API 进行所有网络交互的客户端。
    管理 HTTP Session，处理重试，以及并发扫描同类活动。
    """
    def __init__(self):
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

    def fetch_recent_json(self, timeout=10):
        """获取 recent.json"""
        url = "https://bestdori.com/api/news/dynamic/recent.json"
        try:
            r = self.session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"Failed to fetch recent.json: {e}")
            return None

    def get_current_event_id(self, server_index=3):
        """
        解析 recent.json 获取当前服务器的活动 ID。
        为了解耦，这里只负责获取 ID，不再混合其他逻辑。
        """
        recent = self.fetch_recent_json()
        if not recent:
            return None
        
        # 复用原有的解析逻辑，但简化流程
        events_node = recent.get('events') if 'events' in recent else recent
        if not events_node: 
            return None

        now_ms = int(time.time() * 1000)
        active_candidates = []
        nearest_candidate = None
        nearest_dist = float('inf')

        for eid, meta in events_node.items():
            try:
                starts = meta.get('startAt') or meta.get('start_at')
                ends = meta.get('endAt') or meta.get('end_at')
                if not isinstance(starts, list) or not isinstance(ends, list): continue
                
                # 安全获取对应服务器的时间
                if server_index >= len(starts) or server_index >= len(ends): continue
                start = starts[server_index]
                end = ends[server_index]
                
                # 简单的类型转换
                if start is None or end is None: continue
                start, end = int(start), int(end)

                if start <= now_ms <= end:
                    active_candidates.append((int(eid), start))
                else:
                    # 计算距离
                    dist = min(abs(start - now_ms), abs(end - now_ms))
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_candidate = int(eid)
            except:
                continue

        if active_candidates:
            # 返回开始时间最晚（最新）的活动
            active_candidates.sort(key=lambda x: x[1], reverse=True)
            return active_candidates[0][0]
        
        return nearest_candidate

    def fetch_event_data_pack(self, event_id):
        """
        获取单个活动的完整数据包（Meta, T1000, Scale）。
        原本这部分逻辑散落在 DataHandler 里，现在统一获取。
        """
        meta = fetch_event_meta(event_id)
        if not meta: return None
        
        df = fetch_tier_1000_data(event_id)
        if df is None or df.empty: return None
        
        scale = fetch_top10_max_speed(event_id)
        # 如果获取失败，返回 None，由上层决定是否使用默认值
        
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
        url = f"{BASE_URL}eventtop/data?server={SERVER}&event={event_id}&mid=0&interval=3600000"
        try:
            data = self.session.get(url, timeout=10).json()
            if not data or "points" not in data: return None
            
            # 这里只做最基础的数据提取，不进行复杂的 Pandas 计算，
            # 复杂的计算留给 Engine 层，但为了保持 fetch 的原子性，
            # 我们暂时返回原始 points 列表，或者在这里做轻量计算。
            # 鉴于原逻辑这里计算量不大，我们先保留计算逻辑以返回最终数值。
            import pandas as pd
            import numpy as np
            
            df = pd.DataFrame(data["points"])
            if debug_limit_ts: 
                df = df[df["time"] <= debug_limit_ts]
            if df.empty: return None
            
            df = df.sort_values(["uid", "time"])
            # 简单的差分计算
            df["speed"] = df.groupby("uid")["value"].diff() / (df.groupby("uid")["time"].diff() / 60000)
            valid = df[(df["speed"] > 0) & (df["speed"] < 1000000)]["speed"]
            if valid.empty: return None
            
            return np.mean(valid.nlargest(3).values)
        except Exception as e:
            logger.warning(f"Error fetching current scale: {e}")
            return None

    def find_similar_events(self, target_event_id, event_type, count=5):
        """
        并发扫描同类活动。
        将原 DataHandler 中的复杂线程池逻辑移至此处。
        """
        candidates = []
        # 1. 尝试获取快速索引
        try:
            idx_url = "https://bestdori.com/api/events/all.3.json"
            r = self.session.get(idx_url, timeout=8)
            if r.ok:
                all_idx = r.json()
                for eid_s, meta in all_idx.items():
                    try:
                        eid = int(eid_s)
                        if eid >= target_event_id: continue # 只看历史
                        
                        et = meta.get('eventType') or meta.get('event_type')
                        if et and str(et).lower() == str(event_type).lower():
                            candidates.append(eid)
                    except: continue
                candidates.sort(reverse=True)
        except Exception as e:
            logger.warning(f"Fast index fetch failed: {e}")
            
        # 回退策略
        if not candidates:
            candidates = list(range(target_event_id - 1, max(0, target_event_id - 100), -1))
        
        scan_limit = count + 3
        
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_eid = {
                executor.submit(self.fetch_event_data_pack, eid): eid 
                for eid in candidates[:scan_limit] # 限制提交任务的数量
            }

            for future in as_completed(future_to_eid):
                try:
                    data_pack = future.result()
                    if data_pack and data_pack['scale'] > 0:
                        if data_pack['meta'].get('event_type') == event_type:
                            results.append(data_pack)
                except Exception:
                    pass
        
        # 2. 【关键】对结果进行排序，确保确定性！
        # 按 event_id 倒序排列（优先用最近的活动），然后切片取前 count 个
        results.sort(key=lambda x: x['event_id'], reverse=True)
        final_results = results[:count]
        
        print(f"最终选定参考活动: {[r['event_id'] for r in final_results]}")
        return final_results