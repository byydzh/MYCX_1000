import requests
import pandas as pd
import numpy as np
import json
import os
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from io import BytesIO
from scipy.optimize import curve_fit
import logging
from logging import handlers

# 引入基础工具
from base_distribution import (
    fetch_event_meta, 
    fetch_tier_1000_data,
    calculate_speed_tracker,
    get_day_type, 
    fetch_top10_max_speed,
    BASE_URL, 
    SERVER
)

try:
    from chinese_calendar import is_workday
except Exception:
    is_workday = None

from config import DEFAULT_CONFIG

# Create a module-level requests Session to enable connection reuse and
# avoid creating many short-lived sockets (prevents FD exhaustion/time_wait)
HTTP_SESSION = requests.Session()
try:
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=3)
    HTTP_SESSION.mount('http://', adapter)
    HTTP_SESSION.mount('https://', adapter)
except Exception:
    # if adapter creation fails for any reason, keep using default session
    pass

# Setup logger for detailed run diagnostics (file-only; do not print logs to terminal)
LOG_PATH = os.path.join(os.path.dirname(__file__), 'predictor.log')
logger = logging.getLogger('predictor')
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    # Auto-truncate/rotate log if it grows beyond 1MB to avoid disk bloat
    try:
        if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > (1 * 1024 * 1024):
            # simple truncation: rename the old file with timestamp and start fresh
            try:
                bak = LOG_PATH + ".old"
                if os.path.exists(bak):
                    os.remove(bak)
                os.rename(LOG_PATH, bak)
            except Exception:
                # fallback: truncate in place
                try:
                    open(LOG_PATH, 'w', encoding='utf-8').close()
                except Exception:
                    pass
    except Exception:
        pass

    fh = logging.FileHandler(LOG_PATH, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Disable propagation to avoid duplicate console output from root logger
    logger.propagate = False


def fetch_recent_json(timeout=10):
    """Fetch Bestdori recent.json and return parsed JSON or None on failure."""
    url = "https://bestdori.com/api/news/dynamic/recent.json"
    try:
        r = HTTP_SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Failed to fetch recent.json from Bestdori: {e}")
        return None


def get_current_event_for_server(recent_json=None, server_index=3, now_ts_ms=None, fetch_if_missing=True):
    """
    Determine the current event ID for a server given a Bestdori `recent.json` structure.

    Parameters:
    - recent_json: dict or None. If None and fetch_if_missing==True, the function will fetch the JSON.
    - server_index: int index into arrays like startAt/endAt for the target server.
      NOTE: user environment reported CN at position index 3 (4th position). Default here is 3 to match that.
    - now_ts_ms: optional epoch milliseconds to use as "now" (for testing). Defaults to current time.

    Returns:
    - event_id (int) of the active event for that server, or the nearest event id (int) when no active event,
      or None if parsing/fetch fails.
    """
    if recent_json is None:
        if not fetch_if_missing:
            return None
        recent_json = fetch_recent_json()

    if not recent_json:
        return None

    # The `events` node normally contains mapping event_id -> metadata dict
    events_node = None
    if isinstance(recent_json, dict):
        # try common keys first
        if 'events' in recent_json and isinstance(recent_json['events'], dict):
            events_node = recent_json['events']
        else:
            # If recent_json is already the events dict, accept it
            # (bestdori may return nested structures in some proxies)
            # detect by checking items have 'startAt' keys
            ok = True
            for k, v in recent_json.items():
                if not isinstance(v, dict) or ('startAt' not in v and 'start_at' not in v):
                    ok = False
                    break
            if ok:
                events_node = recent_json

    if not events_node:
        logger.warning('Cannot locate events node in recent.json')
        return None

    now_ms = int(now_ts_ms) if now_ts_ms is not None else int(time.time() * 1000)

    active_candidates = []
    nearest_candidate = None
    nearest_dist = None

    for eid, meta in events_node.items():
        try:
            # support both camelCase and snake_case keys
            starts = meta.get('startAt') or meta.get('start_at')
            ends = meta.get('endAt') or meta.get('end_at')
            # ensure lists
            if not isinstance(starts, (list, tuple)) or not isinstance(ends, (list, tuple)):
                continue

            # get server-specific entries; they might be string numbers or None
            def safe_int(arr, idx):
                try:
                    v = arr[idx]
                    if v is None:
                        return None
                    if isinstance(v, (int, float)):
                        return int(v)
                    s = str(v).strip()
                    if s == '' or s.lower() == 'null':
                        return None
                    return int(s)
                except Exception:
                    return None

            start = safe_int(starts, server_index)
            end = safe_int(ends, server_index)

            # If both start and end exist, check active
            if start is not None and end is not None and start <= now_ms <= end:
                active_candidates.append((int(eid), start, end))
            else:
                # compute minimal distance to now (consider start/end if present)
                d = None
                if start is not None:
                    d = abs(start - now_ms)
                if end is not None:
                    d2 = abs(end - now_ms)
                    d = d if (d is not None and d <= d2) else d2
                if d is not None:
                    if nearest_dist is None or d < nearest_dist:
                        nearest_dist = d
                        nearest_candidate = int(eid)
        except Exception:
            continue

    # If multiple active, pick the one with latest start time (most recent)
    if active_candidates:
        active_candidates.sort(key=lambda x: x[1], reverse=True)
        chosen = active_candidates[0][0]
        logger.info(f"Selected active event for server_index={server_index}: {chosen}")
        return int(chosen)

    # no active events: return nearest by timestamp
    if nearest_candidate is not None:
        logger.info(f"No active event; returning nearest event {nearest_candidate} for server_index={server_index}")
        return int(nearest_candidate)

    return None


# ==========================================
# 1. 昼夜节律处理器 (SeasonalityHandler)
# ==========================================
class SeasonalityHandler:
    def __init__(self, json_path='base_speed_distribution.json', tz_offset=8, panic_ease_power=1.0, weekend_multiplier=1.0, panic_scaler=1.1):
        self.data = self._load_json(json_path)

        if weekend_multiplier != 1.0:
            print(f"正在应用周末增强系数: x{weekend_multiplier} 喵！")
            self._apply_multiplier('weekend', weekend_multiplier)
        
        # 修正逻辑：分别计算平日和周末的均值，再进行 5:2 加权合成
        self.wd_mean, self.we_mean, self.global_mean = self._calculate_weighted_means()
        
        # 注意: `base_speed_distribution.json` 中的小时数据已经是当地时区的小时
        # 因此不再对时间戳进行额外的时区偏移。保留 tz_offset 属性仅作兼容。
        self.tz_offset = tz_offset
        self.panic_ease_power = float(panic_ease_power)
        self.panic_scaler = float(panic_scaler)
        
        print(f"昼夜节律数据已加载:")
        # print(f"  - 平日基准均值 (Weekday): {self.wd_mean:.6f}")
        # print(f"  - 周末基准均值 (Weekend): {self.we_mean:.6f}")
        # print(f"  - 全局加权均值 (Weighted Global): {self.global_mean:.6f} (panic_ease={self.panic_ease_power})")

    def _load_json(self, path):
        if not os.path.exists(path):
            print(f"警告：找不到 {path}，将不使用节律修正。")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def _apply_multiplier(self, dtype, multiplier):
        """直接修改 self.data 字典中的均值数据"""
        if dtype not in self.data: return
        for hour_key in self.data[dtype]:
            if 'mean' in self.data[dtype][hour_key]:
                self.data[dtype][hour_key]['mean'] *= multiplier

    def _calculate_weighted_means(self):
        """
        分别计算平日和周末的日均速度，并按 5:2 权重合成全局均值。
        这样可以防止因周末/平日数据量不均导致的基准偏差。
        """
        if not self.data: return 1.0, 1.0, 1.0
        
        def get_day_type_mean(dtype):
            values = []
            # 遍历 0-23 小时
            for h in range(24):
                item = self.data.get(dtype, {}).get(str(h))
                if item and item.get('mean', 0) > 0:
                    values.append(item['mean'])
            # 如果该类型没有数据，返回 1.0 避免除零，否则返回该类型一天的平均速度
            return np.mean(values) if values else 1.0

        wd_mean = get_day_type_mean('weekday')
        we_mean = get_day_type_mean('weekend')
        
        # 核心修正：加权平均 (平日5天，周末2天)
        # 这样算出来的 global_mean 才是这一周真实的“期望速度”
        weighted_global = (wd_mean * 5.0 + we_mean * 2.0) / 7.0
        
        return wd_mean, we_mean, weighted_global

    def get_factor(self, dt):
        if not self.data: return 1.0
        if isinstance(dt, (int, float)): 
            dt_obj = datetime.fromtimestamp(dt / 1000) + timedelta(hours=self.tz_offset)
        else:
            dt_obj = dt
            
        # Special rules: Friday after 17:00 treated as weekend (players behave like weekend)
        # and Sunday after 23:00 treated as weekday (late-night rollback to weekday pattern)
        if isinstance(dt_obj, datetime):
            if dt_obj.weekday() == 4 and dt_obj.hour >= 17:
                dtype = 'weekend'
            elif dt_obj.weekday() == 6 and dt_obj.hour >= 23:
                dtype = 'weekday'
            else:
                # 2. 使用 chinese_calendar（若可用）判断法定节假日/工作日
                if is_workday is not None:
                    try:
                        if is_workday(dt_obj.date()):
                            dtype = 'weekday'
                        else:
                            dtype = 'weekend'
                    except:
                        dtype = get_day_type(dt_obj)
                else:
                    # 3. Fallback: 普通周末判断
                    dtype = get_day_type(dt_obj)
        else:
            # If dt_obj is not datetime for some reason, fallback
            if is_workday is not None:
                try:
                    if is_workday(dt_obj): dtype = 'weekday'
                    else: dtype = 'weekend'
                except:
                    dtype = get_day_type(dt_obj)
            else:
                dtype = get_day_type(dt_obj)
            
        hour = str(dt_obj.hour)
        stats = self.data.get(dtype, {}).get(hour)
        
        # 这里的逻辑保持不变：用当前小时的均值除以【全局加权均值】
        # 这样如果 dtype 是 weekend，分子通常较大，Factor > 1.0，正确反映周末加速
        if stats and stats['mean'] > 0:
            return stats['mean'] / self.global_mean
        return 1.0 

    def remove_seasonality(self, df):
        df = df.copy()
        # `base_speed_distribution.json` uses local hours. Convert timestamp to local
        # datetime by applying the detected tz_offset.
        df['dt_local'] = pd.to_datetime(df['time'], unit='ms') + pd.Timedelta(hours=self.tz_offset)
        df['season_factor'] = df['dt_local'].apply(self.get_factor)

        # --- Early-hour suppression ---
        if 'hours_elapsed' in df.columns:
            mask_early = (df['hours_elapsed'] < 12.0) & (df['season_factor'] < 1.0)
            if mask_early.any():
                df.loc[mask_early, 'season_factor'] = 1.0

        df['skeleton_speed'] = df['norm_speed'] / df['season_factor']
        return df

    def apply_seasonality(self, t_hours, y_skeleton, start_ts, total_hours=None, t_panic=24.0):
        y_final = []
        factors = []

        for i, h in enumerate(t_hours):
            current_ts = start_ts + (h * 3600 * 1000)
            dt_utc = datetime.fromtimestamp(current_ts / 1000, timezone.utc)
            dt_local = dt_utc + timedelta(hours=self.tz_offset)

            # 1) 原始节律因子
            raw_factor = self.get_factor(dt_local)

            # 2) 恐慌/肾上腺素修正
            final_factor = raw_factor
            if (total_hours is not None) and (t_panic is not None) and t_panic > 0:
                time_left = total_hours - h
                if time_left < t_panic:
                    progress = 1.0 - (max(0.0, time_left) / float(t_panic))
                    eased = float(np.power(progress, self.panic_ease_power)) if progress > 0 else 0.0
                    target_factor = max(raw_factor, self.panic_scaler)
                    final_factor = raw_factor * (1.0 - eased) + target_factor * eased

            factors.append(final_factor)
            y_final.append(y_skeleton[i] * final_factor)

        return np.array(y_final), np.array(factors)

# ==========================================
# 2. 正弦下凹模型 (SineConcaveModeler) - NEW! 🆕
# ==========================================
class CosineModeler:
    def __init__(self):
        pass

    def shape_function(self, t, Base, A, B, B_end, T_panic, T_total):
        # 1. 基础层 + 二次增长层 (Base + A * t + B * t^2)
        #    通过二次项可以更灵活地拟合中段的曲线行为（凹/凸），替代之前的线性项
        y = Base + (A * t) + (B * (t ** 2))

        # 2. 结尾层 (最后 T_panic 小时：下凹的正弦上升)
        t_start_panic = T_total - T_panic
        # 生成 rise（余弦上升），但在加入到 y 前进行灰度过渡处理：
        # 在 slope 阶段的最后4h（t_start_panic-4..t_start_panic）和
        # 余弦上升的前半段（panic 前半）之间做一个平滑混合。
        rise = np.zeros_like(t, dtype=float)
        mask_end = t > t_start_panic
        if np.any(mask_end):
            norm_t = (t[mask_end] - t_start_panic) / T_panic
            norm_t = np.clip(norm_t, 0.0, 1.0)
            # 为了把大部分增量集中在 panic 窗口的后半段，先用 sin 生成基形，
            # 然后提升幂次并应用后半段聚焦包络（只有 norm_t>0.5 时才显著增长）。
            p = 2.5
            focus_power = 3.0
            base = np.sin(norm_t * (np.pi / 2.0))
            focus = np.power(np.clip((norm_t - 0.5) / 0.5, 0.0, 1.0), focus_power)
            rise_vals = B_end * (np.power(base, p) * focus)
            rise[mask_end] = rise_vals

        # 计算混合权重：从 (t_start_panic - 4) 开始，到 (t_start_panic + T_panic/2) 完成
        blend_start = t_start_panic - 4.0
        blend_end = t_start_panic + (T_panic / 2.0)
        blend_len = max(1e-6, blend_end - blend_start)
        blend = np.clip((t - blend_start) / blend_len, 0.0, 1.0)

        # 将 rise 以 blend 权重逐步加入 y（早期不加，靠近 panic 时全加）
        y = y + (rise * blend)

        return np.maximum(y, 0)  # 物理约束：速度不能为负

    def fit(self, t_data, y_data, total_hours):
        # 参数: [Base, A, B, B_end, T_panic]
        # 初始猜测：小幅线性项和非常小的二次项
        p0 = [0.05, 0.001, 0.00001, 0.5, 24.0]

        max_panic = min(72, total_hours / 2)

        # 边界设置：允许 A 正负（轻微上/下斜），B 保持小幅以避免发散
        bounds = (
            [0.0,  -0.01,  -0.001,  0.0,    6.0],      # Lower
            [1.0,   0.01,   0.001,  10.0,   max_panic]   # Upper
        )

        try:
            func = lambda t, base, a, b, bend, tp: self.shape_function(t, base, a, b, bend, tp, total_hours)
            popt, _ = curve_fit(func, t_data, y_data, p0=p0, bounds=bounds, maxfev=40000)
            return popt
        except Exception as e:
            print(f"拟合失败，使用默认参数: {e}")
            return np.array(p0)

# ==========================================
# 3. 数据处理器 (DataHandler)
# ==========================================
class DataHandler:
    def __init__(self, target_event_id=None, debug_hours=None, output_dir=None, config_overrides=None):

        # 1. 加载配置：复制默认配置，避免修改全局变量
        self.config = DEFAULT_CONFIG.copy()
        
        # 2. 应用覆盖：如果传入了新的参数，更新配置
        if config_overrides:
            # 简单的键检查，防止传入无效参数（可选）
            valid_keys = set(self.config.keys())
            filtered_overrides = {k: v for k, v in config_overrides.items() if k in valid_keys}
            if len(filtered_overrides) < len(config_overrides):
                logger.warning(f"Ignored invalid config keys: {set(config_overrides) - valid_keys}")
            
            self.config.update(filtered_overrides)
            # print(f"已应用自定义配置参数: {list(filtered_overrides.keys())} 喵！")
            
        if target_event_id is None:
            recent = fetch_recent_json()
            target_event_id = get_current_event_for_server(recent, server_index=3)
            if target_event_id is None:
                raise ValueError("无法自动获取当前活动 ID，请手动指定 target_event_id 参数。")
        self.target_event_id = target_event_id
        self.meta = fetch_event_meta(target_event_id)
        if not self.meta: raise ValueError("元数据获取失败")
        
        self.debug_hours = debug_hours
        self.event_type = self.meta.get('event_type', 'unknown')
        print(f"目标活动: [{self.target_event_id}] 类型: {self.event_type}")
        logger.info(f"Init DataHandler for event={self.target_event_id} type={self.event_type} debug_hours={self.debug_hours} output_dir={output_dir}")
        
        # 自动探测时区偏移并传入 SeasonalityHandler
        detected_offset = self._detect_timezone_offset(self.meta['start_at'])
        # Output directory for saved plots (default: ./output)
        self.output_dir = output_dir if output_dir is not None else os.path.join('.', 'output')

        # Use configured weekend multiplier and panic ease power when creating seasonality handler
        self.seasonality = SeasonalityHandler(
            tz_offset=detected_offset,
            panic_ease_power=float(self.config.get('panic_ease_power', 1.0)),
            weekend_multiplier=float(self.config.get('weekend_multiplier', 1.0)),
            panic_scaler=float(self.config.get('panic_scaler', 1.1))
        )
        self.modeler = CosineModeler() # 👈 使用下凹正弦上升模型
        
        self.history_events = []
        self.target_data = None
        self.target_scale = 1.0
        self.debug_limit_ts = None
        # use the shared HTTP session for all network I/O in this handler
        self.session = HTTP_SESSION
        # flag: we do not own the module-level session (so close() won't shut it down)
        self._owns_session = False
        
        if debug_hours:
            self.debug_limit_ts = self.meta['start_at'] + (debug_hours * 3600 * 1000)
            print(f"[调试模式] 时间冻结在: +{debug_hours}h")
            logger.info(f"Debug mode: time frozen at +{debug_hours}h (limit_ts={self.debug_limit_ts})")

    def close(self):
        """Close any owned session. Module-level session is not owned by instances."""
        try:
            if getattr(self, '_owns_session', False) and hasattr(self, 'session') and self.session is not None:
                try:
                    self.session.close()
                except Exception:
                    pass
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_target_current_scale(self):
        url = f"{BASE_URL}eventtop/data?server={SERVER}&event={self.target_event_id}&mid=0&interval=3600000"
        try:
            data = self.session.get(url, timeout=10).json()
            if not data or "points" not in data: return None
            df = pd.DataFrame(data["points"])
            if self.debug_limit_ts: df = df[df["time"] <= self.debug_limit_ts].copy()
            if df.empty: return None
            df = df.sort_values(["uid", "time"])
            df["speed"] = df.groupby("uid")["value"].diff() / (df.groupby("uid")["time"].diff() / 60000)
            valid = df[(df["speed"] > 0) & (df["speed"] < 1000000)]["speed"]
            if valid.empty: return None
            return np.mean(valid.nlargest(3).values)
        except: return None

    def _detect_timezone_offset(self, start_ts):
        """
        根据活动开始时间推断时区偏移（小时）。
        假设活动当地开始时间通常在 10:00-19:00 之间。
        返回整数小时偏移（例如 0、8、9）。
        """
        try:
            dt_utc = datetime.fromtimestamp(start_ts / 1000, timezone.utc)
            utc_hour = dt_utc.hour
            print(f"活动开始时间 (UTC): {dt_utc} (Hour: {utc_hour})")

            # 如果 UTC 时间本身落在本地常见启动段，认为 API 已返回本地时间或为 UTC+0
            if 10 <= utc_hour <= 19:
                print("检测为 UTC/本地时间 (无需偏移)")
                return 0

            # 判断是否对应 UTC+8 的本地 10-19 区间
            if 10 <= ((utc_hour + 8) % 24) <= 19:
                print("检测为 UTC+8 (CN/CST)")
                return 8

            # 判断是否对应 UTC+9 的本地 10-19 区间 （实际搞笑用）
            if 10 <= ((utc_hour + 9) % 24) <= 19:
                print("检测为 UTC+9 (JP/JST)")
                return 9

            print("无法自动匹配时区，默认使用 UTC+8")
            return 8
        except Exception:
            return 8

    def load_target_data(self):
        print(f"获取目标活动 {self.target_event_id} 数据...")
        df = fetch_tier_1000_data(self.target_event_id)
        if df is None or df.empty: raise ValueError("T1000 数据为空")
        
        # 自动修正 start_ts 以跳过维护期
        # 1. 找到第一个 value > 0 的数据点（或者直接取第一个点，视数据源而定，通常 T1000 数据开始就是有分数的）
        # 2. 将 start_ts 修正为该数据点时间的前一个整点
        # 例如：实际第一个数据在 15:10，start_ts 修正为 15:00
        
        original_start_ts = self.meta['start_at']
        
        # 确保按时间排序
        df = df.sort_values('time')
        
        # 找到第一个有效数据点的时间戳
        first_valid_ts = None
        if 'value' in df.columns:
             # 过滤掉分数为 0 的点（如果有的话）
             valid_points = df[df['value'] > 0]
             if not valid_points.empty:
                 first_valid_ts = valid_points.iloc[0]['time']
        
        # 如果没找到（或者全为0），就用原来的
        if first_valid_ts is None:
            first_valid_ts = df.iloc[0]['time']

        # 计算修正后的 start_ts
        # 逻辑：将 first_valid_ts 转换为 datetime，向下取整到小时，再转回 timestamp
        dt_first = datetime.fromtimestamp(first_valid_ts / 1000, timezone.utc)
        # 向下取整到最近的整点
        dt_start_corrected = dt_first.replace(minute=0, second=0, microsecond=0)
        corrected_start_ts = int(dt_start_corrected.timestamp() * 1000)

        # 只有当修正后的时间比原始时间晚（说明确实有延迟/维护）时才应用，且不能晚太多（比如超过24小时就不对劲了）
        if corrected_start_ts > original_start_ts and (corrected_start_ts - original_start_ts) < 24 * 3600 * 1000:
            diff_hours = (corrected_start_ts - original_start_ts) / 3600000
            print(f"检测到维护延迟，修正活动开始时间: +{diff_hours:.1f}h")
            print(f"  原定: {datetime.fromtimestamp(original_start_ts/1000)}")
            print(f"  修正: {datetime.fromtimestamp(corrected_start_ts/1000)}")
            
            # 更新 meta 中的 start_at，这样后续所有计算都会基于这个新起点
            self.meta['start_at'] = corrected_start_ts
            start_ts = corrected_start_ts
        else:
            start_ts = original_start_ts

        # 保留未受 debug_hours 限制的完整原始数据，用于绘图真实历史值
        raw_full_df = df.copy()
        
        # If a debug limit timestamp was set in __init__, trim the fetched data to that point.
        if self.debug_limit_ts:
            df = df[df["time"] <= self.debug_limit_ts].copy()
            
        self.target_scale = self._get_target_current_scale()
        if not self.target_scale: self.target_scale = 20000
        print(f"目标 T10 极速 (Scale): {self.target_scale:.0f}")

        df = calculate_speed_tracker(df)
        df["norm_speed"] = df["speed"] / self.target_scale
        
        # 使用（可能修正过的）start_ts 计算 hours_elapsed
        df["hours_elapsed"] = (df["time"] - start_ts) / (1000 * 3600)
        
        self.target_data = df
        
        if self.debug_hours is None:
            try:
                if len(df) > 0:
                    last_time = int(df['time'].max())
                    last_hours = float(df['hours_elapsed'].max())
                    self.debug_limit_ts = last_time
                    self.debug_hours = float(last_hours)
                    logger.info(f"Auto-detected progress: debug_hours={self.debug_hours:.2f}h debug_limit_ts={self.debug_limit_ts}")
                    print(f"进度自动检测: 已观测 {self.debug_hours:.2f} 小时")
            except Exception:
                pass
        
        try:
            full_df = raw_full_df.copy()
            full_df = calculate_speed_tracker(full_df)
            full_df["norm_speed"] = full_df["speed"] / self.target_scale
            full_df["hours_elapsed"] = (full_df["time"] - start_ts) / (1000 * 3600)
            self.full_target_data = full_df
        except Exception:
            self.full_target_data = df.copy()
            
        return df

    def _process_single_candidate(self, curr):
        """
        处理单个活动 ID 的辅助函数，用于线程池调用。
        如果符合条件并成功获取数据，返回处理好的数据字典；否则返回 None。
        """
        try:
            # 1. 获取 Meta 并检查类型
            meta = fetch_event_meta(curr)
            if not meta or meta.get('event_type') != self.event_type:
                return None

            # 2. 获取 T10 极速 (Scale)
            scale = fetch_top10_max_speed(curr)
            if not scale or scale <= 0:
                return None

            # 3. 获取 T1000 历史数据
            df_hist = fetch_tier_1000_data(curr)
            if df_hist is None or df_hist.empty:
                return None

            # 4. 数据处理与归一化
            df_hist = calculate_speed_tracker(df_hist)
            df_hist['norm_speed'] = df_hist['speed'] / scale

            # 5. 时间计算 (使用 meta 中的时间，确保准确性)
            h_start = meta.get('start_at')
            h_end = meta.get('aggregate_at') or meta.get('end_at')

            if h_start is None or h_end is None:
                return None

            df_hist['hours_elapsed'] = (df_hist['time'] - h_start) / (1000 * 3600)
            total_hours = (h_end - h_start) / 3600000

            # 返回成功的数据包
            return {
                'event_id': curr,
                'scale': scale,
                'data': df_hist,
                'total_hours': total_hours,
                'start_at': h_start,
                'early_intensity': 0
            }

        except Exception as e:
            # 线程中的异常最好捕获打印，防止炸掉整个线程池
            # print(f"Error processing event {curr}: {e}")
            return None

    def find_similar_events(self, count=None):
        print(f"寻找同类 [{self.event_type}] 活动...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # 优化路径：试图使用 bestdori 提供的全量索引以快速定位同类型活动 做标记
        # 减少逐一 fetch meta 的开销。若索引不可用或解析失败，回退到线性扫描。
        candidates = []
        try:
            idx_url = "https://bestdori.com/api/events/all.3.json"
            r = self.session.get(idx_url, timeout=8)
            if r.ok:
                all_idx = r.json()
                for eid_s, meta in all_idx.items():
                    try:
                        eid = int(eid_s)
                    except Exception:
                        continue
                    # only consider events that are strictly older than the target
                    if eid >= self.target_event_id:
                        continue
                    # eventType field in index may be 'eventType'
                    et = None
                    if isinstance(meta, dict):
                        et = meta.get('eventType') or meta.get('event_type')
                    if et and isinstance(et, str) and et.lower() == str(self.event_type).lower():
                        candidates.append(eid)
                # newest first
                candidates.sort(reverse=True)
        except Exception as e:
            logger.debug(f"Failed to fetch fast index {idx_url}: {e}")
            candidates = []

        # determine desired similar count from config if not provided
        if count is None:
            count = int(self.config.get('similar_count', 5))

        found = 0
        # 建议 max_workers 设置为 4~8，太高容易被服务器拒绝服务
        max_workers = 5 
        
        # 如果 candidates 列表为空（索引失败），则生成一个回退的 ID 列表
        if not candidates:
            # 比如从 target_event_id - 1 往前推 50 个
            candidates = list(range(self.target_event_id - 1, self.target_event_id - 51, -1))

        # print(f"开始并发扫描，待选列表长度: {len(candidates)}，目标数量: {count} 喵...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            # future_to_eid 是一个字典，用于追踪哪个 future 对应哪个 event_id
            future_to_eid = {
                executor.submit(self._process_single_candidate, eid): eid 
                for eid in candidates
            }

            try:
                # as_completed 会在某个线程完成时立即 yield
                for future in as_completed(future_to_eid):
                    eid = future_to_eid[future]
                    try:
                        result = future.result()
                        if result:
                            self.history_events.append(result)
                            found += 1
                            print(f"匹配成功: Event {result['event_id']} | Scale: {result['scale']:.0f}")

                            # 如果已经找到了足够的数量，就不需要再等待剩下的了
                            if found >= count:
                                # print("已收集足够数据，停止扫描")
                                # 取消剩余未执行的任务（正在执行的无法强制中断，但不会处理结果了）
                                for f in future_to_eid:
                                    f.cancel()
                                break
                    except Exception as exc:
                        print(f"Event {eid} generated an exception: {exc}")
            except KeyboardInterrupt:
                print("手动停止了扫描")
                executor.shutdown(wait=False)
                raise

    def run_prediction(self, return_type=None):
        """
        Run the prediction pipeline.

        Parameters:
        - return_type: None (default) -> keep previous behavior (save image to disk if output_path available and print path).
                       'path' -> save plot to file and return the output_path string.
                       'fig'  -> return the matplotlib Figure object (no file save).
                       'bytes'-> return PNG image bytes (no file save).

        Returns:
        - Depending on return_type, may return None, output_path (str), matplotlib.figure.Figure, or bytes.
        """
        print("\n开始预测计算 (模式: 严格时间对齐 Time-Aligned)...")
        
        # 1. 确定对比窗口 (Comparison Window)
        # 起点：配置中指定 (默认6小时，用于跳过开局暴冲)
        # 终点：使用已观测的最新进度（若 caller 未传入 debug_hours 则已在 load_target_data 自动检测），
        # 上限由配置项 t_end_cap 控制（默认72小时）
        t_start_cmp = float(self.config.get('t_start_cmp', 6.0))
        # If debug_hours is not set, use the latest observed hours from target_data
        try:
            observed_hours = float(self.target_data['hours_elapsed'].max()) if hasattr(self, 'target_data') and len(self.target_data) > 0 else 0.0
        except Exception:
            observed_hours = 0.0
        end_source = self.debug_hours if (self.debug_hours is not None) else observed_hours
        t_end_cmp = min(end_source, float(self.config.get('t_end_cap', 72.0)))

        print(f"锁定对比区间: [ {t_start_cmp}h ~ {t_end_cmp}h ] (end_source={end_source})")
        
        # 内部函数：计算指定区间的稳健均值
        def get_window_intensity(df_in):
            # 严格卡死时间段
            mask = (df_in['hours_elapsed'] >= t_start_cmp) & \
                   (df_in['hours_elapsed'] <= t_end_cmp) & \
                   (np.isfinite(df_in['skeleton_speed']))
            
            data_slice = df_in.loc[mask, 'skeleton_speed']
            
            if len(data_slice) == 0: return None # 如果该区间没数据（比如历史活动数据缺失），返回 None
            
            # 简单的 Sigma Clipping 去除极端异常值
            mean_val = data_slice.mean()
            std_val = data_slice.std()
            if std_val > 0.001:
                clean_slice = data_slice[np.abs(data_slice - mean_val) < 2.0 * std_val]
                if len(clean_slice) > 0:
                    return clean_slice.mean()
            return mean_val

        # 2. 计算【当前活动】在该区间的强度
        target_df = self.seasonality.remove_seasonality(self.target_data)
        logger.debug(f"target_df rows={len(target_df)}; sample hours_elapsed head: {target_df['hours_elapsed'].head(5).tolist()}")
        if 'season_factor' in target_df.columns:
            logger.debug(f"season_factor sample (head): {target_df['season_factor'].head(5).tolist()}")
        curr_intensity = get_window_intensity(target_df)
        
        if curr_intensity is None:
            print("当前活动在对比区间内无有效数据，无法计算 Ratio，默认 1.0")
            curr_intensity = 0.1 # 避免除零
            
        print(f"当前活动区间强度: {curr_intensity:.4f}")

        # 3. 计算【历史活动】在【同一区间】的强度 & 拟合参数
        hist_params = []
        hist_intensities = []
        
        for h in self.history_events:
            df = h['data']
            df_clean = self.seasonality.remove_seasonality(df)
            
            # A. 拟合全量参数 (用于获取形状 Slope, Panic 等)
            #    丢弃第一天 18:00 之前的数据用于拟合（若存在）
            popt = None
            try:
                h_start_ts = h.get('start_at')
                df_for_fit = df_clean.copy()
                if h_start_ts is not None:
                    start_dt = datetime.fromtimestamp(h_start_ts / 1000, timezone.utc) + timedelta(hours=self.seasonality.tz_offset)
                    cutoff_dt = start_dt.replace(hour=18, minute=0, second=0, microsecond=0)
                    cutoff_ts = int(cutoff_dt.timestamp() * 1000)
                    # 若 cutoff 在 start 之前（即活动在当天 18:00 之后开始），则不会丢弃任何数据
                    df_for_fit = df_clean.loc[df_clean['time'] >= cutoff_ts].copy()

                valid_mask = np.isfinite(df_for_fit['skeleton_speed'])
                if valid_mask.sum() >= 5:
                    popt = self.modeler.fit(
                        df_for_fit.loc[valid_mask, 'hours_elapsed'].values,
                        df_for_fit.loc[valid_mask, 'skeleton_speed'].values,
                        h['total_hours']
                    )
                    logger.debug(f"Hist {h['event_id']} fit rows={valid_mask.sum()} used (cutoff applied)")
                else:
                    logger.info(f"Hist {h['event_id']} too few rows after cutoff ({valid_mask.sum()}), skipping fit")
            except Exception as e:
                logger.warning(f"Hist {h['event_id']} fit failed: {e}")
            
            # B. 计算同一时间窗口的强度
            h_int = get_window_intensity(df_clean)
            
            if h_int is not None and popt is not None:
                hist_intensities.append(h_int)
                hist_params.append(popt)
                # popt: [Base, A, B, B_end, T_panic]
                logger.info(f"  - Hist {h['event_id']}: 区间强度={h_int:.6f} | A={popt[1]:.8e} B={popt[2]:.8e} | params={popt}")
            else:
                logger.info(f"  - Hist {h['event_id']}: 在该时间段无数据，跳过对比")

        # 诊断：比较 norm_speed ratio vs skeleton ratio
        mask_cmp = (target_df['hours_elapsed'] >= t_start_cmp) & (target_df['hours_elapsed'] <= t_end_cmp)
        obs_norm_mean = target_df.loc[mask_cmp, 'norm_speed'].mean()
        logger.info(f"DIAG obs_norm_mean={obs_norm_mean:.6f}, obs_skel_mean={curr_intensity:.6f}")
        
        # 汇总历史 norm_speed 同窗均值
        hist_norms = []
        for h in self.history_events:
            dfh = self.seasonality.remove_seasonality(h['data'])
            maskh = (dfh['hours_elapsed'] >= t_start_cmp) & (dfh['hours_elapsed'] <= t_end_cmp)
            if maskh.any():
                hist_norms.append(dfh.loc[maskh, 'norm_speed'].mean())
        logger.info(f"DIAG hist_norms={hist_norms}")
        if hist_norms:
            logger.info(f"DIAG norm_ratio = {obs_norm_mean / np.mean(hist_norms):.6f}")

        # 4. 计算 Ratio — 使用双重度量并保守处理异常值
        #  - skeleton_ratio: 基于去节律化后的 skeleton_speed（更接近模型形状）
        #  - norm_ratio: 基于原始 norm_speed 的观测比（更贴近真实观测）
        if hist_intensities:
            avg_hist_intensity = np.mean(hist_intensities)
            skeleton_ratio = curr_intensity / avg_hist_intensity if avg_hist_intensity > 0 else np.nan

            # 计算历史 norm_speed 的均值（若可得），用于 norm_ratio
            norm_ratio = None
            mean_hist_norm = None
            if hist_norms:
                mean_hist_norm = np.mean(hist_norms)
                if mean_hist_norm and mean_hist_norm > 0:
                    norm_ratio = obs_norm_mean / mean_hist_norm

            # 选择性地混合两种 ratio：当两者都可用时按时间权重混合（从 skeleton 主导 -> 到 normal 主导）
            skeleton_val = float(skeleton_ratio) if np.isfinite(skeleton_ratio) else None
            norm_val = float(norm_ratio) if (norm_ratio is not None and np.isfinite(norm_ratio)) else None

            # compute observed progress s (use configured target window length if available)
            target_total_hours = float(self.config.get('t_end_cap', 72.0))
            observed_hours = float(target_df['hours_elapsed'].max()) if 'hours_elapsed' in target_df.columns else 0.0
            s = 0.0
            if target_total_hours and target_total_hours > 0:
                s = np.clip(observed_hours / float(target_total_hours), 0.0, 1.0)
            if (s is None): s = 0.0

            if skeleton_val is not None and norm_val is not None:
                w_norm = 0.2 + 0.6 * np.cos(s * np.pi - np.pi)
                w_norm = float(np.clip(w_norm, 0.0, 1.0))
                chosen_ratio = skeleton_val * (1.0 - w_norm) + norm_val * w_norm
                logger.info(f"Blend ratios using time-weight: s={s:.3f} w_norm={w_norm:.3f} skeleton={skeleton_val:.6f} norm={norm_val:.6f} -> chosen={chosen_ratio:.6f}")
            else:
                # fallback to whichever is available, else 1.0
                if skeleton_val is not None:
                    chosen_ratio = skeleton_val
                elif norm_val is not None:
                    chosen_ratio = norm_val
                else:
                    chosen_ratio = 1.0

            # Clip ratio 以防极端放大/缩小（阈值来自配置）
            R_MIN = float(self.config.get('ratio_min', 0.25))
            R_MAX = float(self.config.get('ratio_max', 4.0))
            clipped_ratio = float(np.clip(chosen_ratio, R_MIN, R_MAX))

            # 记录诊断信息
            logger.info(
                "Ratio diagnostics: skeleton_ratio=%s norm_ratio=%s chosen=%s clipped=%s avg_hist_int=%s",
                (f"{skeleton_ratio:.6f}" if np.isfinite(skeleton_ratio) else "nan"),
                (f"{norm_ratio:.6f}" if norm_ratio is not None else "n/a"),
                f"{chosen_ratio:.6f}", f"{clipped_ratio:.6f}", f"{avg_hist_intensity:.6f}"
            )
            if clipped_ratio != chosen_ratio:
                logger.warning(f"Ratio clipped from {chosen_ratio:.6f} to {clipped_ratio:.6f} (bounds {R_MIN}-{R_MAX})")

            ratio = clipped_ratio
            print(f"强度修正比率 (skeleton/norm/chosen/clipped): {skeleton_ratio:.6f} / {(norm_ratio if norm_ratio is not None else float('nan')):.6f} -> {chosen_ratio:.6f} -> {ratio:.6f}")
        else:
            print("没有有效的历史对比数据，Ratio 重置为 1.0")
            ratio = 1.0
            avg_hist_intensity = 1.0 # dummy

        # 5.  参数修正与预测
        if hist_params:
            avg_params = np.mean(hist_params, axis=0)
        else:
            # Default: [Base, A, B, B_end, T_panic]
            avg_params = np.array([0.05, 0.001, 0.0, 0.5, 24.0])

        pred_params = avg_params.copy()

        # Apply Ratio to parameters: scale Base, linear A and quadratic B moderately,
        # and magnify B_end slightly as before. T_panic remains unchanged.
        # Param order: [Base, A, B, B_end, T_panic]
        pred_params[0] *= ratio        # Base
        pred_params[1] *= ratio        # A (linear)
        pred_params[2] *= ratio        # B (quadratic)
        pred_params[3] *= (ratio ** 1.1)  # B_end

        target_total_hours = (self.meta['end_at'] - self.meta['start_at']) / 3600000
        
        try:
            # 标准化终端输出格式：更高精度且统一展示所有参数
            print(
                f"预测参数: Base={pred_params[0]:.6f}, "
                f"A={pred_params[1]:.8e}, B={pred_params[2]:.8e}, "
                f"B_end={pred_params[3]:.6f}, T_panic={int(pred_params[4])}"
            )
            logger.info(
                f"Final pred_params: Base={pred_params[0]:.6f}, A={pred_params[1]:.8e}, "
                f"B={pred_params[2]:.8e}, B_end={pred_params[3]:.6f}, T_panic={int(pred_params[4])}"
            )
            logger.debug(f"avg_params: {avg_params}; hist_intensities: {hist_intensities}")
            logger.debug(f"target_scale={self.target_scale}, target_total_hours={target_total_hours}, debug_hours={self.debug_hours}")
        except:
            pass

        # 生成曲线 (后续绘图逻辑不变)
        future_t = np.linspace(0, target_total_hours, 1000) 
        skeleton_pred = self.modeler.shape_function(future_t, *pred_params, target_total_hours)
        speed_pred, _ = self.seasonality.apply_seasonality(
            future_t, skeleton_pred, self.meta['start_at'],
            total_hours=target_total_hours, t_panic=pred_params[4]
        )

        # 积分逻辑前置：准备观测当前分数和时间（用于后续 scaling 诊断）
        if 'ep' in self.target_data.columns:
            score_series = self.target_data['ep']
        elif 'value' in self.target_data.columns:
            score_series = self.target_data['value']
        else:
            score_series = pd.Series(np.zeros(len(self.target_data)), index=self.target_data.index)

        current_max_score = score_series.max()
        current_max_time = self.target_data['hours_elapsed'].max()

        # ---- Final output scaling: align model's cutoff->now mass to observed cutoff->now mass
        # This computes model cumulative since first-day 18:00 and compares to observed
        # cumulative in the same interval, then scales future increments accordingly.
        # (If insufficient data or zero model mass, scale factor defaults to 1.0.)
        try:
            # prepare arrays
            real_speed_ep_min_all = speed_pred * self.target_scale
            if len(future_t) > 1:
                dt_hours_all = float(future_t[1] - future_t[0])
            else:
                dt_hours_all = 0.0
            dt_min_all = dt_hours_all * 60.0
            cum_all = np.cumsum(real_speed_ep_min_all * dt_min_all)

            # cutoff hours (first day 18:00)
            start_ts = self.meta['start_at']
            start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone(timedelta(hours=self.seasonality.tz_offset)))
            cutoff_dt = start_dt.replace(hour=18, minute=0, second=0, microsecond=0)
            cutoff_ts = int(cutoff_dt.timestamp() * 1000)
            cutoff_hours = (cutoff_ts - start_ts) / (1000.0 * 3600.0)
            if cutoff_hours < 0.0:
                cutoff_hours = 0.0

            # map to indices
            idx_cutoff = int(np.searchsorted(future_t, cutoff_hours, side='left'))
            idx_now = int(np.searchsorted(future_t, current_max_time, side='right') - 1)
            idx_cutoff = max(0, min(idx_cutoff, len(cum_all) - 1))
            idx_now = max(0, min(idx_now, len(cum_all) - 1))

            model_since_cutoff = float(cum_all[idx_now] - (cum_all[idx_cutoff-1] if idx_cutoff > 0 else 0.0))

            # observed cumulative before cutoff
            # Use hours_elapsed to avoid timezone/timestamp misalignment; if no exact
            # pre-cutoff row exists, fall back to the nearest earlier point (interpolation).
            hist_df = getattr(self, 'full_target_data', self.target_data)
            hist_col = 'ep' if 'ep' in hist_df.columns else ('value' if 'value' in hist_df.columns else None)
            observed_before_cutoff = 0.0
            if hist_col is not None and 'hours_elapsed' in hist_df.columns:
                hrs = hist_df['hours_elapsed'].values
                scores = hist_df[hist_col].values
                # prefer rows strictly before cutoff_hours
                before_mask = hrs < cutoff_hours
                if np.any(before_mask):
                    # take latest available score before cutoff
                    observed_before_cutoff = float(scores[np.where(before_mask)[0][-1]])
                else:
                    # no earlier row; attempt to use the earliest available score (usually 0)
                    observed_before_cutoff = float(scores[0]) if len(scores) > 0 else 0.0
                logger.debug(f"Observed before cutoff via hours_elapsed: cutoff_hours={cutoff_hours:.3f} observed_before_cutoff={observed_before_cutoff}")
            else:
                # fallback to timestamp-based method if hours_elapsed missing
                if hist_col is not None and 'time' in hist_df.columns:
                    before_mask = hist_df['time'] < cutoff_ts
                    if before_mask.any():
                        observed_before_cutoff = float(hist_df.loc[before_mask, hist_col].iloc[-1])

            observed_since_cutoff = float(current_max_score) - observed_before_cutoff

            if model_since_cutoff > 0 and observed_since_cutoff >= 0:
                raw_scale = observed_since_cutoff / model_since_cutoff
            else:
                raw_scale = 1.0

            SCALE_MIN = float(self.config.get('scale_min', 0.5))
            SCALE_MAX = float(self.config.get('scale_max', 2.0))
            scale_factor = float(np.clip(raw_scale, SCALE_MIN, SCALE_MAX))
            if scale_factor != raw_scale:
                logger.warning(f"Output scaling clipped {raw_scale:.6f} -> {scale_factor:.6f}")
            logger.info(f"Output scaling diagnostics: observed_since_cutoff={observed_since_cutoff:.1f} model_since_cutoff={model_since_cutoff:.1f} raw={raw_scale:.6f} applied={scale_factor:.6f}")

            # --- 回测修正 (t vs t-24) ---
            # 如果当前观测时间超过 50 小时，则用 t-24 的窗口回测：
            # 1) 将模型在 [t-24, t] 的预测累计量与真实累计量比较
            # 2) 得到一个额外的修正因子 applied_correction，并乘到最终 scale_factor 上
            try:
                applied_scale_factor = scale_factor
                now_hours = float(current_max_time)
                if now_hours > 50.0:
                    t0 = max(0.0, now_hours - 24.0)
                    # 在 future_t 上定位索引
                    idx_t0 = int(np.searchsorted(future_t, t0, side='left'))
                    idx_now = int(np.searchsorted(future_t, now_hours, side='right') - 1)
                    idx_t0 = max(0, min(idx_t0, len(future_t) - 1))
                    idx_now = max(0, min(idx_now, len(future_t) - 1))

                    # 计算模型在 [t0, now] 的预测累计（使用当前 scale_factor）
                    if idx_now > idx_t0:
                        dt_hours_all = float(future_t[1] - future_t[0]) if len(future_t) > 1 else 0.0
                        dt_min_all = dt_hours_all * 60.0
                        pred_segment = speed_pred[idx_t0:idx_now]
                        model_24 = float(np.sum(pred_segment) * self.target_scale * scale_factor * dt_min_all)

                        # 取真实历史累计：找到 t0 之前最近的历史得分点
                        hist_df = getattr(self, 'full_target_data', self.target_data)
                        hist_col = 'ep' if 'ep' in hist_df.columns else ('value' if 'value' in hist_df.columns else None)
                        observed_before_t0 = None
                        if hist_col is not None:
                            hrs = hist_df['hours_elapsed'].values
                            scores = hist_df[hist_col].values
                            pos = int(np.searchsorted(hrs, t0, side='left'))
                            if pos > 0:
                                observed_before_t0 = float(scores[pos-1])
                            else:
                                observed_before_t0 = float(scores[0]) if len(scores) > 0 else 0.0

                        if observed_before_t0 is not None:
                            observed_24 = float(current_max_score) - observed_before_t0
                        else:
                            observed_24 = None

                        if (model_24 > 0) and (observed_24 is not None) and (observed_24 >= 0):
                            raw_corr = observed_24 / model_24 if model_24 > 0 else 1.0
                            CORR_MIN = float(self.config.get('corr_min', 0.6))
                            CORR_MAX = float(self.config.get('corr_max', 1.6))
                            corr = float(np.clip(raw_corr, CORR_MIN, CORR_MAX))
                            if corr != raw_corr:
                                logger.warning(f"24h correction clipped {raw_corr:.6f} -> {corr:.6f}")
                            applied_scale_factor = float(scale_factor * corr)
                            logger.info(f"24h backtest diagnostics: observed_24={observed_24:.1f} model_24={model_24:.1f} raw_corr={raw_corr:.6f} applied_corr={corr:.6f} final_scale={applied_scale_factor:.6f}")
                        else:
                            logger.info("24h backtest skipped due to insufficient data or zero model mass")

                else:
                    applied_scale_factor = scale_factor
            except Exception as e:
                logger.warning(f"24h backtest failed: {e}")
                applied_scale_factor = scale_factor
        except Exception as e:
            logger.warning(f"Failed to compute output scaling: {e}")
            scale_factor = 1.0
            applied_scale_factor = 1.0
        
        # 积分逻辑（使用前面已准备好的 score_series/current_max_* 变量）
        future_mask = future_t >= current_max_time
        future_t_clip = future_t[future_mask]
        speed_pred_clip = speed_pred[future_mask]
        
        if len(future_t_clip) > 0:
            # Apply both the primary scale_factor and the optional 24h backtest correction
            final_scale = locals().get('applied_scale_factor', scale_factor)

            # --- Top-speed smoothing: when normalized speed (after scale) > 0.5,
            # apply a smooth attenuation so growth flattens approaching the top line.
            # This operates in normalized units (relative to target_scale) AFTER final_scale.
            try:
                norm_after_scale = speed_pred_clip * float(final_scale)
                # Two-stage attenuation with configured thresholds
                THRESH1 = float(self.config.get('smooth_thresh1', 0.5))
                THRESH2 = float(self.config.get('smooth_thresh2', 0.65))
                HARD_CAP = float(self.config.get('smooth_hard_cap', 0.8))
                ALPHA = 3.0   # mild stage coefficient
                BETA = 22.0   # strong stage coefficient (large -> heavy compression)

                norm_adj = norm_after_scale.copy()

                # Stage 1: mild attenuation between THRESH1 and THRESH2
                mask_stage1 = (norm_after_scale > THRESH1) & (norm_after_scale <= THRESH2)
                if np.any(mask_stage1):
                    excess1 = (norm_after_scale[mask_stage1] - THRESH1) / (THRESH2 - THRESH1)
                    attenuation1 = 1.0 / (1.0 + ALPHA * excess1)
                    norm_adj[mask_stage1] = THRESH1 + (norm_after_scale[mask_stage1] - THRESH1) * attenuation1

                # Stage 2: strong attenuation above THRESH2 (quadratic penalization)
                mask_stage2 = norm_after_scale > THRESH2
                if np.any(mask_stage2):
                    excess2 = (norm_after_scale[mask_stage2] - THRESH2) / (1.0 - THRESH2)
                    attenuation2 = 1.0 / (1.0 + BETA * (excess2 ** 2))
                    norm_adj[mask_stage2] = THRESH2 + (norm_after_scale[mask_stage2] - THRESH2) * attenuation2

                # Enforce hard cap
                norm_adj = np.minimum(norm_adj, HARD_CAP)

                if np.any(norm_adj != norm_after_scale):
                    logger.info(f"Top-smoothing applied (stage1>{THRESH1}, stage2>{THRESH2}, cap={HARD_CAP})")

                # convert back to real speed per minute
                real_speed_ep_min = norm_adj * self.target_scale
            except Exception as e:
                logger.warning(f"Top-smoothing failed: {e}")
                real_speed_ep_min = speed_pred_clip * self.target_scale * final_scale
            dt_hours = (future_t_clip[1] - future_t_clip[0]) if len(future_t_clip) > 1 else 0
            dt_min = dt_hours * 60
            score_increment = np.cumsum(real_speed_ep_min * dt_min)
            score_pred = current_max_score + score_increment
            full_t_score = np.concatenate([self.target_data['hours_elapsed'].values, future_t_clip])
            full_score = np.concatenate([score_series.values, score_pred])
        else:
            full_t_score = self.target_data['hours_elapsed'].values
            full_score = score_series.values

        # --- Ensure plotted predicted speed matches the final adjusted curve used
        # in cumulative computations (apply final scaling + top-smoothing + cap
        # across the full prediction vector for plotting consistency).
        try:
            final_scale_for_plot = float(locals().get('applied_scale_factor', locals().get('scale_factor', 1.0)))
            adj_norm_full = speed_pred * final_scale_for_plot

            # reuse same smoothing parameters as applied earlier (from config)
            THRESH1 = float(self.config.get('smooth_thresh1', 0.5))
            THRESH2 = float(self.config.get('smooth_thresh2', 0.65))
            HARD_CAP = float(self.config.get('smooth_hard_cap', 0.8))
            ALPHA = 3.0
            BETA = 22.0

            norm_adj_full = adj_norm_full.copy()
            mask_stage1 = (adj_norm_full > THRESH1) & (adj_norm_full <= THRESH2)
            if np.any(mask_stage1):
                excess1 = (adj_norm_full[mask_stage1] - THRESH1) / (THRESH2 - THRESH1)
                attenuation1 = 1.0 / (1.0 + ALPHA * excess1)
                norm_adj_full[mask_stage1] = THRESH1 + (adj_norm_full[mask_stage1] - THRESH1) * attenuation1

            mask_stage2 = adj_norm_full > THRESH2
            if np.any(mask_stage2):
                excess2 = (adj_norm_full[mask_stage2] - THRESH2) / (1.0 - THRESH2)
                attenuation2 = 1.0 / (1.0 + BETA * (excess2 ** 2))
                norm_adj_full[mask_stage2] = THRESH2 + (adj_norm_full[mask_stage2] - THRESH2) * attenuation2

            norm_adj_full = np.minimum(norm_adj_full, HARD_CAP)
        except Exception:
            norm_adj_full = speed_pred.copy()

        # Pass the adjusted normalized prediction into plotting so visual matches numeric output
        # Build output filename: default ./output/pred_{eventid}_{timestamp}.png
        try:
            ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f"pred_{self.target_event_id}_{ts_str}.png"
            output_path = os.path.join(self.output_dir, f"{self.target_event_id}/", fname)
        except Exception:
            output_path = None

        # call plot_final and optionally capture return; protect with try/except
        try:
            print("开始绘图/渲染输出...")
            plot_ret = self.plot_final(
                target_df, future_t, skeleton_pred, norm_adj_full, full_t_score, full_score,
                output_path=output_path, return_type=return_type
            )
            # diagnostic: report what plot_final returned
            try:
                tname = type(plot_ret).__name__ if plot_ret is not None else 'NoneType'
            except Exception:
                tname = str(type(plot_ret))
            # print(f"绘图返回: type={tname}")
            logger.info(f"plot_final returned type={tname}")
            if return_type == 'bytes' and plot_ret is not None:
                try:
                    # print(f"绘图字节大小: {len(plot_ret)} bytes")
                    logger.info(f"Plot bytes size: {len(plot_ret)}")
                except Exception:
                    pass
        except Exception as e:
            plot_ret = None
            logger.exception(f"plot_final failed: {e}")
            print(f"绘图失败: {e}")

        logger.info(f"Prediction complete for event={self.target_event_id}; final_score={int(full_score[-1]) if len(full_score)>0 else 0}")
        logger.info("---- END RUN ----\n")

        # If caller requested a return value, return it
        if return_type in ('path', 'fig', 'bytes'):
            return plot_ret
        return None

    def plot_final(self, target_df, t_pred, y_skeleton, y_final, t_score, y_score, output_path=None, return_type=None):
        """
        Draw prediction plots with Real Date-Time X-axis (Fixed for Timezone Alignment).
        Forces all timestamps to be naive local time to prevent matplotlib auto-conversion issues.
        """
        # ensure output directory exists when saving
        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to create output directory {out_dir}: {e}")

        # 1. 准备时间转换基准 (Base Timestamp)
        start_ts = self.meta['start_at']
        tz_offset = self.seasonality.tz_offset
        
        # 先转成 UTC aware，加上偏移量变成当地时间，然后立刻 replace(tzinfo=None) 剥离时区标签。
        # 这样得到的是一个“看起来是当地时间，但没有任何时区包袱”的纯净时间对象。
        # 例如：原本是 UTC 12:00 (aware)，+8h -> Local 20:00 (aware) -> strip -> Local 20:00 (naive)
        start_dt_utc = datetime.fromtimestamp(start_ts / 1000, timezone.utc)
        start_dt_local = (start_dt_utc + timedelta(hours=tz_offset)).replace(tzinfo=None)

        # 2. 辅助函数：将相对小时数数组转换为 naive local datetime 数组
        def to_real_time(hours_array):
            deltas = pd.to_timedelta(hours_array, unit='h')
            return start_dt_local + deltas

        # 3. 转换各个数据源的时间轴
        # A. 观测数据 (target_df)
        # pd.to_datetime(unit='ms') 默认生成 naive UTC。
        # 我们直接加上 Timedelta(hours=offset)，让它在数值上变成当地时间，且保持 naive 状态。
        if 'time' in target_df.columns:
            obs_time = pd.to_datetime(target_df['time'], unit='ms') + pd.Timedelta(hours=tz_offset)
        else:
            obs_time = to_real_time(target_df['hours_elapsed'].values)
        
        # B. 预测数据 (numpy arrays) -> 这里的基准已经是 start_dt_local (naive) 了
        t_pred_dt = to_real_time(t_pred)
        t_score_dt = to_real_time(t_score)
        
        # C. 当前 Debug 时间点
        debug_dt = start_dt_local + timedelta(hours=self.debug_hours)

        # Use object-oriented Figure
        fig = Figure(figsize=(12, 10))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        
        # --- 设置日期格式化器 ---
        date_fmt = mdates.DateFormatter('%m-%d %H:%M')
        date_loc = mdates.AutoDateLocator()

        # --- 子图1: 速度曲线 ---
        ax1.scatter(obs_time, target_df['skeleton_speed'], 
                    s=10, color='gray', alpha=0.3, label='Observed Skeleton')
        ax1.plot(t_pred_dt, y_skeleton, color='blue', linestyle='--', alpha=0.5, label='Predicted Skeleton')
        
        ax1.plot(obs_time, target_df['norm_speed'], 
                 color='red', linewidth=2, label='Observed Speed')
        ax1.plot(t_pred_dt, y_final, color='green', linewidth=2, alpha=0.8, label='Predicted Speed')

        # 绘制真实的“未来”速度 (如果有 full_target_data)
        try:
            if hasattr(self, 'full_target_data') and hasattr(self, 'target_data'):
                obs_end_hours = float(self.target_data['hours_elapsed'].max()) if len(self.target_data) > 0 else 0
                full_df = self.full_target_data
                mask_future_real = full_df['hours_elapsed'].values > obs_end_hours
                if np.any(mask_future_real):
                    future_real_hours = full_df['hours_elapsed'].values[mask_future_real]
                    future_real_time = to_real_time(future_real_hours)
                    ax1.plot(future_real_time,
                             full_df['norm_speed'].values[mask_future_real],
                             color='orange', linestyle='-.', linewidth=2, alpha=0.9, label='Actual Future Speed')
        except Exception:
            pass

        ax1.axvline(x=debug_dt, color='black', linestyle=':', label='Now')
        ax1.set_ylabel("Normalized Speed")
        ax1.set_title(f"Event {self.target_event_id} Speed Prediction (Local Time, UTC+{tz_offset})")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        ax1.xaxis.set_major_formatter(date_fmt)
        ax1.xaxis.set_major_locator(date_loc)

        # --- 子图2: 累计分数曲线 ---
        hist_df = getattr(self, 'full_target_data', self.target_data)
        if 'ep' in hist_df.columns:
            hist_score = hist_df['ep'].values
        elif 'value' in hist_df.columns:
            hist_score = hist_df['value'].values
        else:
            hist_score = np.zeros(len(hist_df))

        # 历史数据时间轴转换 (同样确保 naive)
        if 'time' in hist_df.columns:
            hist_time_dt = pd.to_datetime(hist_df['time'], unit='ms') + pd.Timedelta(hours=tz_offset)
        else:
            hist_time_dt = to_real_time(hist_df['hours_elapsed'].values)

        ax2.plot(hist_time_dt, hist_score, color='red', linewidth=2, label='Observed Score')

        try:
            if len(hist_time_dt) > 0:
                # 处理 Series 或 ndarray 的取值
                obs_final_t = hist_time_dt.iloc[-1] if hasattr(hist_time_dt, 'iloc') else hist_time_dt[-1]
                obs_final_val = float(hist_score[-1])
                ax2.scatter(obs_final_t, obs_final_val, color='darkred', s=50, zorder=5, label='Observed Final')
                ax2.text(obs_final_t, obs_final_val, f"Obs: {int(obs_final_val):,}",
                         ha='left', va='bottom', fontsize=10, color='darkred')
        except Exception:
            pass

        if hasattr(self, 'target_data') and len(self.target_data) > 0:
            obs_end_hours = float(self.target_data['hours_elapsed'].max())
        else:
            obs_end_hours = 0
        
        pred_mask = np.array(t_score) > obs_end_hours
        
        if np.any(pred_mask):
            t_pred_only_dt = np.array(t_score_dt)[pred_mask]
            y_pred_only = np.array(y_score)[pred_mask]
            ax2.plot(t_pred_only_dt, y_pred_only, color='purple', linestyle='--', linewidth=2, label='Predicted Future')
            
            final_score = y_pred_only[-1]
            final_time = t_pred_only_dt[-1]
            ax2.text(final_time, final_score, f"{int(final_score):,}", 
                     ha='right', va='bottom', fontsize=12, fontweight='bold', color='purple')
        else:
            final_score = float(hist_score[-1]) if len(hist_score) > 0 else 0.0

        ax2.axvline(x=debug_dt, color='black', linestyle=':')
        ax2.set_ylabel("Cumulative Event Points")
        ax2.set_xlabel(f"Date Time (Local)")
        ax2.set_title(f"Final Prediction: {int(final_score):,} PT")
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        ax2.xaxis.set_major_formatter(date_fmt)
        ax2.xaxis.set_major_locator(date_loc)
        
        fig.autofmt_xdate()

        try:
            fig.tight_layout()
        except Exception:
            pass

        try:
            wm_text = "@byydzh mycx 1000"
            fig.text(0.99, 0.01, wm_text, fontsize=10, color='gray', alpha=0.6,
                     ha='right', va='bottom', zorder=100)
        except Exception as e:
            logger.debug(f"Failed to draw watermark: {e}")

        if return_type == 'fig':
            return fig

        if return_type == 'bytes':
            try:
                buf = BytesIO()
                FigureCanvasAgg(fig).print_png(buf)
                buf.seek(0)
                img_bytes = buf.getvalue()
                buf.close()
                try:
                    fig.clf()
                    del fig
                except Exception:
                    pass
                return img_bytes
            except Exception as e:
                logger.warning(f"Failed to render plot to bytes: {e}")
                try:
                    fig.clf()
                    del fig
                except Exception:
                    pass
                return None

        if output_path:
            try:
                fig.savefig(output_path, dpi=150)
                print(f"预测图已保存: {output_path}")
                logger.info(f"Saved prediction plot to {output_path}")
            except Exception as e:
                logger.warning(f"Failed to save plot to {output_path}: {e}")
                try:
                    buf = BytesIO()
                    FigureCanvasAgg(fig).print_png(buf)
                    buf.close()
                except Exception:
                    pass
        else:
            try:
                buf = BytesIO()
                FigureCanvasAgg(fig).print_png(buf)
                buf.close()
            except Exception:
                pass

        print(f"最终预测分数: {int(final_score):,} PT")
        try:
            fig.clf()
            del fig
        except Exception:
            pass
        if output_path:
            return output_path
        return None

# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    try:
        # 假设预测 xxx，时间冻结在 xxh
        # handler = DataHandler(276, debug_hours=60)
        handler = DataHandler()
        handler.load_target_data()
        handler.find_similar_events()
        
        if handler.history_events:
            handler.run_prediction()
        else:
            print("没找到历史活动，无法预测。")
            
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
