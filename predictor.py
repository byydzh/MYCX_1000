import requests
import pandas as pd
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
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

# Setup logger for detailed run diagnostics (file-only; do not print logs to terminal)
LOG_PATH = os.path.join(os.path.dirname(__file__), 'predictor.log')
logger = logging.getLogger('predictor')
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_PATH, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Disable propagation to avoid duplicate console output from root logger
    logger.propagate = False

# ==========================================
# 1. 昼夜节律处理器 (SeasonalityHandler)
# ==========================================
class SeasonalityHandler:
    def __init__(self, json_path='base_speed_distribution.json', tz_offset=8, panic_ease_power=1.0):
        self.data = self._load_json(json_path)
        self.global_mean = self._calculate_global_mean()
        # 注意: `base_speed_distribution.json` 中的小时数据已经是当地时区的小时
        # 因此不再对时间戳进行额外的时区偏移。保留 tz_offset 属性仅作兼容。
        self.tz_offset = tz_offset
        # panic_ease_power 控制在 panic 窗口中 progress->eased 的速率，
        # 值越大越慢接近 target（例如 2.0 或 3.0 会比 0.8 慢得多）
        self.panic_ease_power = float(panic_ease_power)
        print(f"📅 昼夜节律数据已加载（使用本地小时），全局基准速度均值: {self.global_mean:.4f} panic_ease_power={self.panic_ease_power}")

    def _load_json(self, path):
        if not os.path.exists(path):
            print(f"⚠️ 警告：找不到 {path}，将不使用节律修正。")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _calculate_global_mean(self):
        if not self.data: return 1.0
        values = []
        for dtype in ['weekday', 'weekend']:
            for h in range(24):
                item = self.data.get(dtype, {}).get(str(h))
                if item and item['count'] > 0:
                    values.append(item['mean'])
        return np.mean(values) if values else 1.0

    def get_factor(self, dt):
        if not self.data: return 1.0
        if isinstance(dt, (int, float)): 
            dt_obj = datetime.fromtimestamp(dt / 1000)
        else:
            dt_obj = dt
            
        if is_workday is not None:
            try:
                if not is_workday(dt_obj.date()): dtype = 'weekend'
                else: dtype = 'weekday'
            except: dtype = get_day_type(dt_obj)
        else:
            dtype = get_day_type(dt_obj)
            
        hour = str(dt_obj.hour)
        stats = self.data.get(dtype, {}).get(hour)
        if stats and stats['mean'] > 0:
            return stats['mean'] / self.global_mean
        return 1.0 

    def remove_seasonality(self, df):
        df = df.copy()
        # `base_speed_distribution.json` uses local hours. Convert timestamp to naive local
        # datetime (no additional tz shift) so hour lookup matches the JSON keys.
        df['dt_local'] = pd.to_datetime(df['time'], unit='ms')
        df['season_factor'] = df['dt_local'].apply(self.get_factor)

        # --- Early-hour suppression: avoid amplifying early spikes by tiny factors ---
        # If the dataframe contains a precomputed `hours_elapsed` column, treat the
        # first 12 hours as an 'adrenaline' period and prevent season_factor < 1.0
        # from shrinking norm_speed (which would inflate skeleton_speed).
        if 'hours_elapsed' in df.columns:
            mask_early = (df['hours_elapsed'] < 12.0) & (df['season_factor'] < 1.0)
            if mask_early.any():
                df.loc[mask_early, 'season_factor'] = 1.0

        df['skeleton_speed'] = df['norm_speed'] / df['season_factor']
        return df

    def apply_seasonality(self, t_hours, y_skeleton, start_ts, total_hours=None, t_panic=24.0):
        """
        Apply seasonality factors to a skeleton speed series.

        Extended with an "adrenaline" mechanism: when time enters the final
        `t_panic` hours (counting down from `total_hours`), the seasonality
        effect is linearly blended back toward a neutral factor (>=1.0) to
        simulate players ignoring day/night and ramping up performance.
        """
        y_final = []
        factors = []

        for i, h in enumerate(t_hours):
            current_ts = start_ts + (h * 3600 * 1000)
            # base_speed_distribution.json 存储的是本地小时，因此直接将时间戳转为本地
            # （naive）datetime 用于查表，不再额外添加 tz 偏移。
            dt_local = datetime.fromtimestamp(current_ts / 1000)

            # 1) 原始节律因子（基于本地小时）
            raw_factor = self.get_factor(dt_local)

            # 2) 恐慌/肾上腺素修正（如果提供了 total_hours）
            final_factor = raw_factor
            if (total_hours is not None) and (t_panic is not None) and t_panic > 0:
                time_left = total_hours - h
                if time_left < t_panic:
                    # progress: 0.0 -> 1.0 as we move through panic window
                    progress = 1.0 - (max(0.0, time_left) / float(t_panic))

                    # 使用可配置的非线性缓动：增大幂次可使得在 panic 窗口早期更慢接近目标因子
                    eased = float(np.power(progress, self.panic_ease_power)) if progress > 0 else 0.0

                    # 目标因子提高到更高水平以模拟更激进的冲刺期（可调）
                    target_factor = max(raw_factor, 1.0)

                    # 非线性插值（快速上升到 target_factor）
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
        """
        t: 时间数组
        A_start: 开局倍率 (前24小时)
        Base: 基础巡航速度
        Slope: 线性增长系数 (使基础速度随时间线性增加)
        B_end: 结尾冲刺倍率
        T_panic: 冲刺持续时间 (小时)
        T_total: 总时长 (固定值)
        """
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
            print(f"⚠️ 拟合失败，使用默认参数: {e}")
            return np.array(p0)

# ==========================================
# 3. 数据处理器 (DataHandler)
# ==========================================
class DataHandler:
    def __init__(self, target_event_id, debug_hours=None, output_dir=None):
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

        self.seasonality = SeasonalityHandler(tz_offset=detected_offset)
        self.modeler = CosineModeler() # 👈 使用下凹正弦上升模型
        
        self.history_events = []
        self.target_data = None
        self.target_scale = 1.0
        self.debug_limit_ts = None
        
        if debug_hours:
            self.debug_limit_ts = self.meta['start_at'] + (debug_hours * 3600 * 1000)
            print(f"[调试模式] 时间冻结在: +{debug_hours}h")
            logger.info(f"Debug mode: time frozen at +{debug_hours}h (limit_ts={self.debug_limit_ts})")

    def _get_target_current_scale(self):
        url = f"{BASE_URL}eventtop/data?server={SERVER}&event={self.target_event_id}&mid=0&interval=3600000"
        try:
            data = requests.get(url, timeout=10).json()
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
            dt_utc = datetime.utcfromtimestamp(start_ts / 1000)
            utc_hour = dt_utc.hour
            print(f"🕒 活动开始时间 (UTC): {dt_utc} (Hour: {utc_hour})")

            # 如果 UTC 时间本身落在本地常见启动段，认为 API 已返回本地时间或为 UTC+0
            if 10 <= utc_hour <= 19:
                print("✅ 检测为 UTC/本地时间 (无需偏移)")
                return 0

            # 判断是否对应 UTC+8 的本地 10-19 区间
            if 10 <= ((utc_hour + 8) % 24) <= 19:
                print("✅ 检测为 UTC+8 (CN/CST)")
                return 8

            # 判断是否对应 UTC+9 的本地 10-19 区间
            if 10 <= ((utc_hour + 9) % 24) <= 19:
                print("✅ 检测为 UTC+9 (JP/JST)")
                return 9

            print("⚠️ 无法自动匹配时区，默认使用 UTC+8")
            return 8
        except Exception:
            return 8

    def load_target_data(self):
        print(f"🚀 获取目标活动 {self.target_event_id} 数据...")
        df = fetch_tier_1000_data(self.target_event_id)
        if df is None or df.empty: raise ValueError("T1000 数据为空")
        # 保留未受 debug_hours 限制的完整原始数据，用于绘图真实历史值
        raw_full_df = df.copy()
        
        if self.debug_limit_ts:
            df = df[df["time"] <= self.debug_limit_ts].copy()
            
        self.target_scale = self._get_target_current_scale()
        if not self.target_scale: self.target_scale = 20000
        print(f"⚡ 目标 T10 极速 (Scale): {self.target_scale:.0f}")

        df = calculate_speed_tracker(df)
        df["norm_speed"] = df["speed"] / self.target_scale
        
        start_ts = self.meta['start_at']
        df["hours_elapsed"] = (df["time"] - start_ts) / (1000 * 3600)
        
        self.target_data = df
        # 处理并保存完整未截断的数据供绘图使用（保留原始完整历史）
        try:
            full_df = raw_full_df.copy()
            full_df = calculate_speed_tracker(full_df)
            full_df["norm_speed"] = full_df["speed"] / self.target_scale
            full_df["hours_elapsed"] = (full_df["time"] - start_ts) / (1000 * 3600)
            self.full_target_data = full_df
        except Exception:
            # 失败时回退到已截断的数据，保证不抛异常
            self.full_target_data = df.copy()
        return df

    def find_similar_events(self, count=5):
        print(f"🔍 寻找同类 [{self.event_type}] 活动...")
        found = 0
        curr = self.target_event_id - 1
        while found < count and curr > self.target_event_id - 200:
            try:
                meta = fetch_event_meta(curr)
                if not meta or meta.get('event_type') != self.event_type:
                    # print(f"⚠️ Event {curr} 是 {meta.get('event_type', 'unknown')}，跳过")
                    continue
                
                scale = fetch_top10_max_speed(curr)
                if not scale or scale <= 0: 
                    # print(f"⚠️ Event {curr} 获取 T10 极速失败")
                    continue
                
                df_hist = fetch_tier_1000_data(curr)
                if df_hist is None or df_hist.empty: 
                    # print(f"⚠️ Event {curr} 获取 T1000 数据失败")
                    continue
                
                df_hist = calculate_speed_tracker(df_hist)
                df_hist['norm_speed'] = df_hist['speed'] / scale
                
                h_start = meta['start_at']
                h_end = meta.get('aggregate_at') or meta.get('end_at')
                df_hist['hours_elapsed'] = (df_hist['time'] - h_start) / (1000 * 3600)
                total_hours = (h_end - h_start) / 3600000
                
                self.history_events.append({
                    'event_id': curr, 'scale': scale, 'data': df_hist,
                    'total_hours': total_hours, 'start_at': h_start, 'early_intensity': 0
                })
                found += 1
                print(f"✅ 匹配: Event {curr} | Scale: {scale:.0f}")
            except: pass
            finally: curr -= 1; time.sleep(0.1)

    def run_prediction(self):
        print("\n🔮 开始预测计算 (模式: 严格时间对齐 Time-Aligned)...")
        
        # 1. 确定对比窗口 (Comparison Window)
        # 起点：6小时 (跳过开局暴冲)
        # 终点：当前时间，但上限锁死在 72小时 (3天)，正如主人所令
        t_start_cmp = 6.0
        t_end_cmp = min(self.debug_hours, 72.0)
        
        print(f"⏱️ 锁定对比区间: [ {t_start_cmp}h ~ {t_end_cmp}h ]")
        
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
            print("⚠️ 当前活动在对比区间内无有效数据，无法计算 Ratio，默认 1.0")
            curr_intensity = 0.1 # 避免除零
            
        print(f"⚡ 当前活动区间强度: {curr_intensity:.4f}")

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
                    start_dt = datetime.fromtimestamp(h_start_ts / 1000)
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
                m = f"  - Hist {h['event_id']}: 区间强度={h_int:.4f} | A={popt[1]:.5f} B={popt[2]:.6f} | params={popt}"
                logger.info(m)
            else:
                m = f"  - Hist {h['event_id']}: ⚠️ 在该时间段无数据，跳过对比"
                logger.info(m)

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

            # 选择性地混合两种 ratio：当两者都可用时取中位数（robust），否则用可用者
            ratio_candidates = []
            if np.isfinite(skeleton_ratio):
                ratio_candidates.append(float(skeleton_ratio))
            if norm_ratio is not None and np.isfinite(norm_ratio):
                ratio_candidates.append(float(norm_ratio))

            if len(ratio_candidates) == 0:
                chosen_ratio = 1.0
            else:
                chosen_ratio = float(np.median(ratio_candidates))

            # Clip ratio 以防极端放大/缩小（阈值可调整）
            R_MIN, R_MAX = 0.6, 1.6
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
            print(f"⚖️ 强度修正比率 (skeleton/norm/chosen/clipped): {skeleton_ratio:.4f} / {(norm_ratio if norm_ratio is not None else float('nan')):.4f} -> {chosen_ratio:.3f} -> {ratio:.3f}")
        else:
            print("⚠️ 没有有效的历史对比数据，Ratio 重置为 1.0")
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
            print(f"📝 预测参数: Base={pred_params[0]:.3f}, Slope={pred_params[1]:.5f}")
            logger.info(f"Final pred_params: {pred_params}")
            logger.debug(f"avg_params: {avg_params}; hist_intensities: {hist_intensities}")
            logger.debug(f"target_scale={self.target_scale}, target_total_hours={target_total_hours}, debug_hours={self.debug_hours}")
        except: pass

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
            start_dt = datetime.fromtimestamp(start_ts / 1000)
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
            hist_df = getattr(self, 'full_target_data', self.target_data)
            hist_col = 'ep' if 'ep' in hist_df.columns else ('value' if 'value' in hist_df.columns else None)
            observed_before_cutoff = 0.0
            if hist_col is not None:
                before_mask = hist_df['time'] < cutoff_ts
                if before_mask.any():
                    observed_before_cutoff = float(hist_df.loc[before_mask, hist_col].iloc[-1])

            observed_since_cutoff = float(current_max_score) - observed_before_cutoff

            if model_since_cutoff > 0 and observed_since_cutoff >= 0:
                raw_scale = observed_since_cutoff / model_since_cutoff
            else:
                raw_scale = 1.0

            SCALE_MIN, SCALE_MAX = 0.5, 2.0
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
                            CORR_MIN, CORR_MAX = 0.6, 1.6
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
                # Two-stage attenuation:
                # - mild smoothing between 0.5 and 0.65
                # - strong, nonlinear damping above 0.65 to make reaching 0.7+ unlikely
                # - enforce a hard cap below 0.8 so 0.8 is strictly unreachable
                THRESH1 = 0.5
                THRESH2 = 0.65
                HARD_CAP = 0.8
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

                # Enforce hard cap so 0.8 is unreachable
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

            # reuse same smoothing parameters as applied earlier
            THRESH1 = 0.5
            THRESH2 = 0.65
            HARD_CAP = 0.8
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
            output_path = os.path.join(self.output_dir, fname)
        except Exception:
            output_path = None

        self.plot_final(target_df, future_t, skeleton_pred, norm_adj_full, full_t_score, full_score, output_path=output_path)
        logger.info(f"Prediction complete for event={self.target_event_id}; final_score={int(full_score[-1]) if len(full_score)>0 else 0}")
        logger.info("---- END RUN ----\n")

    def plot_final(self, target_df, t_pred, y_skeleton, y_final, t_score, y_score, output_path=None):
        """
        Draw prediction plots and save to `output_path` if provided.

        Parameters:
        - target_df: DataFrame of observed points (with 'hours_elapsed', 'norm_speed', 'skeleton_speed')
        - t_pred, y_skeleton: arrays for predicted skeleton curve
        - y_final: adjusted predicted normalized speed curve (post-scale and smoothing)
        - t_score, y_score: arrays for combined historical+predicted cumulative score timeline
        - output_path: if provided, save figure to this path; otherwise call plt.show().
        """
        # ensure output directory exists when saving
        if output_path:
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to create output directory {out_dir}: {e}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # --- 子图1: 速度曲线 ---
        ax1.scatter(target_df['hours_elapsed'], target_df['skeleton_speed'], 
                    s=10, color='gray', alpha=0.3, label='Observed Skeleton')
        ax1.plot(t_pred, y_skeleton, color='blue', linestyle='--', alpha=0.5, label='Predicted Skeleton (Sine - Concave)')
        
        ax1.plot(target_df['hours_elapsed'], target_df['norm_speed'], 
                 color='red', linewidth=2, label='Observed Speed')
        ax1.plot(t_pred, y_final, color='green', linewidth=2, alpha=0.8, label='Predicted Speed')
        # 如果存在完整的未截断数据，绘制真实的“未来”速度以便对比
        try:
            if hasattr(self, 'full_target_data') and hasattr(self, 'target_data'):
                obs_end = float(self.target_data['hours_elapsed'].max()) if len(self.target_data) > 0 else 0
                full_df = self.full_target_data
                mask_future_real = full_df['hours_elapsed'].values > obs_end
                if np.any(mask_future_real):
                    ax1.plot(full_df['hours_elapsed'].values[mask_future_real],
                             full_df['norm_speed'].values[mask_future_real],
                             color='orange', linestyle='-.', linewidth=2, alpha=0.9, label='Actual Future Speed')
        except Exception:
            pass

        ax1.axvline(x=self.debug_hours, color='black', linestyle=':', label='Now')
        ax1.set_ylabel("Normalized Speed")
        ax1.set_title(f"Event {self.target_event_id} Speed Prediction")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # --- 子图2: 累计分数曲线 ---
        # 绘制历史累计分数：使用完整未截断的 `self.full_target_data`（若存在），否则退回到 `self.target_data`
        hist_df = getattr(self, 'full_target_data', self.target_data)
        if 'ep' in hist_df.columns:
            hist_score = hist_df['ep'].values
        elif 'value' in hist_df.columns:
            hist_score = hist_df['value'].values
        else:
            hist_score = np.zeros(len(hist_df))

        hist_hours = hist_df['hours_elapsed'].values
        ax2.plot(hist_hours, hist_score, color='red', linewidth=2, label='Observed Score')

        # 标注真实最终观测值（最后一个历史点）
        try:
            if len(hist_hours) > 0:
                obs_final_time = float(hist_hours[-1])
                obs_final_val = float(hist_score[-1])
                ax2.scatter(obs_final_time, obs_final_val, color='darkred', s=50, zorder=5, label='Observed Final')
                ax2.text(obs_final_time, obs_final_val, f"Observed: {int(obs_final_val):,}",
                         ha='left', va='bottom', fontsize=10, color='darkred')
        except Exception:
            pass

        # 只画预测的未来部分，避免与历史重叠
        # 使用被截断的观测数据（self.target_data）的最后时刻作为“现在”的分界点，
        # 否则 full_target_data 可能包含更晚的历史值导致没有未来段被绘制。
        if hasattr(self, 'target_data') and len(self.target_data) > 0:
            obs_end = float(self.target_data['hours_elapsed'].max())
        else:
            obs_end = hist_hours.max() if len(hist_hours) > 0 else 0
        pred_mask = np.array(t_score) > obs_end
        if np.any(pred_mask):
            t_pred_only = np.array(t_score)[pred_mask]
            y_pred_only = np.array(y_score)[pred_mask]
            ax2.plot(t_pred_only, y_pred_only, color='purple', linestyle='--', linewidth=2, label='Predicted Future')
            final_score = y_pred_only[-1]
            ax2.text(t_pred_only[-1], final_score, f"{int(final_score):,}", 
                     ha='right', va='bottom', fontsize=12, fontweight='bold', color='purple')
        else:
            # 如果没有未来点，则使用最后一个点作为最终值展示
            final_score = float(hist_score[-1]) if len(hist_score) > 0 else 0.0

        ax2.axvline(x=self.debug_hours, color='black', linestyle=':')
        ax2.set_ylabel("Cumulative Event Points")
        ax2.set_xlabel("Hours Elapsed")
        ax2.set_title(f"Final Prediction: {int(final_score):,} PT")
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if output_path:
            try:
                fig.savefig(output_path, dpi=150)
                print(f"📁 预测图已保存: {output_path}")
                logger.info(f"Saved prediction plot to {output_path}")
            except Exception as e:
                logger.warning(f"Failed to save plot to {output_path}: {e}")
                plt.show()
        else:
            plt.show()

        print(f"最终预测分数: {int(final_score):,} PT")
        # close the figure to free memory
        plt.close(fig)

# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    try:
        # 假设预测 312，时间冻结在 60h
        handler = DataHandler(312, debug_hours=60)
        handler.load_target_data()
        handler.find_similar_events()
        
        if handler.history_events:
            handler.run_prediction()
        else:
            print("😿 没找到历史活动，无法预测。")
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()