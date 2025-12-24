# math_models.py
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from scipy.optimize import curve_fit
from base_distribution import get_day_type  # 假设 base_distribution 还在

try:
    from chinese_calendar import is_workday
except ImportError:
    is_workday = None

class SeasonalityHandler:
    """处理昼夜节律和周末效应"""
    def __init__(self, json_path='base_speed_distribution.json', tz_offset=8, 
                 panic_ease_power=1.0, weekend_multiplier=1.0, panic_scaler=1.1):
        self.data = self._load_json(json_path)
        if weekend_multiplier != 1.0:
            self._apply_multiplier('weekend', weekend_multiplier)
        self.wd_mean, self.we_mean, self.global_mean = self._calculate_weighted_means()
        self.tz_offset = tz_offset
        self.panic_ease_power = float(panic_ease_power)
        self.panic_scaler = float(panic_scaler)

    def _load_json(self, path):
        if not os.path.exists(path): return {}
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)

    def _apply_multiplier(self, dtype, multiplier):
        if dtype not in self.data: return
        for hour_key in self.data[dtype]:
            if 'mean' in self.data[dtype][hour_key]:
                self.data[dtype][hour_key]['mean'] *= multiplier

    def _calculate_weighted_means(self):
        if not self.data: return 1.0, 1.0, 1.0
        def get_mean(dtype):
            vals = [self.data.get(dtype, {}).get(str(h), {}).get('mean', 0) for h in range(24)]
            vals = [v for v in vals if v > 0]
            return np.mean(vals) if vals else 1.0
        wd, we = get_mean('weekday'), get_mean('weekend')
        return wd, we, (wd * 5.0 + we * 2.0) / 7.0

    def get_factor(self, dt):
        if not self.data: return 1.0
        # 逻辑简化：确保 dt 是 datetime 对象
        if isinstance(dt, (int, float)):
            dt = datetime.fromtimestamp(dt / 1000, timezone.utc) + timedelta(hours=self.tz_offset)
        
        # 判断类型 (Weekday/Weekend)
        dtype = 'weekday'
        if is_workday:
            try:
                dtype = 'weekday' if is_workday(dt) else 'weekend'
            except: dtype = get_day_type(dt)
        else:
            dtype = get_day_type(dt)
            
        # 特殊规则
        if dt.weekday() == 4 and dt.hour >= 17: dtype = 'weekend'
        elif dt.weekday() == 6 and dt.hour >= 23: dtype = 'weekday'
            
        stats = self.data.get(dtype, {}).get(str(dt.hour))
        return (stats['mean'] / self.global_mean) if stats and stats['mean'] > 0 else 1.0

    def remove_seasonality(self, df):
        df = df.copy()
        df['dt_local'] = pd.to_datetime(df['time'], unit='ms') + pd.Timedelta(hours=self.tz_offset)
        df['season_factor'] = df['dt_local'].apply(self.get_factor)
        # Early suppression
        if 'hours_elapsed' in df.columns:
            mask = (df['hours_elapsed'] < 12.0) & (df['season_factor'] < 1.0)
            if mask.any(): df.loc[mask, 'season_factor'] = 1.0
        df['skeleton_speed'] = df['norm_speed'] / df['season_factor']
        return df

    def apply_seasonality(self, t_hours, y_skeleton, start_ts, total_hours=None, t_panic=24.0):
        y_final, factors = [], []
        for i, h in enumerate(t_hours):
            ts = start_ts + (h * 3600 * 1000)
            dt = datetime.fromtimestamp(ts / 1000, timezone.utc) + timedelta(hours=self.tz_offset)
            raw = self.get_factor(dt)
            
            final = raw
            if total_hours and t_panic > 0:
                left = total_hours - h
                if left < t_panic:
                    prog = 1.0 - (max(0, left) / t_panic)
                    eased = float(np.power(prog, self.panic_ease_power)) if prog > 0 else 0
                    final = raw * (1.0 - eased) + max(raw, self.panic_scaler) * eased
            factors.append(final)
            y_final.append(y_skeleton[i] * final)
        return np.array(y_final), np.array(factors)

class CosineModeler:
    """正弦下凹增长模型"""
    def shape_function(self, t, Base, A, B, B_end, T_panic, T_total):
        y = Base + (A * t) + (B * (t ** 2))
        t_panic_start = T_total - T_panic
        
        # 结尾下凹上升逻辑
        rise = np.zeros_like(t, dtype=float)
        mask_end = t > t_panic_start
        if np.any(mask_end):
            norm_t = np.clip((t[mask_end] - t_panic_start) / T_panic, 0, 1)
            base_shape = np.sin(norm_t * (np.pi / 2.0))
            focus = np.power(np.clip((norm_t - 0.5) / 0.5, 0, 1), 3.0)
            rise[mask_end] = B_end * (np.power(base_shape, 2.5) * focus)
            
        blend_start = t_panic_start - 4.0
        blend = np.clip((t - blend_start) / max(1e-6, (t_panic_start + T_panic/2) - blend_start), 0, 1)
        return np.maximum(y + (rise * blend), 0)

    def fit(self, t, y, total_hours):
        p0 = [0.05, 0.001, 0.00001, 0.5, 24.0]
        bounds = ([0, -0.01, -0.001, 0, 6], [1, 0.01, 0.001, 10, min(72, total_hours/2)])
        try:
            popt, _ = curve_fit(lambda t_val, *p: self.shape_function(t_val, *p, total_hours), 
                                t, y, p0=p0, bounds=bounds, maxfev=40000)
            return popt
        except: return np.array(p0)