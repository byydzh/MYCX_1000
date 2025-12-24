# prediction_engine.py
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging
from datetime import datetime, timedelta, timezone

from domain_models import EventData, PredictionResult
from config import DEFAULT_CONFIG

logger = logging.getLogger('predictor.engine')

class PredictionEngine:
    """
    核心预测引擎。
    职责：
    1. 接收清洗好的 EventData。
    2. 执行去节律化 (De-seasonality)。
    3. 计算对比系数 (Ratio)。
    4. 执行拟合与预测 (Fitting & Prediction)。
    5. 应用修正逻辑 (Backtest Correction & Smoothing)。
    """
    def __init__(self, seasonality_handler, modeler, config: dict = None):
        self.seasonality = seasonality_handler
        self.modeler = modeler
        self.config = config or DEFAULT_CONFIG

    def _get_window_intensity(self, df: pd.DataFrame, t_start: float, t_end: float) -> Optional[float]:
        """计算指定时间窗口内的平均“骨架速度”(Skeleton Speed)。"""
        mask = (df['hours_elapsed'] >= t_start) & \
               (df['hours_elapsed'] <= t_end) & \
               (np.isfinite(df['skeleton_speed']))
        
        data_slice = df.loc[mask, 'skeleton_speed']
        if len(data_slice) == 0: return None
        
        # Sigma Clipping 去除异常值
        mean_val = data_slice.mean()
        std_val = data_slice.std()
        if std_val > 0.001:
            clean_slice = data_slice[np.abs(data_slice - mean_val) < 2.0 * std_val]
            if len(clean_slice) > 0:
                return clean_slice.mean()
        return mean_val

    def _calculate_ratio(self, target: EventData, history: List[EventData], 
                         t_start: float, t_end: float) -> float:
        """计算当前活动相对于历史活动的强度比率 (Ratio)。"""
        # 1. 计算当前活动强度
        curr_intensity = self._get_window_intensity(target.df, t_start, t_end)
        if curr_intensity is None:
            logger.warning("当前活动在对比区间内无有效数据，Ratio 默认为 1.0")
            return 1.0

        # 2. 计算历史平均强度
        hist_intensities = []
        hist_norms = []
        
        # 辅助校验：观测 Norm Speed 均值
        mask_cmp = (target.df['hours_elapsed'] >= t_start) & (target.df['hours_elapsed'] <= t_end)
        obs_norm_mean = target.df.loc[mask_cmp, 'norm_speed'].mean() if mask_cmp.any() else 0

        for h in history:
            if 'skeleton_speed' not in h.df.columns:
                h.df = self.seasonality.remove_seasonality(h.df)
            
            h_int = self._get_window_intensity(h.df, t_start, t_end)
            if h_int is not None:
                hist_intensities.append(h_int)
                
            mask_h = (h.df['hours_elapsed'] >= t_start) & (h.df['hours_elapsed'] <= t_end)
            if mask_h.any():
                hist_norms.append(h.df.loc[mask_h, 'norm_speed'].mean())

        if not hist_intensities:
            logger.warning("没有有效的历史对比数据，Ratio 默认为 1.0")
            return 1.0

        # 3. 计算两种 Ratio
        avg_hist_intensity = np.mean(hist_intensities)
        skeleton_ratio = curr_intensity / avg_hist_intensity if avg_hist_intensity > 0 else 1.0
        
        norm_ratio = 1.0
        if hist_norms:
            mean_hist_norm = np.mean(hist_norms)
            if mean_hist_norm > 0:
                norm_ratio = obs_norm_mean / mean_hist_norm

        # 4. 混合逻辑
        target_total_hours = float(self.config.get('t_end_cap', 72.0))
        observed_hours = float(target.df['hours_elapsed'].max())
        s = np.clip(observed_hours / target_total_hours, 0.0, 1.0)
        
        w_norm = 0.2 + 0.6 * np.cos(s * np.pi - np.pi)
        w_norm = float(np.clip(w_norm, 0.0, 1.0))
        
        chosen_ratio = skeleton_ratio * (1.0 - w_norm) + norm_ratio * w_norm
        
        # 5. 截断保护
        R_MIN = float(self.config.get('ratio_min', 0.25))
        R_MAX = float(self.config.get('ratio_max', 4.0))
        return float(np.clip(chosen_ratio, R_MIN, R_MAX))

    def _fit_history_params(self, history: List[EventData]) -> np.ndarray:
        """拟合历史参数"""
        hist_params = []
        for h in history:
            if 'skeleton_speed' not in h.df.columns:
                h.df = self.seasonality.remove_seasonality(h.df)
            
            valid_mask = np.isfinite(h.df['skeleton_speed'])
            if valid_mask.sum() >= 5:
                try:
                    popt = self.modeler.fit(
                        h.df.loc[valid_mask, 'hours_elapsed'].values,
                        h.df.loc[valid_mask, 'skeleton_speed'].values,
                        h.meta.total_hours
                    )
                    hist_params.append(popt)
                except Exception:
                    pass
        
        if not hist_params:
            return np.array([0.05, 0.001, 0.0, 0.5, 24.0])
        return np.mean(hist_params, axis=0)

    def _calculate_scale_factor(self, target: EventData, future_t: np.ndarray, 
                                speed_pred_norm: np.ndarray) -> float:
        """
        [核心逻辑] 计算缩放系数 (Scale Factor)。
        通过对比“模型预测的积分”和“实际观测的积分增量”来校准。
        包含：
        1. Cutoff Alignment: 对齐首日18:00到现在的积分。
        2. 24h Backtest: 如果进度过半，额外回测过去24小时的吻合度。
        """
        try:
            # 准备基础变量
            real_speed_all = speed_pred_norm * target.scale
            dt_hours = (future_t[1] - future_t[0]) if len(future_t) > 1 else 0.0
            dt_min = dt_hours * 60.0
            cum_model = np.cumsum(real_speed_all * dt_min) # 模型的累积积分曲线

            current_max_time = target.df['hours_elapsed'].max()
            current_max_score = target.df['value'].max()

            # --- 步骤 1: 确定 Cutoff 时间 (首日 18:00) ---
            # 目的：跳过开服前几小时的不稳定期，从稳定的 18:00 开始计算积分
            start_ts = target.meta.start_at
            tz_offset = self.seasonality.tz_offset
            
            # 构造带时区的 datetime
            start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone(timedelta(hours=tz_offset)))
            cutoff_dt = start_dt.replace(hour=18, minute=0, second=0, microsecond=0)
            cutoff_ts = int(cutoff_dt.timestamp() * 1000)
            
            # 转为相对小时数
            cutoff_hours = (cutoff_ts - start_ts) / 3600000.0
            if cutoff_hours < 0.0: cutoff_hours = 0.0

            # --- 步骤 2: 计算 Model Since Cutoff ---
            # 在 future_t 中找到 cutoff 和 now 的索引
            idx_cutoff = int(np.searchsorted(future_t, cutoff_hours, side='left'))
            idx_now = int(np.searchsorted(future_t, current_max_time, side='right') - 1)
            
            # 索引边界保护
            idx_cutoff = max(0, min(idx_cutoff, len(cum_model) - 1))
            idx_now = max(0, min(idx_now, len(cum_model) - 1))

            # 模型在 [Cutoff, Now] 期间产生的积分增量
            model_val_cutoff = cum_model[idx_cutoff-1] if idx_cutoff > 0 else 0.0
            model_since_cutoff = float(cum_model[idx_now] - model_val_cutoff)

            # --- 步骤 3: 计算 Observed Since Cutoff ---
            # 找到实际数据中 Cutoff 时刻的分数
            observed_before_cutoff = 0.0
            
            # 使用 hours_elapsed 查找比 cutoff_hours 小的最后一个点
            hrs = target.df['hours_elapsed'].values
            scores = target.df['value'].values
            before_mask = hrs < cutoff_hours
            
            if np.any(before_mask):
                observed_before_cutoff = float(scores[np.where(before_mask)[0][-1]])
            else:
                observed_before_cutoff = float(scores[0]) if len(scores) > 0 else 0.0

            observed_since_cutoff = float(current_max_score) - observed_before_cutoff

            # --- 步骤 4: 初步 Scale 计算 ---
            if model_since_cutoff > 0 and observed_since_cutoff >= 0:
                raw_scale = observed_since_cutoff / model_since_cutoff
            else:
                raw_scale = 1.0

            # 截断
            SCALE_MIN = float(self.config.get('scale_min', 0.5))
            SCALE_MAX = float(self.config.get('scale_max', 2.0))
            scale_factor = float(np.clip(raw_scale, SCALE_MIN, SCALE_MAX))
            
            logger.info(f"Scale Diagnostics: ObsDelta={observed_since_cutoff:.0f}, ModDelta={model_since_cutoff:.0f}, Raw={raw_scale:.4f}, Clipped={scale_factor:.4f}")

            # --- 步骤 5: 24h 回测修正 (Backtest Correction) ---
            # 如果活动进行超过 50 小时，检查过去 24 小时的拟合情况
            applied_scale_factor = scale_factor
            
            if current_max_time > 50.0:
                t0 = max(0.0, current_max_time - 24.0)
                
                # 定位索引
                idx_t0 = int(np.searchsorted(future_t, t0, side='left'))
                idx_now = int(np.searchsorted(future_t, current_max_time, side='right') - 1)
                idx_t0 = max(0, min(idx_t0, len(future_t) - 1))
                idx_now = max(0, min(idx_now, len(future_t) - 1))

                if idx_now > idx_t0:
                    # 模型在过去 24h 的增量 (应用了当前的 scale_factor)
                    # 注意：这里我们手动积分这一段，因为 cum_model 是未缩放的
                    pred_segment = speed_pred_norm[idx_t0:idx_now]
                    model_24 = float(np.sum(pred_segment) * target.scale * scale_factor * dt_min)

                    # 实际在过去 24h 的增量
                    pos = int(np.searchsorted(hrs, t0, side='left'))
                    observed_before_t0 = float(scores[pos-1]) if pos > 0 else float(scores[0])
                    observed_24 = float(current_max_score) - observed_before_t0

                    # 计算修正比率
                    if model_24 > 0 and observed_24 >= 0:
                        raw_corr = observed_24 / model_24
                        CORR_MIN = float(self.config.get('corr_min', 0.6))
                        CORR_MAX = float(self.config.get('corr_max', 1.6))
                        corr = float(np.clip(raw_corr, CORR_MIN, CORR_MAX))
                        
                        applied_scale_factor = scale_factor * corr
                        logger.info(f"24h Backtest: Obs24={observed_24:.0f}, Mod24={model_24:.0f}, Corr={corr:.4f} -> FinalScale={applied_scale_factor:.4f}")

            return applied_scale_factor

        except Exception as e:
            logger.error(f"Error calculating scale factor: {e}")
            return 1.0

    def _apply_smoothing(self, speed_pred_norm: np.ndarray, scale_factor: float) -> np.ndarray:
        """应用高分段的平滑压制逻辑 (Top-speed Smoothing)。"""
        norm_after_scale = speed_pred_norm * scale_factor
        
        THRESH1 = float(self.config.get('smooth_thresh1', 0.5))
        THRESH2 = float(self.config.get('smooth_thresh2', 0.65))
        HARD_CAP = float(self.config.get('smooth_hard_cap', 0.8))
        ALPHA = 3.0
        BETA = 22.0

        norm_adj = norm_after_scale.copy()

        # Stage 1: Mild
        mask_stage1 = (norm_after_scale > THRESH1) & (norm_after_scale <= THRESH2)
        if np.any(mask_stage1):
            excess1 = (norm_after_scale[mask_stage1] - THRESH1) / (THRESH2 - THRESH1)
            attenuation1 = 1.0 / (1.0 + ALPHA * excess1)
            norm_adj[mask_stage1] = THRESH1 + (norm_after_scale[mask_stage1] - THRESH1) * attenuation1

        # Stage 2: Strong
        mask_stage2 = norm_after_scale > THRESH2
        if np.any(mask_stage2):
            excess2 = (norm_after_scale[mask_stage2] - THRESH2) / (1.0 - THRESH2)
            attenuation2 = 1.0 / (1.0 + BETA * (excess2 ** 2))
            norm_adj[mask_stage2] = THRESH2 + (norm_after_scale[mask_stage2] - THRESH2) * attenuation2

        return np.minimum(norm_adj, HARD_CAP)

    def predict(self, target: EventData, history: List[EventData], 
                debug_hours: Optional[float] = None) -> PredictionResult:
        """主预测入口。"""
        # 0. 准备数据：去节律化
        target.df = self.seasonality.remove_seasonality(target.df)
        
        # 1. 确定对比窗口
        t_start_cmp = float(self.config.get('t_start_cmp', 6.0))
        observed_hours = float(target.df['hours_elapsed'].max())
        end_source = debug_hours if debug_hours is not None else observed_hours
        t_end_cmp = min(end_source, float(self.config.get('t_end_cap', 72.0)))
        
        # 2. 计算 Ratio
        ratio = self._calculate_ratio(target, history, t_start_cmp, t_end_cmp)
        logger.info(f"Calculated Ratio: {ratio:.4f}")

        # 3. 获取并修正参数
        avg_params = self._fit_history_params(history)
        pred_params = avg_params.copy()
        pred_params[0] *= ratio        # Base
        pred_params[1] *= ratio        # A
        pred_params[2] *= ratio        # B
        pred_params[3] *= (ratio ** 1.1) # B_end
        
        # 4. 生成基础预测曲线 (Skeleton)
        total_hours = target.meta.total_hours
        future_t = np.linspace(0, total_hours, 1000)
        skeleton_pred = self.modeler.shape_function(future_t, *pred_params, total_hours)
        
        # 5. 应用节律 (Re-seasonality)
        speed_pred_norm, _ = self.seasonality.apply_seasonality(
            future_t, skeleton_pred, target.meta.start_at,
            total_hours=total_hours, t_panic=pred_params[4]
        )
        
        # 6. 计算 Scale Factor (积分对齐 + 24h回测)
        scale_factor = self._calculate_scale_factor(target, future_t, speed_pred_norm)
        
        # 7. 应用平滑压制
        final_speed_norm = self._apply_smoothing(speed_pred_norm, scale_factor)
        
        # 8. 积分求分数
        real_speed_ep_min = final_speed_norm * target.scale
        dt_hours = (future_t[1] - future_t[0]) if len(future_t) > 1 else 0
        dt_min = dt_hours * 60
        
        current_max_score = target.df['value'].max() if not target.df.empty else 0
        current_max_time = target.df['hours_elapsed'].max() if not target.df.empty else 0
        
        future_mask = future_t >= current_max_time
        future_t_clip = future_t[future_mask]
        speed_clip = real_speed_ep_min[future_mask]
        
        score_increment = np.cumsum(speed_clip * dt_min)
        score_pred = current_max_score + score_increment
        
        final_score = score_pred[-1] if len(score_pred) > 0 else current_max_score

        full_t_score = np.concatenate([target.df['hours_elapsed'].values, future_t_clip])
        full_score = np.concatenate([target.df['value'].values, score_pred])

        return PredictionResult(
            event_id=target.meta.event_id,
            future_t=future_t,
            future_score=score_pred,
            future_speed=final_speed_norm,
            final_score=final_score,
            used_params=pred_params,
            ratio=ratio,
            scale_factor=scale_factor,
            full_t_score=full_t_score,
            full_score=full_score
        )