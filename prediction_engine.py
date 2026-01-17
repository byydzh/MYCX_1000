# prediction_engine.py
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging
from scipy.optimize import minimize
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
    5. 应用修正逻辑 (Kalman Filter Correction & Smoothing)。
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

    def _refit_shape_params(self, target: EventData, initial_params: np.ndarray) -> np.ndarray:
        """
        使用带正则化的最小二乘法，基于当前观测对形状参数 Base, A, B 进行在线重拟合。
        如果观测点不足则返回 initial_params 不变。
        """
        try:
            if 'skeleton_speed' not in target.df.columns:
                return initial_params

            valid_mask = np.isfinite(target.df['skeleton_speed'])
            if valid_mask.sum() < int(self.config.get('refit_min_points', 10)):
                return initial_params

            t_obs = target.df.loc[valid_mask, 'hours_elapsed'].values
            v_obs = target.df.loc[valid_mask, 'skeleton_speed'].values

            prior = initial_params.copy()
            total_hours = target.meta.total_hours

            # 自适应正则强度，依据观测量级和观测能量
            lambda_reg = float(self.config.get('refit_lambda', 0.3)) * (np.mean(v_obs ** 2) + 1e-6)

            def loss(x):
                # x[0] -> Base, x[1] -> A, x[2] -> B
                tmp = prior.copy()
                tmp[0] = float(x[0])
                tmp[1] = float(x[1])
                tmp[2] = float(x[2])

                v_pred = self.modeler.shape_function(t_obs, *tmp, total_hours)
                mse = np.mean((v_obs - v_pred) ** 2)

                # 相对正则化，防止量级问题
                reg_Base = ((x[0] - prior[0]) ** 2) / (prior[0] ** 2 + 1e-9)
                reg_A = ((x[1] - prior[1]) ** 2) / (prior[1] ** 2 + 1e-9)
                reg_B = ((x[2] - prior[2]) ** 2) / (prior[2] ** 2 + 1e-9)
                return mse + lambda_reg * (reg_Base + reg_A + reg_B)

            # Base, A, B 均需非负
            bounds = [(0.0, None), (0.0, None), (0.0, None)]
            x0 = [prior[0], prior[1], prior[2]]
            res = minimize(loss, x0=x0, bounds=bounds, method='L-BFGS-B')

            if res.success:
                new_params = prior.copy()
                new_params[0] = float(res.x[0])
                new_params[1] = float(res.x[1])
                new_params[2] = float(res.x[2])
                logger.info(f"Refit params: Base {prior[0]:.4f}->{new_params[0]:.4f}, A {prior[1]:.5f}->{new_params[1]:.5f}, B {prior[2]:.6f}->{new_params[2]:.6f}")
                return new_params
            else:
                return initial_params

        except Exception as e:
            logger.warning(f"_refit_shape_params failed: {e}")
            return initial_params

    def _run_kalman_filter(self, times: np.ndarray, obs_scores: np.ndarray, 
                           model_speed_func, dt_step=1.0) -> Tuple[float, float]:
        # print(f"\n{'='*20} START KALMAN FILTER DEBUG {'='*20}")
        # print(f"{'Time':<6} | {'ObsDelta':<8} | {'ModDelta':<8} | {'Raw_Z':<8} | {'Z_clip':<6} | {'K_gain':<6} | {'Scale':<6} | {'Trend':<8}")
        # print("-" * 90)

        # --- 1. 初始化状态 ---
        # State x = [scale, trend]^T
        x = np.array([[1.0], [0.0]])
        
        # 状态协方差 P (初始不确定性)
        P = np.array([[0.5, 0], [0, 0.01]])
        
        # 状态转移矩阵 F
        F = np.array([[1.0, dt_step], [0.0, 1.0]])
        
        # 过程噪声 Q
        Q = np.array([[1e-5, 0], [0, 1e-7]]) 
        
        # 测量矩阵 H
        H = np.array([[1.0, 0.0]])
        
        # --- 2. 准备观测数据 ---
        t_start = times[0]
        t_end = times[-1]
        current_t = t_start
        
        while current_t + dt_step <= t_end:
            t1 = current_t
            t2 = current_t + dt_step
            
            # --- A. 预测步骤 (Predict) ---
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            
            # --- B. 构建观测 (Measurement) ---
            # 1. 计算实际增量
            score_t1 = np.interp(t1, times, obs_scores)
            score_t2 = np.interp(t2, times, obs_scores)
            delta_obs = score_t2 - score_t1
            
            # 2. 计算模型增量
            mid_t = (t1 + t2) / 2.0
            model_speed = model_speed_func(mid_t)
            delta_model = model_speed * dt_step
            
            # --- C. 更新步骤 (Update) ---
            # 只有当模型预测有显著增量时，观测才有效
            if delta_model > 50.0: 
                raw_z = delta_obs / delta_model
                
                # 观测值硬截断
                z_val = np.clip(raw_z, 0.1, 5.0)
                z = np.array([[z_val]])
                
                # 动态调整测量噪声 R
                penalty = 0.0
                if abs(raw_z - z_val) > 0.1:
                    penalty = 100.0 
                
                base_R = 0.1
                adaptive_R = base_R + (2000.0 / (delta_model + 1.0)) * 0.01 + penalty
                R = np.array([[adaptive_R]])
                
                # 卡尔曼增益 K
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                
                # 更新状态
                y = z - (H @ x_pred) # 残差
                x = x_pred + K @ y
                P = (np.eye(2) - K @ H) @ P_pred

                # === DEBUG PRINT ===
                # 重点观察 Raw_Z 是否因为 delta_model 太小而变得巨大
                # print(f"{current_t:<6.1f} | {delta_obs:<8.1f} | {delta_model:<8.1f} | {raw_z:<8.3f} | {z_val:<6.2f} | {K[0,0]:<6.3f} | {x[0,0]:<6.3f} | {x[1,0]:<8.5f}")

            else:
                # 如果跳过了更新，也打印一下，看看是不是因为 delta_model 太小
                x = x_pred
                P = P_pred
                # print(f"{current_t:<6.1f} | {delta_obs:<8.1f} | {delta_model:<8.1f} | {'SKIP':<8} | {'-'*6} | {'-'*6} | {x[0,0]:<6.3f} | {x[1,0]:<8.5f} <--- Skipped (Model too small)")
            
            current_t += dt_step
            
        # print(f"{'='*20} END DEBUG {'='*20}\n")
        return float(x[0, 0]), float(x[1, 0])

    def _calculate_scale_factor(self, target: EventData, future_t: np.ndarray, 
                                speed_pred_norm: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        使用卡尔曼滤波计算 Scale Factor。
        返回: (Applied_Scale_Factor, Dynamic_Adjustment_Array)
        """
        try:
            # 1. 确定 Cutoff 时间 (首日 18:00)
            # 目的：跳过开服前几小时的不稳定期
            start_ts = target.meta.start_at
            tz_offset = self.seasonality.tz_offset
            start_dt = datetime.fromtimestamp(start_ts / 1000, tz=timezone(timedelta(hours=tz_offset)))
            cutoff_dt = start_dt.replace(hour=18, minute=0, second=0, microsecond=0)
            cutoff_ts = int(cutoff_dt.timestamp() * 1000)
            cutoff_hours = max(0.0, (cutoff_ts - start_ts) / 3600000.0)
            
            current_max_time = target.df['hours_elapsed'].max()
            
            # 如果数据太少（还没到 cutoff），直接返回默认值
            if current_max_time < cutoff_hours + 1.0:
                return 1.0, np.ones_like(future_t)

            # 2. 准备数据给 KF
            # 截取 Cutoff 之后的数据
            mask = target.df['hours_elapsed'] >= cutoff_hours
            if mask.sum() < 2: return 1.0, np.ones_like(future_t)
            
            kf_times = target.df.loc[mask, 'hours_elapsed'].values
            kf_scores = target.df.loc[mask, 'value'].values
            
            # 构造一个快速查询模型速度的函数
            # speed_pred_norm 对应 future_t
            def get_model_speed(t):
                # 这里的 speed 是 normalized speed，需要乘 target.scale 才是 pt/hour
                norm_spd = np.interp(t, future_t, speed_pred_norm)
                return norm_spd * target.scale * 60.0 

            # 3. 运行 KF
            est_scale, est_trend = self._run_kalman_filter(kf_times, kf_scores, get_model_speed, dt_step=1.0)
            
            logger.info(f"Kalman Filter Result: Scale={est_scale:.4f}, Trend={est_trend:.6f}/hr")
            
            # 4. 构造未来的 Scale 曲线
            # 我们不希望 Trend 无限延伸，因此应用衰减
            # Scale(t) = Scale_now + Trend_now * (1 - e^(-lambda * dt)) / lambda  (类似阻尼)
            # 或者简单点：让 Trend 线性衰减到 0
            
            # 截断保护
            est_scale = np.clip(est_scale, 0.5, 3.0)
            # 限制 trend 的幅度，防止过拟合
            est_trend = np.clip(est_trend, -0.03, 0.05)
            
            # 构造 adjustment array
            scale_curve = np.ones_like(future_t) * est_scale
            
            # 找到现在的索引
            idx_now = np.searchsorted(future_t, current_max_time)
            
            # 对未来应用带阻尼的趋势
            if idx_now < len(future_t):
                future_deltas = future_t[idx_now:] - current_max_time
                # 阻尼系数：每过 2 小时，趋势影响力减半 (防止长期趋势过度外推导致 Scale 负值)
                decay_lambda = np.log(2) / 2.0
                
                # 积分形式的阻尼： Trend * (1 - exp(-lambda * t)) / lambda ???
                # 不，Scale 是速度的系数。Trend 是 Scale 的变化率。
                # Scale(t) = Scale_0 + \int Trend(t) dt
                # 设 Trend(t) = Trend_0 * exp(-lambda * t)
                # 则 Scale(t) = Scale_0 + Trend_0 * (1 - exp(-lambda * t)) / lambda
                
                trend_impact = est_trend * (1.0 - np.exp(-decay_lambda * future_deltas)) / decay_lambda
                scale_curve[idx_now:] += trend_impact
            
            # 强制非负保护 (Scale 必须 > 0.1)
            scale_curve = np.maximum(scale_curve, 0.1)
                
            # 对过去的部分（仅用于绘图或对齐），简单设为 est_scale
            # 实际计算积分时，过去的部分其实不重要，因为我们是基于 current_score 往后加
            return float(est_scale), scale_curve

        except Exception as e:
            logger.error(f"Error in Kalman Filter: {e}")
            return 1.0, np.ones_like(future_t)

    def _generate_hypothetical_path(self, target: EventData, manual_points: List[dict],
                                    pred_params: np.ndarray) -> Tuple[pd.DataFrame, float]:
        """
        生成连接当前数据末尾到人工干预点的“符合节律的”虚拟路径。
        
        Args:
            target: 当前真实数据
            manual_points: 人工干预点列表 [{'hours': h, 'score': s}, ...]
            pred_params: 基础预测参数 (用于生成形状)
            
        Returns:
            synthetic_df: 包含真实数据+虚拟路径的完整 DataFrame
            last_manual_time: 最后一个人工点的时间 (新的 "Now")
        """
        if not manual_points:
            return target.df, target.df['hours_elapsed'].max()

        # 1. 排序并过滤无效点
        sorted_points = sorted(manual_points, key=lambda x: x['hours'])
        current_max_time = target.df['hours_elapsed'].max()
        current_max_score = target.df['value'].max()
        
        valid_points = [p for p in sorted_points if p['hours'] > current_max_time and p['score'] > current_max_score]
        if not valid_points:
            return target.df, current_max_time

        # 2. 准备基础数据
        full_df = target.df.copy()
        last_t = current_max_time
        last_s = current_max_score
        
        total_hours = target.meta.total_hours
        
        # 3. 逐段生成路径
        for pt in valid_points:
            next_t = float(pt['hours'])
            next_s = float(pt['score'])
            
            if next_t <= last_t: continue
            
            # 生成该区间的密集时间点 (每 6 分钟一个点)
            segment_t = np.arange(last_t, next_t, 0.1)
            if len(segment_t) == 0: continue
            
            # A. 计算该区间的“理论无缩放增量” (Raw Increment)
            # 使用传入的 pred_params 生成基础骨架
            skeleton_segment = self.modeler.shape_function(segment_t, *pred_params, total_hours)
            
            # 应用节律
            speed_norm_segment, _ = self.seasonality.apply_seasonality(
                segment_t, skeleton_segment, target.meta.start_at,
                total_hours=total_hours, t_panic=pred_params[4]
            )
            
            # 积分得到无缩放的总增量
            # speed_norm 是归一化的，需要乘 target.scale 才是 pt/hr
            # 积分: sum(speed * dt)
            dt = 0.1
            raw_increment = np.sum(speed_norm_segment * target.scale * dt)
            
            # B. 计算所需的 Scale Factor
            # 我们需要: last_s + raw_increment * required_scale = next_s
            target_increment = next_s - last_s
            
            if raw_increment > 0:
                required_scale = target_increment / raw_increment
            else:
                required_scale = 1.0 # 避免除零，虽然不太可能
            
            # C. 生成该段的虚拟数据
            # Score(t) = last_s + cumsum(speed(t) * scale * dt)
            # segment_speed_real 单位是 pt/hour
            segment_speed_real = speed_norm_segment * target.scale * required_scale
            segment_score_inc = np.cumsum(segment_speed_real * dt)
            segment_score = last_s + segment_score_inc
            
            # D. 构造 DataFrame 片段并追加
            # 注意：我们需要构造完整的列以保持一致性
            # time = start_at + hours * 3600 * 1000
            segment_ts = target.meta.start_at + (segment_t * 3600 * 1000)
            
            # 计算 norm_speed
            # norm_speed = speed(pt/min) / scale
            # segment_speed_real 是 pt/hour
            # 所以 speed(pt/min) = segment_speed_real / 60.0
            # norm_speed = (segment_speed_real / 60.0) / target.scale
            #            = (speed_norm_segment * target.scale * required_scale / 60.0) / target.scale
            #            = speed_norm_segment * required_scale / 60.0
            
            # calculate_derived_columns 里：
            # speed = diff_val / diff_time(min)  -> pt/min
            # norm_speed = speed / scale         -> (pt/min) / scale
            
            # 而这里 segment_speed_real 是 pt/hour
            # 所以对应的 speed (pt/min) 是 segment_speed_real / 60.0
            speed_pt_min = segment_speed_real / 60.0
            
            # 对应的 norm_speed
            norm_speed_val = speed_pt_min / target.scale
            
            segment_df = pd.DataFrame({
                'hours_elapsed': segment_t,
                'value': segment_score,
                'time': segment_ts,
                'speed': speed_pt_min,
                'norm_speed': norm_speed_val,
                'is_manual': True # 标记为人工数据
            })
            
            full_df = pd.concat([full_df, segment_df], ignore_index=True)
            
            last_t = next_t
            last_s = next_s

        return full_df, last_t

    def _apply_smoothing(self, speed_pred_norm: np.ndarray, scale_curve: np.ndarray) -> np.ndarray:
        """
        应用高分段的平滑压制逻辑 (Top-speed Smoothing)。
        现在 scale_curve 是一个数组。
        """
        # 逐点相乘
        norm_after_scale = speed_pred_norm * scale_curve
        
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
                debug_hours: Optional[float] = None,
                manual_points: Optional[List[dict]] = None) -> PredictionResult:
        """主预测入口。"""
        
        # 0. 准备数据：去节律化
        # 注意：如果已经有人工数据，这里 remove_seasonality 可能会有问题，
        # 但目前 target.df 还是纯净的。
        target.df = self.seasonality.remove_seasonality(target.df)
        
        # 1. 确定对比窗口 (基于真实数据)
        t_start_cmp = float(self.config.get('t_start_cmp', 6.0))
        observed_hours = float(target.df['hours_elapsed'].max())
        end_source = debug_hours if debug_hours is not None else observed_hours
        
        # 动态调整对比上限：允许对比到活动结束前 24 小时，或者至少 72 小时
        # 这样在活动后半段，Ratio 依然会随着新数据的进入而更新
        dynamic_cap = max(float(self.config.get('t_end_cap', 72.0)), target.meta.total_hours - 24.0)
        t_end_cmp = min(end_source, dynamic_cap)
        
        # 2. 计算 Ratio (基于真实数据)
        # 我们希望 Ratio 反映的是“该活动目前的自然强度”，而不是被人工干预后的强度
        ratio = self._calculate_ratio(target, history, t_start_cmp, t_end_cmp)
        logger.info(f"Calculated Ratio: {ratio:.4f}")

        # 3. 获取并修正参数
        avg_params = self._fit_history_params(history)
        pred_params = avg_params.copy()
        
        # === 时长归一化修正 (Time-Scale Normalization) ===
        # 计算历史活动的平均时长
        hist_durations = [h.meta.total_hours for h in history if h.meta.total_hours > 0]
        avg_hist_duration = np.mean(hist_durations) if hist_durations else 192.0 # 默认8天
        target_duration = target.meta.total_hours
        
        # 计算时长倍率 (Scale Length)
        # 如果当前活动比历史长，len_ratio > 1.0
        len_ratio = target_duration / avg_hist_duration if avg_hist_duration > 0 else 1.0
        
        # if len_ratio > 1.1 or len_ratio < 0.9:
        if len_ratio > 1.15:
            logger.info(f"检测到时长差异 (Target: {target_duration:.1f}h vs Hist: {avg_hist_duration:.1f}h), "
                        f"应用参数稀释: ratio={len_ratio:.2f}")
            
            # 修正线性项 A: V ~ A*t -> 为了保持 V 不变，A 需除以 len_ratio
            pred_params[1] /= len_ratio
            
            # 修正二次项 B: V ~ B*t^2 -> 为了保持 V 不变，B 需除以 len_ratio^2
            pred_params[2] /= (len_ratio ** 2.0)
            
            # 恐慌点 B_end 通常与时长关系不大
        # =======================================================

        pred_params[0] *= ratio        # Base
        pred_params[1] *= ratio        # A
        pred_params[2] *= ratio        # B
        pred_params[3] *= (ratio ** 1.1) # B_end

        # --- 尝试用当前观测在线重拟合形状参数 (A, B)，并按置信度融合 ---
        try:
            target_params = self._refit_shape_params(target, pred_params)
            if target_params is not None:
                conf = float(np.clip(observed_hours / float(self.config.get('refit_conf_norm_hours', 24.0)), 0.0, 0.9))
                # 当观测足够多时，优先使用观测拟合的参数
                if conf > 0.0:
                    logger.info(f"Blending shape params with weight={conf:.3f}")
                    pred_params = (1.0 - conf) * pred_params + conf * target_params
        except Exception as e:
            logger.warning(f"Shape refit blend failed: {e}")
        
        # === 3.5 插入人工干预逻辑 ===
        # 如果有人工点，我们需要先生成“虚拟历史”，然后基于这个虚拟历史来做后续的 Scale 计算
        
        # 标记原始数据，方便绘图区分
        if 'is_manual' not in target.df.columns:
            target.df['is_manual'] = False
            
        if manual_points:
            logger.info(f"应用人工干预点: {len(manual_points)} 个")
            # 使用当前的 pred_params (已应用 Ratio) 来生成符合当前趋势的虚拟路径
            # 这样生成的路径既符合人工设定的终点，又符合当前活动的节律特征
            target.df, new_max_time = self._generate_hypothetical_path(target, manual_points, pred_params)
            logger.info(f"数据已扩展至: {new_max_time:.1f}h")
            
        # ==========================

        # 4. 生成基础预测曲线 (Skeleton)
        total_hours = target.meta.total_hours
        future_t = np.linspace(0, total_hours, 1000)
        skeleton_pred = self.modeler.shape_function(future_t, *pred_params, total_hours)
        
        # 5. 应用节律 (Re-seasonality)
        speed_pred_norm, _ = self.seasonality.apply_seasonality(
            future_t, skeleton_pred, target.meta.start_at,
            total_hours=total_hours, t_panic=pred_params[4]
        )
        
        # 6. 计算 Scale Factor (Kalman Filter)
        # 注意：此时 target.df 可能已经包含了人工生成的“未来数据”
        # _calculate_scale_factor 会自动使用最新的数据（包括人工数据）来更新 KF 状态
        # 从而让预测曲线自然地接在人工路径后面
        base_scale, scale_curve = self._calculate_scale_factor(target, future_t, speed_pred_norm)
        
        # 限制 Scale 对 Panic Term 的过度影响。
        # 在进入恐慌期（Panic Phase）后，逐渐让 Scale 回归 1.0，
        # 从而让结尾的走势主要由 B_end 和 panic_scaler 等模型参数决定，防止乘数爆炸。
        
        t_panic_duration = float(pred_params[4]) # 获取拟合出的恐慌时长
        if t_panic_duration > 0:
            t_start_panic = total_hours - t_panic_duration
            
            # 1. 计算每个时间点进入恐慌期的进度 (0.0 ~ 1.0)
            # 使用 clip 确保只在最后阶段生效
            panic_progression = np.clip((future_t - t_start_panic) / t_panic_duration, 0.0, 1.0)
            
            # 2. 只有在真正进入恐慌期的时间点才生效
            mask_active = future_t > t_start_panic
            
            # 3. 混合逻辑：
            # 进度为 0 时（刚进恐慌期）：完全使用 scale_curve
            # 进度为 1 时（活动结束）：完全使用 1.0 (完全信任模型参数)
            # 这里使用平方插值 (prog^2) 让过渡更平滑，越接近结束，归一化力度越大
            blend_weight = np.power(panic_progression, 2.0)
            
            # 应用阻尼
            scale_curve[mask_active] = (
                scale_curve[mask_active] * (1.0 - blend_weight[mask_active]) + 
                1.0 * blend_weight[mask_active]
            )
        
        # 7. 应用平滑压制 (传入曲线)
        final_speed_norm = self._apply_smoothing(speed_pred_norm, scale_curve)
        
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
            scale_factor=base_scale, # 这里只存一个基准值用于展示
            full_t_score=full_t_score,
            full_score=full_score
        )