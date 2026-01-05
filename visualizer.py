# visualizer.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from datetime import datetime, timedelta, timezone

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from math_models import CosineModeler
from domain_models import EventData, PredictionResult

class Visualizer:
    def __init__(self, output_dir='output', tz_offset=8):
        self.output_dir = output_dir
        self.tz_offset = tz_offset
        self.modeler = CosineModeler()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    def _to_real_time(self, hours_array, start_ts):
        """将相对小时数转换为本地时间对象"""
        start_dt_utc = datetime.fromtimestamp(start_ts / 1000, timezone.utc)
        start_dt_local = (start_dt_utc + timedelta(hours=self.tz_offset)).replace(tzinfo=None)
        deltas = pd.to_timedelta(hours_array, unit='h')
        return start_dt_local + deltas

    def plot_prediction(self, target: EventData, result: PredictionResult,
                        debug_hours: float = None, manual_points: list = None, save=True):
        """绘制最终预测图 (包含真实未来对比)"""
        
        start_ts = target.meta.start_at
        
        # 1. 准备基础时间轴
        # 区分真实观测数据和人工生成的数据
        if 'is_manual' in target.df.columns:
            mask_real = ~target.df['is_manual']
            mask_manual = target.df['is_manual']
            
            obs_time = self._to_real_time(target.df.loc[mask_real, 'hours_elapsed'].values, start_ts)
            obs_score = target.df.loc[mask_real, 'value'].values
            obs_speed = target.df.loc[mask_real, 'norm_speed'].values
            
            manual_time = self._to_real_time(target.df.loc[mask_manual, 'hours_elapsed'].values, start_ts)
            manual_score = target.df.loc[mask_manual, 'value'].values
            manual_speed = target.df.loc[mask_manual, 'norm_speed'].values
        else:
            obs_time = self._to_real_time(target.df['hours_elapsed'].values, start_ts)
            obs_score = target.df['value'].values
            obs_speed = target.df['norm_speed'].values
            manual_time, manual_score, manual_speed = [], [], []

        pred_time = self._to_real_time(result.future_t, start_ts)
        full_time = self._to_real_time(result.full_t_score, start_ts)
        
        # 重算骨架曲线
        skeleton_y = self.modeler.shape_function(
            result.future_t,
            *result.used_params,
            target.meta.total_hours
        )
        
        # 确定 "Now" 线的位置：如果有 debug_hours，优先用它；否则用真实数据的最后时间
        # 注意：不应该用人工数据的最后时间，因为那是“未来”
        if debug_hours:
            now_hours = debug_hours
        else:
            # 找到最后一个非人工点的时间
            if 'is_manual' in target.df.columns:
                real_df = target.df[~target.df['is_manual']]
                now_hours = real_df['hours_elapsed'].max() if not real_df.empty else 0
            else:
                now_hours = target.df['hours_elapsed'].max()
                
        now_dt = self._to_real_time([now_hours], start_ts)[0]

        # 2. 绘图设置
        fig = Figure(figsize=(12, 10))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        date_fmt = mdates.DateFormatter('%m-%d %H:%M')

        # --- Subplot 1: Speed ---
        # 观测骨架
        if 'skeleton_speed' in target.df.columns:
            # 确保 skeleton_speed 和 obs_time 长度一致
            # obs_time 是基于 mask_real 筛选的，所以这里也需要筛选
            if 'is_manual' in target.df.columns:
                mask_real = ~target.df['is_manual']
                skel_speed = target.df.loc[mask_real, 'skeleton_speed'].values
            else:
                skel_speed = target.df['skeleton_speed'].values
            
            # 再次检查长度，防止潜在的不一致
            if len(obs_time) == len(skel_speed):
                ax1.scatter(obs_time, skel_speed, s=10, c='gray', alpha=0.3, label='观测骨架')
        
        # 预测骨架
        ax1.plot(pred_time, skeleton_y, color='blue', linestyle='--', alpha=0.5, label='预测骨架')

        # 观测速度 (截断后)
        ax1.plot(obs_time, obs_speed, c='red', lw=2, label='观测速度')
        
        # 人工干预速度 (虚线)
        if len(manual_time) > 0:
            ax1.plot(manual_time, manual_speed, c='magenta', lw=2, ls='--', alpha=0.7, label='假设路径')

        # 预测速度
        ax1.plot(pred_time, result.future_speed, c='green', lw=2, alpha=0.8, label='预测速度')
        
        # 真实未来速度 (橙色线)
        if target.full_df is not None:
            full_df = target.full_df
            # 筛选出当前时间之后的数据
            mask_future = full_df['hours_elapsed'] > now_hours
            if mask_future.any():
                future_real_time = self._to_real_time(full_df.loc[mask_future, 'hours_elapsed'].values, start_ts)
                future_real_speed = full_df.loc[mask_future, 'norm_speed'].values
                ax1.plot(future_real_time, future_real_speed,
                         color='orange', linestyle='-.', linewidth=2, alpha=0.9, label='实际未来速度')

        ax1.axvline(x=now_dt, color='black', linestyle=':', label='当前时刻')
        ax1.set_title(f"活动 {target.meta.event_id} 速度预测")
        ax1.set_ylabel("归一化速度")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(date_fmt)

        # --- Subplot 2: Score ---
        # 观测分数 (截断后)
        ax2.plot(obs_time, obs_score, c='red', lw=2, label='观测分数')
        
        # 人工干预分数 (虚线)
        if len(manual_time) > 0:
            ax2.plot(manual_time, manual_score, c='magenta', lw=2, ls='--', alpha=0.7, label='假设路径')
            # 绘制关键点 (星星)
            if manual_points:
                mp_hours = [p['hours'] for p in manual_points]
                mp_scores = [p['score'] for p in manual_points]
                mp_times = self._to_real_time(mp_hours, start_ts)
                ax2.scatter(mp_times, mp_scores, marker='*', s=150, c='magenta', zorder=10, label='干预点')

        # 预测分数曲线
        ax2.plot(full_time, result.full_score, c='purple', ls='--', lw=2, label='预测曲线')
        
        # 真实分数曲线 (全量) - 仅在回测(debug_hours不为空)时绘制
        real_final_score = 0
        if debug_hours is not None and target.full_df is not None:
            full_df = target.full_df
            full_real_time = self._to_real_time(full_df['hours_elapsed'].values, start_ts)
            
            # 只有当 full_df 确实包含比当前观测点更未来的数据时才画
            if not full_df.empty:
                ax2.plot(full_real_time, full_df['value'], c='orange', alpha=0.4, lw=1, label='实际曲线')
                
                # 获取真实最终分
                real_final_score = full_df.iloc[-1]['value']
                # [修正语法错误] DatetimeIndex 不支持 .iloc，直接用 [-1]
                real_final_time = full_real_time[-1]
                
                # 绘制深红点
                ax2.scatter(real_final_time, real_final_score, color='darkred', s=60, zorder=5, label='实际最终分')

        # --- 文字标签 (三巨头) ---
        # 1. 当前分 (Current)
        current_score = target.df['value'].max() if not target.df.empty else 0
        ax2.text(now_dt, current_score, f" 当前: {int(current_score):,}",
                 ha='left', va='top', fontsize=10, color='red', fontweight='bold')

        # 2. 预测最终分 (Predicted)
        final_t = full_time[-1]
        final_s = result.final_score
        ax2.text(final_t, final_s, f"预测: {int(final_s):,}\n",
                 ha='right', va='bottom', fontsize=11, fontweight='bold', color='purple')

        # 3. 真实最终分 (Actual) - 仅在回测且有值时显示
        if real_final_score > 0:
            # # 为了防止和预测分重叠，如果两者接近，稍微错开一点位置
            offset_y = real_final_score
            if abs(real_final_score - final_s) < (final_s * 0.15):
                offset_y = min(0.8 * real_final_score, final_s * 0.85)  # 向下挪一点
            
            # 计算真实结束时间
            real_final_t = self._to_real_time([target.meta.total_hours], start_ts)[0]
            ax2.text(real_final_t, offset_y, f"\n实际: {int(real_final_score):,}",
                     ha='right', va='top', fontsize=11, fontweight='bold', color='darkred')

        ax2.axvline(x=now_dt, color='black', linestyle=':')
        ax2.set_title(f"分数预测: {int(final_s):,} PT")
        ax2.set_ylabel("活动分数")
        ax2.set_xlabel("本地时间")
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(date_fmt)
        
        # 水印
        try:
            wm_text = "@byydzh mycx 1000"
            fig.text(0.99, 0.01, wm_text, fontsize=10, color='gray', alpha=0.6,
                     ha='right', va='bottom', zorder=100)
        except Exception:
            pass

        fig.autofmt_xdate()
        fig.tight_layout()

        if save:
            ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f"pred_{target.meta.event_id}_{ts_str}.png"
            out_path = os.path.join(self.output_dir, str(target.meta.event_id))
            os.makedirs(out_path, exist_ok=True)
            full_path = os.path.join(out_path, fname)
            fig.savefig(full_path, dpi=150)
            print(f"绘图已保存: {full_path}")
            return full_path
        
        return fig