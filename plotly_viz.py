# plotly_viz.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from math_models import CosineModeler
from domain_models import EventData, PredictionResult


def _to_real_time(hours_array, start_ts, tz_offset=8):
    start_dt_utc = datetime.fromtimestamp(start_ts / 1000, timezone.utc)
    start_dt_local = (start_dt_utc + timedelta(hours=tz_offset)).replace(tzinfo=None)
    deltas = pd.to_timedelta(hours_array, unit='h')
    result = start_dt_local + deltas
    if hasattr(result, 'to_pydatetime'):
        converted = result.to_pydatetime()
        if isinstance(converted, np.ndarray):
            return converted
        return np.array([converted]) if not hasattr(converted, '__iter__') else converted
    return result


def _first(arr):
    if hasattr(arr, 'iloc'):
        val = arr.iloc[0]
    elif hasattr(arr, '__getitem__'):
        val = arr[0]
    else:
        val = arr
    return val.to_pydatetime() if hasattr(val, 'to_pydatetime') else val


def _last(arr):
    if hasattr(arr, 'iloc'):
        val = arr.iloc[-1]
    elif hasattr(arr, '__getitem__'):
        val = arr[-1]
    else:
        val = arr
    return val.to_pydatetime() if hasattr(val, 'to_pydatetime') else val


def plot_prediction_plotly(
    target: EventData,
    result: PredictionResult,
    debug_hours: float = None,
    manual_points: list = None,
    tz_offset: int = 8,
) -> go.Figure:
    """使用 Plotly 绘制交互式预测图，复刻原 matplotlib 双面板视觉风格"""

    modeler = CosineModeler()
    start_ts = target.meta.start_at

    has_manual = 'is_manual' in target.df.columns
    if has_manual:
        mask_real = ~target.df['is_manual']
        mask_manual = target.df['is_manual']
        obs_hours = target.df.loc[mask_real, 'hours_elapsed'].values
        obs_score = target.df.loc[mask_real, 'value'].values
        obs_speed = target.df.loc[mask_real, 'norm_speed'].values
        manual_hours = target.df.loc[mask_manual, 'hours_elapsed'].values
        manual_score = target.df.loc[mask_manual, 'value'].values
        manual_speed = target.df.loc[mask_manual, 'norm_speed'].values
    else:
        obs_hours = target.df['hours_elapsed'].values
        obs_score = target.df['value'].values
        obs_speed = target.df['norm_speed'].values
        manual_hours = np.array([])
        manual_score = np.array([])
        manual_speed = np.array([])

    obs_time = _to_real_time(obs_hours, start_ts, tz_offset)
    pred_time = _to_real_time(result.future_t, start_ts, tz_offset)
    full_time = _to_real_time(result.full_t_score, start_ts, tz_offset)
    manual_time_dt = _to_real_time(manual_hours, start_ts, tz_offset) if len(manual_hours) > 0 else []

    skeleton_y = modeler.shape_function(
        result.future_t,
        *result.used_params,
        target.meta.total_hours,
    )

    if debug_hours is not None:
        now_hours = debug_hours
    else:
        if has_manual:
            real_df = target.df[~target.df['is_manual']]
            now_hours = real_df['hours_elapsed'].max() if not real_df.empty else 0
        else:
            now_hours = target.df['hours_elapsed'].max()
    now_dt = _first(_to_real_time(np.array([now_hours]), start_ts, tz_offset))

    # ===== 提取真实最终分 =====
    real_final_score = 0
    real_final_time = None
    if debug_hours is not None and target.full_df is not None and not target.full_df.empty:
        full_df = target.full_df
        real_final_score = full_df.iloc[-1]['value']
        real_final_time = _to_real_time(
            np.array([target.meta.total_hours]), start_ts, tz_offset
        )
        real_final_time = _first(real_final_time)

    pred_final_score = result.final_score
    pred_final_time = _last(full_time)
    error_val = pred_final_score - real_final_score if real_final_score > 0 else None
    error_pct = (error_val / real_final_score * 100) if error_val is not None else None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.45, 0.55],
        subplot_titles=(
            f"Event {target.meta.event_id} Speed Prediction",
            f"Score Prediction: {int(pred_final_score):,} PT",
        ),
    )

    # ========================================
    # 面板 1: Speed
    # ========================================
    if 'skeleton_speed' in target.df.columns:
        if has_manual:
            skel_speed = target.df.loc[mask_real, 'skeleton_speed'].values
        else:
            skel_speed = target.df['skeleton_speed'].values
        if len(obs_time) == len(skel_speed):
            fig.add_trace(
                go.Scatter(
                    x=obs_time, y=skel_speed,
                    mode='markers',
                    name='Observed Skeleton',
                    marker=dict(color='lightgray', size=4),
                    opacity=0.5,
                    hovertemplate='%{x|%m-%d %H:%M}<br>Skeleton: %{y:.4f}<extra></extra>',
                ),
                row=1, col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=pred_time, y=skeleton_y,
            mode='lines',
            name='Predicted Skeleton',
            line=dict(color='blue', dash='dash', width=1),
            opacity=0.6,
            hovertemplate='%{x|%m-%d %H:%M}<br>Skeleton: %{y:.4f}<extra></extra>',
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=obs_time, y=obs_speed,
            mode='lines',
            name='Observed Speed',
            line=dict(color='red', width=2),
            hovertemplate='%{x|%m-%d %H:%M}<br>Speed: %{y:.4f}<extra></extra>',
        ),
        row=1, col=1,
    )

    if len(manual_time_dt) > 0:
        fig.add_trace(
            go.Scatter(
                x=manual_time_dt, y=manual_speed,
                mode='lines',
                name='Hypothetical Path',
                line=dict(color='magenta', dash='dash', width=2),
                opacity=0.7,
                hovertemplate='%{x|%m-%d %H:%M}<br>Speed: %{y:.4f}<extra></extra>',
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=pred_time, y=result.future_speed,
            mode='lines',
            name='Predicted Speed',
            line=dict(color='green', width=2),
            opacity=0.8,
            hovertemplate='%{x|%m-%d %H:%M}<br>Speed: %{y:.4f}<extra></extra>',
        ),
        row=1, col=1,
    )

    if target.full_df is not None:
        full_df = target.full_df
        mask_future = full_df['hours_elapsed'] > now_hours
        if mask_future.any():
            future_real_time = _to_real_time(
                full_df.loc[mask_future, 'hours_elapsed'].values, start_ts, tz_offset
            )
            future_real_speed = full_df.loc[mask_future, 'norm_speed'].values
            fig.add_trace(
                go.Scatter(
                    x=future_real_time, y=future_real_speed,
                    mode='lines',
                    name='Actual Future Speed',
                    line=dict(color='orange', dash='dashdot', width=2),
                    opacity=0.9,
                    hovertemplate='%{x|%m-%d %H:%M}<br>Speed: %{y:.4f}<extra></extra>',
                ),
                row=1, col=1,
            )

    # ========================================
    # 面板 2: Score
    # ========================================
    fig.add_trace(
        go.Scatter(
            x=obs_time, y=obs_score,
            mode='lines',
            name='Observed Score',
            line=dict(color='red', width=2),
            hovertemplate='%{x|%m-%d %H:%M}<br>Score: %{y:,.0f}<extra></extra>',
            legendgroup='score',
            showlegend=True,
        ),
        row=2, col=1,
    )

    if len(manual_time_dt) > 0:
        fig.add_trace(
            go.Scatter(
                x=manual_time_dt, y=manual_score,
                mode='lines',
                name='Hypothetical Path',
                line=dict(color='magenta', dash='dash', width=2),
                opacity=0.7,
                hovertemplate='%{x|%m-%d %H:%M}<br>Score: %{y:,.0f}<extra></extra>',
                legendgroup='score',
                showlegend=True,
            ),
            row=2, col=1,
        )
        if manual_points:
            mp_hours = np.array([p['hours'] for p in manual_points])
            mp_scores = np.array([p['score'] for p in manual_points])
            mp_times = _to_real_time(mp_hours, start_ts, tz_offset)
            fig.add_trace(
                go.Scatter(
                    x=mp_times, y=mp_scores,
                    mode='markers',
                    marker=dict(symbol='star', size=12, color='magenta'),
                    name='Manual Points',
                    hovertemplate='%{x|%m-%d %H:%M}<br>Score: %{y:,.0f}<extra></extra>',
                    legendgroup='score',
                    showlegend=True,
                ),
                row=2, col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=full_time, y=result.full_score,
            mode='lines',
            name='Predicted Curve',
            line=dict(color='purple', dash='dash', width=3),
            hovertemplate='%{x|%m-%d %H:%M}<br>Score: %{y:,.0f}<extra></extra>',
            legendgroup='score',
            showlegend=True,
        ),
        row=2, col=1,
    )

    if debug_hours is not None and target.full_df is not None and not target.full_df.empty:
        full_df = target.full_df
        full_real_time = _to_real_time(full_df['hours_elapsed'].values, start_ts, tz_offset)
        fig.add_trace(
            go.Scatter(
                x=full_real_time, y=full_df['value'],
                mode='lines',
                name='Actual Curve',
                line=dict(color='orange', width=2),
                opacity=0.75,
                hovertemplate='%{x|%m-%d %H:%M}<br>Score: %{y:,.0f}<extra></extra>',
                legendgroup='score',
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[real_final_time], y=[real_final_score],
                mode='markers',
                name='Actual Final',
                marker=dict(color='darkred', size=10),
                hovertemplate='Actual Final<br>%{x|%m-%d %H:%M}<br>%{y:,.0f}<extra></extra>',
                legendgroup='score',
                showlegend=True,
            ),
            row=2, col=1,
        )

    # ========================================
    # Now 竖线 — 用 shape yref=paper 贯穿双面板
    # ========================================
    fig.add_shape(
        type='line',
        x0=now_dt, x1=now_dt, y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='black', width=1.5, dash='dot'),
    )
    fig.add_annotation(
        x=now_dt, y=1.01, xref='x', yref='paper',
        text='Now',
        showarrow=False,
        font=dict(color='black', size=10),
        yanchor='bottom',
    )

    # ========================================
    # 分数标注: Pred / Act / Error
    # ========================================
    fig.add_annotation(
        x=pred_final_time, y=pred_final_score,
        text=f"<b>Pred: {int(pred_final_score):,}</b>",
        showarrow=False,
        font=dict(color='purple', size=14),
        xanchor='right',
        yanchor='bottom',
        row=2, col=1,
    )

    if real_final_score > 0:
        offset_y = real_final_score
        if abs(real_final_score - pred_final_score) < (pred_final_score * 0.15):
            offset_y = min(0.8 * real_final_score, pred_final_score * 0.85)
        fig.add_annotation(
            x=real_final_time, y=offset_y,
            text=f"<b>Act: {int(real_final_score):,}</b>",
            showarrow=False,
            font=dict(color='darkred', size=14),
            xanchor='right',
            yanchor='top',
            row=2, col=1,
        )

        if error_val is not None:
            error_label_y = max(pred_final_score, real_final_score) * 1.05
            fig.add_annotation(
                x=real_final_time, y=error_label_y,
                text=f"<b>Error: {error_val:+,.0f}  /  {error_pct:+.1f}%</b>",
                showarrow=False,
                font=dict(color='black', size=13),
                xanchor='right',
                yanchor='bottom',
                row=2, col=1,
            )

    # ========================================
    # 水印
    # ========================================
    fig.add_annotation(
        x=1, y=0, xref='paper', yref='paper',
        text='@byydzh mycx 1000',
        showarrow=False,
        font=dict(color='gray', size=10),
        opacity=0.6,
        xanchor='right', yanchor='bottom',
    )

    # ========================================
    # Layout
    # ========================================
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

    fig.update_yaxes(title_text="Normalized Speed", row=1, col=1)
    fig.update_yaxes(title_text="Event Points", row=2, col=1)
    fig.update_xaxes(title_text="Local Time", row=2, col=1)

    fig.update_layout(
        height=850,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.03,
            xanchor='center',
            x=0.5,
        ),
        margin=dict(l=60, r=60, t=90, b=60),
        font=dict(size=12),
    )

    return fig
