# plotly_viz.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from domain_models import EventData, PredictionResult

TIER_COLORS: Dict[int, str] = {
    500: '#1f77b4',
    1000: '#2ca02c',
    1500: '#ff7f0e',
    2000: '#d62728',
}

TIER_DASH_STYLES: Dict[int, str] = {
    500: 'solid',
    1000: 'dash',
    1500: 'dot',
    2000: 'dashdot',
}


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
    tier_data: Dict[int, EventData],
    tier_results: Dict[int, PredictionResult],
    debug_hours: float = None,
    manual_points: list = None,
    tz_offset: int = 8,
) -> go.Figure:
    """使用 Plotly 绘制多线交互式预测图（速度 + 分数双面板）"""

    tiers = sorted(t for t in tier_data if t in tier_results)
    if not tiers:
        return go.Figure()

    primary = tier_data[tiers[0]]
    start_ts = primary.meta.start_at
    event_id = primary.meta.event_id

    has_manual = 'is_manual' in primary.df.columns

    if debug_hours is not None:
        now_hours = debug_hours
    else:
        if has_manual:
            real_df = primary.df[~primary.df['is_manual']] if has_manual else primary.df
            now_hours = real_df['hours_elapsed'].max() if not real_df.empty else 0
        else:
            now_hours = primary.df['hours_elapsed'].max()
    now_dt = _first(_to_real_time(np.array([now_hours]), start_ts, tz_offset))

    # ===== 构建标题 =====
    pred_scores = [f"T{t}: {int(tier_results[t].final_score):,}" for t in tiers]
    title_str = f"Event {event_id} Prediction  |  {'  |  '.join(pred_scores)}"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.45, 0.55],
        subplot_titles=(
            f"Event {event_id} Speed Prediction",
            "Score Prediction",
        ),
    )

    # ========== 面板 1: Speed（各线叠加）==========
    for tier in tiers:
        data = tier_data[tier]
        result = tier_results[tier]
        color = TIER_COLORS.get(tier, '#888888')
        dash = TIER_DASH_STYLES.get(tier, 'solid')
        tier_label = f"T{tier}"

        if has_manual:
            mask_real = ~data.df['is_manual']
            mask_manual = data.df['is_manual']
            obs_hours = data.df.loc[mask_real, 'hours_elapsed'].values
            obs_speed = data.df.loc[mask_real, 'norm_speed'].values
        else:
            obs_hours = data.df['hours_elapsed'].values
            obs_speed = data.df['norm_speed'].values
        obs_time = _to_real_time(obs_hours, start_ts, tz_offset)

        pred_time = _to_real_time(result.future_t, start_ts, tz_offset)

        fig.add_trace(
            go.Scatter(
                x=obs_time, y=obs_speed,
                mode='lines',
                name=f'{tier_label} Speed',
                line=dict(color=color, width=2),
                legendgroup=f't{tier}',
                hovertemplate=f'T{tier} Speed: %{{y:.4f}}<extra></extra>',
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=pred_time, y=result.future_speed,
                mode='lines',
                name=f'{tier_label} Pred Speed',
                line=dict(color=color, dash=dash, width=1.5),
                opacity=0.7,
                legendgroup=f't{tier}',
                hovertemplate=f'T{tier} Pred Speed: %{{y:.4f}}<extra></extra>',
            ),
            row=1, col=1,
        )

    # 人工干预路径（只画一次，不区分层级）
    if has_manual:
        mask_manual = primary.df['is_manual']
        manual_hours = primary.df.loc[mask_manual, 'hours_elapsed'].values
        manual_speed = primary.df.loc[mask_manual, 'norm_speed'].values
        if len(manual_hours) > 0:
            manual_time_dt = _to_real_time(manual_hours, start_ts, tz_offset)
            fig.add_trace(
                go.Scatter(
                    x=manual_time_dt, y=manual_speed,
                    mode='lines',
                    name='Hypothetical Path',
                    line=dict(color='magenta', dash='dash', width=2),
                    opacity=0.7,
                    hovertemplate='Hypothetical: %{y:.4f}<extra></extra>',
                ),
                row=1, col=1,
            )

    # ========== 面板 2: Score（各线叠加）==========
    for tier in tiers:
        data = tier_data[tier]
        result = tier_results[tier]
        color = TIER_COLORS.get(tier, '#888888')
        dash = TIER_DASH_STYLES.get(tier, 'solid')
        tier_label = f"T{tier}"

        if has_manual:
            mask_real = ~data.df['is_manual']
            obs_hours = data.df.loc[mask_real, 'hours_elapsed'].values
            obs_score = data.df.loc[mask_real, 'value'].values
        else:
            obs_hours = data.df['hours_elapsed'].values
            obs_score = data.df['value'].values
        obs_time = _to_real_time(obs_hours, start_ts, tz_offset)

        full_time = _to_real_time(result.full_t_score, start_ts, tz_offset)

        fig.add_trace(
            go.Scatter(
                x=obs_time, y=obs_score,
                mode='lines',
                name=f'{tier_label} Score',
                line=dict(color=color, width=2),
                legendgroup=f't{tier}',
                hovertemplate=f'T{tier} Score: %{{y:,.0f}}<extra></extra>',
            ),
            row=2, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=full_time, y=result.full_score,
                mode='lines',
                name=f'{tier_label} Pred Curve',
                line=dict(color=color, dash=dash, width=2.5),
                legendgroup=f't{tier}',
                hovertemplate=f'T{tier} Pred: %{{y:,.0f}}<extra></extra>',
            ),
            row=2, col=1,
        )

        pred_final_time = _last(full_time)
        pred_final_score = result.final_score

        fig.add_annotation(
            x=pred_final_time, y=pred_final_score,
            text=f"<b>T{tier}: {int(pred_final_score):,}</b>",
            showarrow=False,
            font=dict(color=color, size=12),
            xanchor='right',
            yanchor='bottom',
            row=2, col=1,
        )

    # 人工干预（分数面板，只画一次）
    if has_manual:
        manual_score = primary.df.loc[mask_manual, 'value'].values
        if len(manual_hours) > 0:
            manual_time_dt = _to_real_time(manual_hours, start_ts, tz_offset)
            fig.add_trace(
                go.Scatter(
                    x=manual_time_dt, y=manual_score,
                    mode='lines',
                    name='Hypothetical Path',
                    line=dict(color='magenta', dash='dash', width=2),
                    opacity=0.7,
                    hovertemplate='Hypothetical: %{y:,.0f}<extra></extra>',
                    legendgroup='manual',
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
                    hovertemplate='Manual: %{y:,.0f}<extra></extra>',
                    legendgroup='manual',
                    showlegend=True,
                ),
                row=2, col=1,
            )

    # 实际曲线（debug 模式，逐线绘制）
    if debug_hours is not None:
        for tier in tiers:
            data = tier_data[tier]
            if data.full_df is None or data.full_df.empty:
                continue
            color = TIER_COLORS.get(tier, '#888888')
            full_df = data.full_df
            full_real_time = _to_real_time(full_df['hours_elapsed'].values, start_ts, tz_offset)
            fig.add_trace(
                go.Scatter(
                    x=full_real_time, y=full_df['value'],
                    mode='lines',
                    name=f'T{tier} Actual',
                    line=dict(color=color, dash='dot', width=1.5),
                    opacity=0.5,
                    hovertemplate=f'T{tier} Actual: %{{y:,.0f}}<extra></extra>',
                    legendgroup=f't{tier}',
                    showlegend=True,
                ),
                row=2, col=1,
            )
            real_final_score = full_df.iloc[-1]['value']
            real_final_time = _last(full_real_time)
            fig.add_trace(
                go.Scatter(
                    x=[real_final_time], y=[real_final_score],
                    mode='markers',
                    name=f'T{tier} Actual Final',
                    marker=dict(color=color, size=10, symbol='x'),
                    hovertemplate=f'T{tier} Final: %{{y:,.0f}}<extra></extra>',
                    legendgroup=f't{tier}',
                    showlegend=True,
                ),
                row=2, col=1,
            )
            fig.add_annotation(
                x=real_final_time, y=real_final_score,
                text=f"<b>T{tier} Act: {int(real_final_score):,}</b>",
                showarrow=False,
                font=dict(color=color, size=11),
                xanchor='right',
                yanchor='top',
                row=2, col=1,
            )

    # ===== Now 竖线 =====
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

    # ===== 水印 =====
    fig.add_annotation(
        x=1, y=0, xref='paper', yref='paper',
        text='@byydzh mycx multi',
        showarrow=False,
        font=dict(color='gray', size=10),
        opacity=0.6,
        xanchor='right', yanchor='bottom',
    )

    # ===== Layout =====
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
