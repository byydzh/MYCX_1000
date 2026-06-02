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


def plot_graph_rollout_plotly(
    trajectories: Dict[int, list],
    start_hour: float,
    total_hours: float,
    start_ts: int,
    scale: float,
    reward_tiers: list = None,
    actual_tier_data: Dict[int, pd.DataFrame] = None,
    observed_state_trajectories: Dict[int, list] = None,
    observed_state_hours=None,
    max_tiers: int = 8,
    tz_offset: int = 8,
) -> go.Figure:
    """绘制 GraphStateSpaceEngine 的实验 rollout 与状态诊断。"""
    if not trajectories:
        return go.Figure()

    reward_set = {int(t) for t in (reward_tiers or [])}
    tiers = sorted(trajectories.keys())
    visible_tiers = [t for t in tiers if t in reward_set]
    for tier in tiers:
        if tier not in visible_tiers:
            visible_tiers.append(tier)
        if len(visible_tiers) >= max_tiers:
            break

    max_len = max(len(trajectories[t]) for t in visible_tiers)
    if max_len <= 1:
        hour_grid = np.array([float(start_hour)])
    else:
        hour_grid = np.linspace(float(start_hour), float(total_hours), max_len)
    time_grid = _to_real_time(hour_grid, start_ts, tz_offset)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.48, 0.27, 0.25],
        subplot_titles=(
            "Graph Score Rollout",
            "Graph Normalized Speed",
            "State Diagnostics",
        ),
    )

    actual_tier_data = actual_tier_data or {}
    observed_state_trajectories = observed_state_trajectories or {}
    observed_state_hours = np.asarray(observed_state_hours if observed_state_hours is not None else [], dtype=float)
    observed_state_time = _to_real_time(observed_state_hours, start_ts, tz_offset) if len(observed_state_hours) else []
    for idx, tier in enumerate(visible_tiers):
        nodes = trajectories[tier]
        color = TIER_COLORS.get(tier, None)
        if color is None:
            palette = [
                '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728',
                '#9467bd', '#8c564b', '#17becf', '#7f7f7f',
            ]
            color = palette[idx % len(palette)]
        dash = 'solid' if tier in reward_set else 'dash'
        label = f"T{tier}" + (" reward" if tier in reward_set else "")

        scores = np.array([n.score for n in nodes], dtype=float)
        speeds = np.array([n.speed / max(scale, 1e-6) for n in nodes], dtype=float)
        pressure = np.array([n.pressure for n in nodes], dtype=float)
        affinity = np.array([n.target_affinity for n in nodes], dtype=float)
        dominant_mode = np.array([
            max((n.mode_mix or {'unknown': 1.0}).items(), key=lambda item: item[1])[0]
            for n in nodes
        ], dtype=object)
        observed_nodes = observed_state_trajectories.get(tier, [])
        observed_speed = np.array([n.speed / max(scale, 1e-6) for n in observed_nodes], dtype=float)
        observed_pressure = np.array([n.pressure for n in observed_nodes], dtype=float)
        observed_affinity = np.array([n.target_affinity for n in observed_nodes], dtype=float)
        observed_dominant_mode = np.array([
            max((n.mode_mix or {'unknown': 1.0}).items(), key=lambda item: item[1])[0]
            for n in observed_nodes
        ], dtype=object)

        if tier in actual_tier_data and actual_tier_data[tier] is not None:
            actual_df = actual_tier_data[tier].copy()
            actual_df = actual_df.rename(columns={'ep': 'value', 'timestamp': 'time'})
            if 'time' in actual_df.columns and 'value' in actual_df.columns:
                actual_hours = (actual_df['time'] - start_ts) / 3600000.0
                actual_time = _to_real_time(actual_hours.values, start_ts, tz_offset)
                fig.add_trace(
                    go.Scatter(
                        x=actual_time,
                        y=actual_df['value'],
                        mode='lines',
                        name=f"{label} actual",
                        line=dict(color=color, width=1.2, dash='dot'),
                        opacity=0.45,
                        legendgroup=f'graph-t{tier}',
                        hovertemplate=f'T{tier} actual: %{{y:,.0f}}<extra></extra>',
                    ),
                    row=1, col=1,
                )

        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=scores,
                mode='lines',
                name=f"{label} rollout",
                line=dict(color=color, width=2.2, dash=dash),
                legendgroup=f'graph-t{tier}',
                hovertemplate=f'T{tier} graph score: %{{y:,.0f}}<extra></extra>',
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=speeds,
                mode='lines',
                name=f"{label} speed",
                line=dict(color=color, width=1.8, dash=dash),
                legendgroup=f'graph-t{tier}',
                hovertemplate=f'T{tier} graph norm speed: %{{y:.4f}}<extra></extra>',
            ),
            row=2, col=1,
        )

        if len(observed_nodes) == len(observed_state_hours) and len(observed_nodes) > 0:
            fig.add_trace(
                go.Scatter(
                    x=observed_state_time,
                    y=observed_speed,
                    mode='lines',
                    name=f"{label} filtered state speed",
                    line=dict(color=color, width=1.2, dash='dot'),
                    opacity=0.55,
                    legendgroup=f'graph-t{tier}',
                    hovertemplate=f'T{tier} filtered norm speed: %{{y:.4f}}<extra></extra>',
                ),
                row=2, col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=pressure,
                customdata=dominant_mode,
                mode='lines',
                name=f"{label} pressure",
                line=dict(color=color, width=1.4),
                opacity=0.85,
                legendgroup=f'graph-t{tier}',
                hovertemplate=f'T{tier} pressure: %{{y:.4f}}<br>mode: %{{customdata}}<extra></extra>',
            ),
            row=3, col=1,
        )
        if len(observed_nodes) == len(observed_state_hours) and len(observed_nodes) > 0:
            fig.add_trace(
                go.Scatter(
                    x=observed_state_time,
                    y=observed_pressure,
                    customdata=observed_dominant_mode,
                    mode='lines',
                    name=f"{label} filtered pressure",
                    line=dict(color=color, width=1.0, dash='dot'),
                    opacity=0.45,
                    legendgroup=f'graph-t{tier}',
                    hovertemplate=f'T{tier} filtered pressure: %{{y:.4f}}<br>mode: %{{customdata}}<extra></extra>',
                ),
                row=3, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=observed_state_time,
                    y=observed_affinity,
                    mode='lines',
                    name=f"{label} filtered affinity",
                    line=dict(color=color, width=1.0, dash='dash'),
                    opacity=0.35,
                    legendgroup=f'graph-t{tier}',
                    hovertemplate=f'T{tier} filtered affinity: %{{y:.3f}}<extra></extra>',
                ),
                row=3, col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=time_grid,
                y=affinity,
                mode='lines',
                name=f"{label} affinity",
                line=dict(color=color, width=1.2, dash='dash'),
                opacity=0.6,
                legendgroup=f'graph-t{tier}',
                hovertemplate=f'T{tier} affinity: %{{y:.3f}}<extra></extra>',
            ),
            row=3, col=1,
        )

    now_dt = _first(_to_real_time(np.array([start_hour]), start_ts, tz_offset))
    fig.add_shape(
        type='line',
        x0=now_dt, x1=now_dt, y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='black', width=1.3, dash='dot'),
    )
    fig.add_annotation(
        x=now_dt, y=1.01, xref='x', yref='paper',
        text='Graph Start',
        showarrow=False,
        font=dict(color='black', size=10),
        yanchor='bottom',
    )

    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Speed / T10", row=2, col=1)
    fig.update_yaxes(title_text="Diagnostic", row=3, col=1)
    fig.update_xaxes(title_text="Local Time", row=3, col=1)

    title_reward = f" reward tiers: {sorted(reward_set)}" if reward_set else " reward tiers: none"
    fig.update_layout(
        title=f"Graph State-Space Experimental Rollout |{title_reward}",
        height=930,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.03,
            xanchor='center',
            x=0.5,
        ),
        margin=dict(l=60, r=60, t=100, b=60),
        font=dict(size=12),
    )

    return fig


def plot_graph_cell_state_plotly(
    cell_rows: list,
    reward_tiers: list = None,
    observed_tiers: list = None,
) -> go.Figure:
    """绘制 Graph hidden cohort cell 在起点的完整状态快照。"""
    if not cell_rows:
        return go.Figure()

    df = pd.DataFrame(cell_rows).sort_values('rank_center').reset_index(drop=True)
    reward_set = {int(t) for t in (reward_tiers or [])}
    observed_set = {int(t) for t in (observed_tiers or [])}
    mode_cols = [
        'mode_idle',
        'mode_cruising',
        'mode_watching',
        'mode_chasing',
        'mode_defending',
        'mode_panic_rushing',
        'mode_dropped',
    ]
    existing_mode_cols = [col for col in mode_cols if col in df.columns]
    mode_labels = [col.replace('mode_', '') for col in existing_mode_cols]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.38, 0.32, 0.30],
        subplot_titles=(
            "Cell Score / Mode",
            "Cell Continuous State",
            "Mode Probability Matrix",
        ),
    )

    mode_colors = {
        'idle': '#9ca3af',
        'cruising': '#2563eb',
        'watching': '#8b5cf6',
        'chasing': '#f97316',
        'defending': '#16a34a',
        'panic_rushing': '#dc2626',
        'dropped': '#64748b',
    }
    def _col(name, default):
        return df.get(name, pd.Series([default] * len(df), index=df.index)).to_numpy()

    custom = np.stack([
        _col('rank_start', np.nan),
        _col('rank_end', np.nan),
        _col('cell_size', np.nan),
        _col('speed_norm', np.nan),
        _col('pressure', np.nan),
        _col('target_affinity', np.nan),
        _col('capacity_norm_speed', np.nan),
        _col('target_rank', np.nan),
        _col('target_score', np.nan),
        _col('target_gap_norm', 0.0),
        _col('target_surplus_norm', 0.0),
        _col('pressure_source', 'unknown'),
        _col('target_probs_label', ''),
        _col('behavior_coeffs_label', ''),
        _col('target_follow_norm', 0.0),
        _col('target_plan_lag_norm', 0.0),
        _col('follow_rank', np.nan),
    ], axis=-1)
    hover = (
        "rank %{customdata[0]:.0f}-%{customdata[1]:.0f}"
        "<br>size %{customdata[2]:.0f}"
        "<br>score %{y:,.0f}"
        "<br>speed/T10 %{customdata[3]:.4f}"
        "<br>pressure %{customdata[4]:.3f}"
        "<br>affinity %{customdata[5]:.3f}"
        "<br>capacity/T10 %{customdata[6]:.3f}"
        "<br>target T%{customdata[7]:.0f}: %{customdata[8]:,.0f}"
        "<br>gap_norm %{customdata[9]:.4f}"
        "<br>surplus_norm %{customdata[10]:.4f}"
        "<br>source %{customdata[11]}"
        "<br>target probs %{customdata[12]}"
        "<br>coeffs %{customdata[13]}"
        "<br>follow_norm %{customdata[14]:.4f}"
        "<br>plan_lag_norm %{customdata[15]:.4f}"
        "<br>follow_rank %{customdata[16]:.0f}"
        "<extra></extra>"
    )

    for mode, part in df.groupby('dominant_mode', sort=False):
        color = mode_colors.get(str(mode), '#111827')
        part_custom = custom[part.index]
        size = np.clip(7 + np.log1p(part['cell_size'].to_numpy()) * 3.5, 7, 20)
        fig.add_trace(
            go.Scatter(
                x=part['rank_center'],
                y=part['score'],
                mode='markers',
                name=f"mode: {mode}",
                marker=dict(
                    size=size,
                    color=color,
                    opacity=0.82,
                    line=dict(width=np.where(part['is_observed_anchor'], 2.2, 0.5), color='#111827'),
                ),
                customdata=part_custom,
                hovertemplate=hover,
            ),
            row=1,
            col=1,
        )

    state_traces = [
        ('speed_norm', 'speed/T10', '#2563eb'),
        ('base_norm_speed', 'base/T10', '#60a5fa'),
        ('capacity_norm_speed', 'capacity/T10', '#0f766e'),
        ('speed_profile_norm', 'profile/T10', '#94a3b8'),
        ('speed_follow_norm', 'follow/T10', '#22c55e'),
        ('speed_committed_norm', 'committed/T10', '#f97316'),
        ('pressure', 'pressure', '#f97316'),
        ('target_affinity', 'affinity', '#8b5cf6'),
    ]
    for col, label, color in state_traces:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df['rank_center'],
                y=df[col],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=1.5),
                marker=dict(size=4),
                customdata=custom,
                hovertemplate=(
                    "rank %{customdata[0]:.0f}-%{customdata[1]:.0f}"
                    f"<br>{label}: %{{y:.4f}}"
                    "<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    if existing_mode_cols:
        labels = [
            f"{int(row.rank_start)}-{int(row.rank_end)}"
            for row in df.itertuples(index=False)
        ]
        heat_z = df[existing_mode_cols].to_numpy(dtype=float).T
        fig.add_trace(
            go.Heatmap(
                x=df['rank_center'],
                y=mode_labels,
                z=heat_z,
                colorscale='Viridis',
                zmin=0,
                zmax=1,
                colorbar=dict(title='prob'),
                customdata=np.tile(np.asarray(labels, dtype=object), (len(mode_labels), 1)),
                hovertemplate="rank %{customdata}<br>mode %{y}: %{z:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    for tier in sorted(observed_set | reward_set):
        line_color = '#111827' if tier in reward_set else '#94a3b8'
        line_dash = 'dash' if tier in reward_set else 'dot'
        fig.add_vline(
            x=tier,
            line_width=1.0,
            line_dash=line_dash,
            line_color=line_color,
            opacity=0.55,
            row='all',
            col=1,
        )

    fig.update_xaxes(type='log', title_text='Rank center (log scale)', row=3, col=1)
    fig.update_yaxes(title_text='Score', row=1, col=1)
    fig.update_yaxes(title_text='State value', row=2, col=1)
    fig.update_yaxes(title_text='Mode', row=3, col=1)
    fig.update_layout(
        title="Graph Hidden Cell State Snapshot",
        height=920,
        template='plotly_white',
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.03,
            xanchor='center',
            x=0.5,
        ),
        margin=dict(l=60, r=60, t=100, b=60),
        font=dict(size=12),
    )
    return fig
