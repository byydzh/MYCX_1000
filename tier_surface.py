# tier_surface.py
import numpy as np
import pandas as pd


def build_tier_surface(tier_data: dict, meta: dict):
    """
    从全量 tier tracker 数据构建 time × tier 观测矩阵。

    Args:
        tier_data: {tier: DataFrame | None}，每层 tracker 原始数据
        meta: 活动元数据，含 start_at (ms)

    Returns:
        pd.DataFrame: 行=hours_elapsed, 列=tier, 值=score。

    注意：
        这里刻意只保留 API 真实给出的档线，不在 tier 维度插值。
        奖励档线附近的竞争压力往往是离散断点，连续插值会抹平这些断点。
    """
    start_ts = meta.get('start_at', 0)
    available_tiers = [t for t, df in tier_data.items() if df is not None and not df.empty]

    if not available_tiers:
        return pd.DataFrame()

    frames = []
    for tier in available_tiers:
        df = tier_data[tier].copy()
        if 'ep' in df.columns:
            df = df.rename(columns={'ep': 'value'})
        if 'time' not in df.columns:
            continue
        df['hours'] = (df['time'] - start_ts) / 3600000.0
        df = df.sort_values('hours')
        df['tier'] = tier
        frames.append(df[['hours', 'tier', 'value']])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    surface = combined.pivot_table(
        index='hours', columns='tier', values='value', aggfunc='last'
    )
    surface = surface.sort_index()
    return surface


def align_observed_tier_surface(surface: pd.DataFrame, freq_hours: float | None = None):
    """
    只在时间轴对真实档线做对齐，不在档线轴插值。

    Args:
        surface: build_tier_surface 返回的真实档线观测矩阵。
        freq_hours: 可选时间网格步长；None 时沿用原始观测时间点。

    Returns:
        (aligned_surface, observed_tiers, hour_grid)
    """
    if surface.empty:
        return surface, np.array([]), np.array([])

    observed_tiers = np.array(sorted(surface.columns.tolist()), dtype=int)
    source = surface.sort_index()

    if freq_hours is not None and freq_hours > 0:
        start_h = float(source.index.min())
        end_h = float(source.index.max())
        hour_grid = np.arange(start_h, end_h + (freq_hours * 0.5), freq_hours)
        aligned = source.reindex(source.index.union(hour_grid)).sort_index()
        aligned = aligned.interpolate(method='index', limit_direction='both')
        aligned = aligned.reindex(hour_grid)
    else:
        aligned = source.copy()

    # 分数只能随时间非递减；用 cummax 抑制 tracker 抖动。
    aligned = aligned.ffill().bfill().cummax()
    return aligned, observed_tiers, aligned.index.values


def build_tier_adjacency_features(surface: pd.DataFrame):
    """
    基于真实相邻档线构建图边特征，不生成虚拟档线。

    Returns:
        pd.DataFrame columns:
            hours, tier_hi, tier_lo, score_gap, tier_gap, gap_per_rank

    说明：
        tier_hi 数值更小、竞争更靠前；tier_lo 数值更大。
        score_gap = score_hi - score_lo，用于描述相邻档线压力差。
    """
    if surface.empty:
        return pd.DataFrame(columns=[
            'hours', 'tier_hi', 'tier_lo', 'score_gap', 'tier_gap', 'gap_per_rank'
        ])

    aligned, tiers, _ = align_observed_tier_surface(surface)
    rows = []
    for h, row in aligned.iterrows():
        for tier_hi, tier_lo in zip(tiers[:-1], tiers[1:]):
            score_hi = row.get(tier_hi)
            score_lo = row.get(tier_lo)
            if not np.isfinite(score_hi) or not np.isfinite(score_lo):
                continue

            tier_gap = int(tier_lo - tier_hi)
            score_gap = float(score_hi - score_lo)
            rows.append({
                'hours': float(h),
                'tier_hi': int(tier_hi),
                'tier_lo': int(tier_lo),
                'score_gap': score_gap,
                'tier_gap': tier_gap,
                'gap_per_rank': score_gap / max(tier_gap, 1),
            })

    return pd.DataFrame(rows)


def interpolate_tier_surface(surface: pd.DataFrame):
    """
    Backward-compatible wrapper.

    旧实现会在 tier 维度做 PCHIP 插值；当前模型路线改为保留真实档线，
    所以这里仅做时间轴对齐并返回真实 observed_tiers。
    """
    return align_observed_tier_surface(surface)
