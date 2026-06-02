# app.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import streamlit as st
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import API_SOURCE_CONFIGS, DEFAULT_CONFIG, list_models, list_presets, load_preset
from data_source import create_data_source
from domain_models import EventData, EventMeta
from graph_engine import GraphModelConfig, GraphStateSpaceEngine
from math_models import SeasonalityHandler, CosineModeler
from prediction_engine import PredictionEngine
from plotly_viz import plot_graph_cell_state_plotly, plot_graph_rollout_plotly, plot_prediction_plotly

logger = logging.getLogger('predictor.app')

# ==========================================
# 0. 辅助函数 (从 main_pipeline 复用逻辑)
# ==========================================
def wrap_event_data(data_pack) -> EventData:
    """将原始数据包转换为领域对象"""
    if not data_pack: return None
    meta_obj = data_pack['meta']
    if isinstance(meta_obj, dict):
        meta_obj = EventMeta.from_dict(data_pack['event_id'], meta_obj)
        
    return EventData(
        meta=meta_obj,
        df=data_pack['dataframe'],
        scale=data_pack['scale'],
        tier=data_pack.get('tier', 1000),
    )

def calculate_derived_columns(event_data: EventData) -> EventData:
    """计算派生列：hours_elapsed, speed, norm_speed"""
    df = event_data.df
    event_data.clean_data()
    
    # 维护延迟修复
    original_start = event_data.meta.start_at
    valid_points = df[df['value'] > 0]
    if not valid_points.empty:
        first_valid_ts = valid_points.iloc[0]['time']
        # 限制修正范围在开服 24 小时内，避免误判
        if first_valid_ts > original_start and (first_valid_ts - original_start) < 86400000:
            from datetime import datetime, timezone
            # 注意：这里假设 timestamp 是 UTC 时间戳
            dt_first = datetime.fromtimestamp(first_valid_ts / 1000, timezone.utc)
            # 向下取整到小时 (或者根据实际需求调整)
            dt_corrected = dt_first.replace(minute=0, second=0, microsecond=0)
            corrected_start = int(dt_corrected.timestamp() * 1000)
            
            if corrected_start > original_start:
                # 在 Streamlit 里可以用 st.toast 或 print
                # print(f"检测到维护延迟，修正 start_at")
                event_data.meta.start_at = corrected_start

    start_ts = event_data.meta.start_at
    df['hours_elapsed'] = (df['time'] - start_ts) / 3600000.0
    
    if 'speed' not in df.columns:
        diff_val = df['value'].diff()
        diff_time = df['time'].diff() / 60000.0 
        speed = diff_val / diff_time
        df['speed'] = speed.fillna(0.0)
        df.loc[~np.isfinite(df['speed']), 'speed'] = 0.0
        df.loc[df['speed'] < 0, 'speed'] = 0.0
        
    if 'norm_speed' not in df.columns:
        if event_data.scale is None or event_data.scale <= 0:
            raise ValueError("无法获取有效的 T10 scale，请检查所选 API 数据源是否提供对应的 tier=10 / eventtop 数据。")
        df['norm_speed'] = df['speed'] / event_data.scale
        
    event_data.df = df
    return event_data


MODEL_STATE_KEY = "selected_model"
PRESET_STATE_KEY = "selected_preset"
PRESET_SIGNATURE_KEY = "preset_applied_signature"
IGNORE_IDS_TEXT_KEY = "cfg_ignore_event_ids_text"
DEFAULT_PRESET_BY_MODEL = {
    "skeleton_kf": "learned_notebook",
}

PRIMARY_CONFIG_KEYS = [
    "api_source",
    "weekend_multiplier",
    "panic_scaler",
    "panic_ease_power",
    "refit_weight_scale",
    "similar_count",
    "ratio_min",
    "ratio_max",
    "scale_min",
    "scale_max",
    "t_start_cmp",
    "t_end_cap",
    "corr_min",
    "corr_max",
    "smooth_thresh1",
    "smooth_thresh2",
    "smooth_hard_cap",
]

ADVANCED_CONFIG_KEYS = [
    "refit_min_points",
    "refit_lambda",
    "refit_start_hours",
    "refit_recent_hours",
    "refit_conf_norm_hours",
    "refit_conf_max",
    "refit_base_min_ratio",
    "refit_base_max_ratio",
    "refit_linear_bound_scale",
    "refit_linear_zero_ratio",
    "refit_quad_min_ratio",
    "refit_quad_max_ratio",
]


def _config_state_key(param_name: str) -> str:
    return f"cfg_{param_name}"


def _format_ignore_ids(ignore_ids) -> str:
    if not ignore_ids:
        return ""
    return ", ".join(str(int(item)) for item in ignore_ids)


def _parse_ignore_ids(ignore_ids_str: str):
    if not ignore_ids_str or not ignore_ids_str.strip():
        return []
    return [int(x.strip()) for x in ignore_ids_str.replace("，", ",").split(",") if x.strip()]


def _apply_preset_to_session(model_id: str, preset_name: str) -> dict:
    preset_config = load_preset(model_id, preset_name)
    for key in PRIMARY_CONFIG_KEYS + ADVANCED_CONFIG_KEYS:
        st.session_state[_config_state_key(key)] = preset_config.get(key, DEFAULT_CONFIG.get(key))

    st.session_state[IGNORE_IDS_TEXT_KEY] = _format_ignore_ids(
        preset_config.get("ignore_event_ids", DEFAULT_CONFIG.get("ignore_event_ids", []))
    )
    st.session_state[PRESET_SIGNATURE_KEY] = f"{model_id}:{preset_name}"
    return preset_config


GRAPH_STATE_MEANINGS = [
    {"state": "rank_start/rank_end", "meaning": "这个 hidden cell 覆盖的玩家排名范围。"},
    {"state": "cell_size", "meaning": "cell 代表的玩家数量；奖励线和观测锚点附近会更细。"},
    {"state": "score", "meaning": "该 cell 当前估计分数，不是单个玩家真实分数。"},
    {"state": "speed_norm", "meaning": "当前速度 / T10 scale，用来比较不同活动强度。"},
    {"state": "base_norm_speed", "meaning": "去掉时段季节性后的基础速度倾向。"},
    {"state": "capacity_norm_speed", "meaning": "理论能力上限；当前主要来自歌榜 rank shape 的平方，并用观测峰值作为下界。它只作为追线速度上限，不是日常巡航。"},
    {"state": "target_probs_label", "meaning": "该 cell 心中目标线的概率分布；例如 T1000/T2000/T500 的混合。"},
    {"state": "target_importance", "meaning": "当前活动档线形状反推的目标线后验重要性；奖励线只是先验之一。"},
    {"state": "target_rank", "meaning": "target_probs 中概率最高的目标，只是主目标，不代表唯一目标。"},
    {"state": "target_score", "meaning": "主目标线的 projected final score。"},
    {"state": "target_gap_norm", "meaning": "追上目标最终线所需的归一化速度缺口；大于 0 表示还没到目标。"},
    {"state": "target_profile_norm", "meaning": "历史进度曲线在当前时间点给出的去季节性计划速度；用于区分计划追逐和恐慌追逐。"},
    {"state": "target_follow_norm", "meaning": "当前活动内侧引导线的去季节性速度；例如 T1000 目标通常会参考 T500 的实际走向。"},
    {"state": "target_plan_lag_norm", "meaning": "相对计划累计线的落后量；用于判断是否从计划追逐转入 panic。"},
    {"state": "follow_rank", "meaning": "当前目标参考的内侧引导线 rank。"},
    {"state": "target_surplus_norm", "meaning": "高于目标最终线的归一化余量；足够大后才会进入防守逻辑。"},
    {"state": "target_boundary_proximity", "meaning": "rank 上是否贴近目标线；远离目标线时 surplus 会被视为目标达成后的松弛。"},
    {"state": "pressure", "meaning": "当前行为压力；由目标压力、邻近追赶压力或防守压力组合而来。"},
    {"state": "target_pressure", "meaning": "来自目标最终线缺口的压力。"},
    {"state": "neighbor_pressure", "meaning": "来自相邻 cell 追近、追尾或防守威胁的压力。"},
    {"state": "density_pressure", "meaning": "来自局部分数密度压缩的压力。"},
    {"state": "pressure_source", "meaning": "主压力来源：target / near_target / defend / satisfied。"},
    {"state": "speed_target_norm", "meaning": "追上实时目标预测线所需速度，截断在 cruise 与 capacity 之间；不是额外压力项。"},
    {"state": "speed_profile_norm", "meaning": "历史进度曲线的一阶速度基准。"},
    {"state": "speed_follow_norm", "meaning": "跟随内侧引导线时的速度基准。"},
    {"state": "speed_lag_norm", "meaning": "相对计划累计线落后时需要补的速度基准。"},
    {"state": "speed_panic_feature_norm", "meaning": "panic 特征，等于 lag/capacity 截断后折算到 capacity。"},
    {"state": "speed_committed_norm", "meaning": "planned_chasing 状态的速度候选。"},
    {"state": "behavior_coeffs_label", "meaning": "从当前活动已观测小时最小二乘拟合出的行为权重：cruise/profile/follow/gap/lag/panic。"},
    {"state": "speed_defend_norm", "meaning": "defending 状态的速度候选。"},
    {"state": "speed_limit_reason", "meaning": "本步最终速度主要限制来源：behavior_direct / capacity_cap。"},
    {"state": "speed_*_norm", "meaning": "速度计算拆分：先由各行为状态混合生成 preseason，再统一乘 speed_season_effect，然后只按 capacity 截断。"},
    {"state": "target_affinity", "meaning": "该 cell 对目标线的心理吸引强度。"},
    {"state": "dominant_mode", "meaning": "概率最高的行为状态，不代表 mode_probs 中其他状态为 0。"},
    {"state": "mode_*", "meaning": "idle/cruising/watching/chasing/defending/panic_rushing/dropped 的概率。"},
]


def _parse_node_selector(selector: str, cell_table: pd.DataFrame) -> pd.DataFrame:
    if cell_table is None or cell_table.empty:
        return pd.DataFrame()
    selector = (selector or "").strip()
    if not selector:
        return pd.DataFrame()

    selected_indices = set()
    tokens = [
        token.strip()
        for token in selector.replace("，", ",").replace(";", ",").split(",")
        if token.strip()
    ]
    for token in tokens:
        lowered = token.lower()
        if lowered.startswith(("cell:", "id:")):
            raw_ids = lowered.split(":", 1)[1]
            for part in raw_ids.replace("|", ",").split(","):
                part = part.strip()
                if part.isdigit():
                    matched = cell_table[cell_table["cell_id"] == int(part)]
                    selected_indices.update(matched.index.tolist())
            continue

        if "-" in token:
            left, right = token.split("-", 1)
            try:
                start = int(float(left.strip()))
                end = int(float(right.strip()))
            except ValueError:
                continue
            lo, hi = sorted((start, end))
            matched = cell_table[
                (cell_table["rank_end"] >= lo)
                & (cell_table["rank_start"] <= hi)
            ]
            selected_indices.update(matched.index.tolist())
            continue

        try:
            rank = int(float(token))
        except ValueError:
            continue
        matched = cell_table[
            (cell_table["rank_start"] <= rank)
            & (cell_table["rank_end"] >= rank)
        ]
        selected_indices.update(matched.index.tolist())

    if not selected_indices:
        return pd.DataFrame(columns=cell_table.columns)
    return cell_table.loc[sorted(selected_indices)].copy()


def _weighted_mean(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns or group.empty:
        return 0.0
    weights = group.get("cell_size", pd.Series([1] * len(group), index=group.index)).astype(float)
    values = pd.to_numeric(group[column], errors="coerce").fillna(0.0)
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 0 else 0.0


def _aggregate_target_probs(group: pd.DataFrame) -> str:
    if "target_probs" not in group.columns or group.empty:
        return ""
    weights = group.get("cell_size", pd.Series([1] * len(group), index=group.index)).astype(float)
    totals = {}
    weight_total = float(weights.sum()) or 1.0
    for idx, probs in group["target_probs"].items():
        if not isinstance(probs, dict):
            continue
        weight = float(weights.loc[idx])
        for tier, prob in probs.items():
            label = str(tier)
            if not label.startswith("T"):
                label = f"T{label}"
            totals[label] = totals.get(label, 0.0) + float(prob) * weight / weight_total
    return ", ".join(
        f"{tier}:{prob:.2f}"
        for tier, prob in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:6]
    )


def _aggregate_behavior_coeffs(group: pd.DataFrame) -> str:
    if "behavior_coeffs" not in group.columns or group.empty:
        return ""
    weights = group.get("cell_size", pd.Series([1] * len(group), index=group.index)).astype(float)
    weight_total = float(weights.sum()) or 1.0
    totals = {}
    for idx, coeffs in group["behavior_coeffs"].items():
        if not isinstance(coeffs, dict):
            continue
        weight = float(weights.loc[idx])
        for name, value in coeffs.items():
            totals[str(name)] = totals.get(str(name), 0.0) + float(value) * weight / weight_total
    return ", ".join(
        f"{name}:{value:.2f}"
        for name, value in sorted(totals.items())
        if abs(value) > 1e-3
    )


def _aggregate_category(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns or group.empty:
        return ""
    weights = group.get("cell_size", pd.Series([1] * len(group), index=group.index)).astype(float)
    totals = {}
    weight_total = float(weights.sum()) or 1.0
    for idx, value in group[column].items():
        totals[str(value)] = totals.get(str(value), 0.0) + float(weights.loc[idx]) / weight_total
    return ", ".join(
        f"{name}:{prob:.2f}"
        for name, prob in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    )


def _weighted_formula(group: pd.DataFrame, column: str, max_terms: int = 8) -> str:
    if column not in group.columns or group.empty:
        return ""
    weights = group.get("cell_size", pd.Series([1] * len(group), index=group.index)).astype(float)
    values = pd.to_numeric(group[column], errors="coerce").fillna(0.0)
    total = float(weights.sum())
    if total <= 0:
        return ""
    terms = [
        f"{float(values.loc[idx]):.4f}*{float(weights.loc[idx]):.0f}"
        for idx in group.index[:max_terms]
    ]
    if len(group) > max_terms:
        terms.append("...")
    numerator = float((values * weights).sum())
    return f"({'+'.join(terms)})/{total:.0f} = {numerator / total:.4f}"


def _node_group_summary(group: pd.DataFrame) -> pd.DataFrame:
    if group is None or group.empty:
        return pd.DataFrame(columns=["state", "value"])
    mode_cols = [col for col in group.columns if col.startswith("mode_")]
    mode_mix = ", ".join(
        f"{col.replace('mode_', '')}:{_weighted_mean(group, col):.2f}"
        for col in mode_cols
        if _weighted_mean(group, col) > 0.01
    )
    rows = [
        ("cells", str(len(group))),
        ("rank_range", f"{int(group['rank_start'].min())}-{int(group['rank_end'].max())}"),
        ("score_range", f"{group['score'].min():,.0f} - {group['score'].max():,.0f}"),
        ("speed_norm_avg", f"{_weighted_mean(group, 'speed_norm'):.4f}"),
        ("speed_norm_formula", _weighted_formula(group, "speed_norm")),
        ("base_norm_speed_avg", f"{_weighted_mean(group, 'base_norm_speed'):.4f}"),
        ("capacity_norm_speed_avg", f"{_weighted_mean(group, 'capacity_norm_speed'):.4f}"),
        ("pressure_avg", f"{_weighted_mean(group, 'pressure'):.4f}"),
        ("target_pressure_avg", f"{_weighted_mean(group, 'target_pressure'):.4f}"),
        ("neighbor_pressure_avg", f"{_weighted_mean(group, 'neighbor_pressure'):.4f}"),
        ("density_pressure_avg", f"{_weighted_mean(group, 'density_pressure'):.4f}"),
        ("pressure_source_mix", _aggregate_category(group, "pressure_source")),
        ("target_probs_avg", _aggregate_target_probs(group)),
        ("target_importance_avg", f"{_weighted_mean(group, 'target_importance'):.4f}"),
        ("target_gap_norm_avg", f"{_weighted_mean(group, 'target_gap_norm'):.4f}"),
        ("target_profile_norm_avg", f"{_weighted_mean(group, 'target_profile_norm'):.4f}"),
        ("target_follow_norm_avg", f"{_weighted_mean(group, 'target_follow_norm'):.4f}"),
        ("target_plan_lag_norm_avg", f"{_weighted_mean(group, 'target_plan_lag_norm'):.4f}"),
        ("follow_rank_mix", _aggregate_category(group, "follow_rank")),
        ("target_surplus_norm_avg", f"{_weighted_mean(group, 'target_surplus_norm'):.4f}"),
        ("target_boundary_proximity_avg", f"{_weighted_mean(group, 'target_boundary_proximity'):.4f}"),
        ("target_affinity_avg", f"{_weighted_mean(group, 'target_affinity'):.4f}"),
        ("speed_cruise_norm_avg", f"{_weighted_mean(group, 'speed_cruise_norm'):.4f}"),
        ("speed_target_norm_avg", f"{_weighted_mean(group, 'speed_target_norm'):.4f}"),
        ("speed_profile_norm_avg", f"{_weighted_mean(group, 'speed_profile_norm'):.4f}"),
        ("speed_follow_norm_avg", f"{_weighted_mean(group, 'speed_follow_norm'):.4f}"),
        ("speed_lag_norm_avg", f"{_weighted_mean(group, 'speed_lag_norm'):.4f}"),
        ("speed_panic_feature_norm_avg", f"{_weighted_mean(group, 'speed_panic_feature_norm'):.4f}"),
        ("speed_committed_norm_avg", f"{_weighted_mean(group, 'speed_committed_norm'):.4f}"),
        ("behavior_coeffs_avg", _aggregate_behavior_coeffs(group)),
        ("speed_defend_norm_avg", f"{_weighted_mean(group, 'speed_defend_norm'):.4f}"),
        ("speed_boundary_drive_avg", f"{_weighted_mean(group, 'speed_boundary_drive'):.4f}"),
        ("speed_preseason_norm_avg", f"{_weighted_mean(group, 'speed_preseason_norm'):.4f}"),
        ("speed_season_effect_avg", f"{_weighted_mean(group, 'speed_season_effect'):.4f}"),
        ("speed_desired_norm_avg", f"{_weighted_mean(group, 'speed_desired_norm'):.4f}"),
        ("speed_limit_reason_mix", _aggregate_category(group, "speed_limit_reason")),
        ("dominant_mode_mix", _aggregate_category(group, "dominant_mode")),
        ("mode_probs_avg", mode_mix),
    ]
    return pd.DataFrame(rows, columns=["state", "value"])


def _phase_name(hour: float, start_hour: float, total_hours: float) -> str:
    if float(hour) - float(start_hour) <= 24.0:
        return "start_to_24h"
    if float(total_hours) - float(hour) <= 24.0:
        return "final_24h"
    return "middle"


def _actual_curve_from_df(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["hour", "score"])
    actual = df.copy()
    if "timestamp" in actual.columns and "time" not in actual.columns:
        actual = actual.rename(columns={"timestamp": "time"})
    if "ep" in actual.columns and "value" not in actual.columns:
        actual = actual.rename(columns={"ep": "value"})
    if "time" not in actual.columns or "value" not in actual.columns:
        return pd.DataFrame(columns=["hour", "score"])
    start_at = float(meta.get("start_at", 0) or 0)
    actual["hour"] = (pd.to_numeric(actual["time"], errors="coerce") - start_at) / 3600000.0
    actual["score"] = pd.to_numeric(actual["value"], errors="coerce")
    actual = actual[np.isfinite(actual["hour"]) & np.isfinite(actual["score"])].copy()
    actual = actual.sort_values("hour").drop_duplicates("hour", keep="last")
    return actual[["hour", "score"]]


def _curve_metric_row(tier: int, phase: str, hours: np.ndarray, pred: np.ndarray, actual: np.ndarray, scale: float) -> dict:
    err = pred - actual
    abs_err = np.abs(err)
    max_idx = int(np.argmax(abs_err)) if len(abs_err) else 0

    speed_mae_norm = 0.0
    speed_signed_norm = 0.0
    speed_max_abs_norm = 0.0
    if len(hours) >= 2:
        dt_min = np.maximum(np.diff(hours) * 60.0, 1e-6)
        pred_speed = np.diff(pred) / dt_min
        actual_speed = np.diff(actual) / dt_min
        speed_err_norm = (pred_speed - actual_speed) / max(float(scale), 1e-6)
        speed_mae_norm = float(np.mean(np.abs(speed_err_norm)))
        speed_signed_norm = float(np.mean(speed_err_norm))
        speed_max_abs_norm = float(np.max(np.abs(speed_err_norm)))

    return {
        "tier": int(tier),
        "phase": phase,
        "points": int(len(hours)),
        "score_mae": float(np.mean(abs_err)) if len(abs_err) else 0.0,
        "score_rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.0,
        "score_signed": float(np.mean(err)) if len(err) else 0.0,
        "score_max_abs": float(abs_err[max_idx]) if len(abs_err) else 0.0,
        "score_max_hour": float(hours[max_idx]) if len(hours) else 0.0,
        "score_max_pred": float(pred[max_idx]) if len(pred) else 0.0,
        "score_max_actual": float(actual[max_idx]) if len(actual) else 0.0,
        "speed_mae_norm": speed_mae_norm,
        "speed_signed_norm": speed_signed_norm,
        "speed_max_abs_norm": speed_max_abs_norm,
    }


def _graph_curve_error_diagnostics(
    trajectories: dict,
    actual_tier_data: dict,
    meta: dict,
    start_hour: float,
    total_hours: float,
    scale: float,
) -> pd.DataFrame:
    rows = []
    if not trajectories or not actual_tier_data:
        return pd.DataFrame()

    for tier, nodes in sorted(trajectories.items()):
        if not nodes or int(tier) not in actual_tier_data:
            continue
        actual_curve = _actual_curve_from_df(actual_tier_data.get(int(tier)), meta)
        actual_curve = actual_curve[
            (actual_curve["hour"] >= float(start_hour))
            & (actual_curve["hour"] <= float(total_hours))
        ].copy()
        if len(actual_curve) < 2:
            continue

        pred_hours = np.linspace(float(start_hour), float(total_hours), len(nodes))
        pred_scores = np.asarray([float(node.score) for node in nodes], dtype=float)
        actual_hours = actual_curve["hour"].to_numpy(dtype=float)
        actual_scores = actual_curve["score"].to_numpy(dtype=float)
        pred_at_actual = np.interp(actual_hours, pred_hours, pred_scores)

        rows.append(_curve_metric_row(int(tier), "all", actual_hours, pred_at_actual, actual_scores, scale))

        phases = np.asarray([_phase_name(hour, start_hour, total_hours) for hour in actual_hours], dtype=object)
        for phase in ("start_to_24h", "middle", "final_24h"):
            mask = phases == phase
            if int(mask.sum()) >= 2:
                rows.append(
                    _curve_metric_row(
                        int(tier),
                        phase,
                        actual_hours[mask],
                        pred_at_actual[mask],
                        actual_scores[mask],
                        scale,
                    )
                )

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(["phase", "score_mae"], ascending=[True, False]).reset_index(drop=True)

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="预测面板", page_icon="🐱", layout="wide")
st.title("🐱 实时预测面板")

# ==========================================
# 2. 初始化 Session State
# ==========================================
if 'img_bytes' not in st.session_state:
    st.session_state['img_bytes'] = None
if 'last_update_str' not in st.session_state:
    st.session_state['last_update_str'] = "暂无数据"
if 'has_initialized' not in st.session_state:
    st.session_state['has_initialized'] = False
if PRESET_SIGNATURE_KEY not in st.session_state:
    st.session_state[PRESET_SIGNATURE_KEY] = None
if 'current_score' not in st.session_state:
    st.session_state['current_score'] = None
if 'predicted_score' not in st.session_state:
    st.session_state['predicted_score'] = None
if 'actual_score' not in st.session_state:
    st.session_state['actual_score'] = None
if 'current_event_id' not in st.session_state:
    st.session_state['current_event_id'] = None
if 'is_debug_mode' not in st.session_state:
    st.session_state['is_debug_mode'] = False
if 'tier_results' not in st.session_state:
    st.session_state['tier_results'] = {}
if 'tier_targets' not in st.session_state:
    st.session_state['tier_targets'] = {}
if 'tier_errors' not in st.session_state:
    st.session_state['tier_errors'] = {}
if 'selected_tiers' not in st.session_state:
    st.session_state['selected_tiers'] = [500, 1000, 1500, 2000]
if 'graph_fig' not in st.session_state:
    st.session_state['graph_fig'] = None
if 'graph_cell_fig' not in st.session_state:
    st.session_state['graph_cell_fig'] = None
if 'graph_cell_table' not in st.session_state:
    st.session_state['graph_cell_table'] = None
if 'graph_cell_frames' not in st.session_state:
    st.session_state['graph_cell_frames'] = None
if 'graph_curve_errors' not in st.session_state:
    st.session_state['graph_curve_errors'] = None
if 'graph_summary' not in st.session_state:
    st.session_state['graph_summary'] = None
if 'graph_error' not in st.session_state:
    st.session_state['graph_error'] = None
if 'run_timings' not in st.session_state:
    st.session_state['run_timings'] = {}

# ==========================================
# 3. 侧边栏控制
# ==========================================
st.sidebar.header("控制台 🎮")

available_models = list_models()
if not available_models:
    available_models = [{
        "id": "skeleton_kf",
        "name": "Skeleton + Kalman Filter",
        "description": "Fallback model registry entry.",
    }]

model_options = [item["id"] for item in available_models]
model_lookup = {item["id"]: item for item in available_models}
if st.session_state.get(MODEL_STATE_KEY) not in model_options:
    st.session_state[MODEL_STATE_KEY] = model_options[0]

selected_model = st.sidebar.selectbox(
    "预测模型",
    options=model_options,
    format_func=lambda model_id: model_lookup[model_id].get("name", model_id),
    key=MODEL_STATE_KEY,
    disabled=len(model_options) <= 1,
)
selected_model_meta = model_lookup[selected_model]
if selected_model_meta.get("description"):
    st.sidebar.caption(selected_model_meta["description"])

available_presets = list_presets(selected_model)
if not available_presets:
    available_presets = [{
        "id": "default",
        "name": "default",
        "description": "Fallback to DEFAULT_CONFIG",
    }]

preset_options = [item["id"] for item in available_presets]
preset_lookup = {item["id"]: item for item in available_presets}
preferred_preset = DEFAULT_PRESET_BY_MODEL.get(selected_model, preset_options[0])
if preferred_preset not in preset_options:
    preferred_preset = preset_options[0]
if st.session_state.get(PRESET_STATE_KEY) not in preset_options:
    st.session_state[PRESET_STATE_KEY] = preferred_preset

selected_preset = st.sidebar.selectbox(
    "配置预设",
    options=preset_options,
    format_func=lambda preset_id: preset_lookup[preset_id].get("name", preset_id),
    key=PRESET_STATE_KEY,
)

selected_preset_meta = preset_lookup[selected_preset]
if selected_preset_meta.get("description"):
    st.sidebar.caption(selected_preset_meta["description"])

current_signature = f"{selected_model}:{selected_preset}"
if st.session_state.get(PRESET_SIGNATURE_KEY) != current_signature:
    _apply_preset_to_session(selected_model, selected_preset)

ALL_TIERS = [500, 1000, 1500, 2000]
st.session_state['selected_tiers'] = [
    tier for tier in st.session_state.get('selected_tiers', ALL_TIERS)
    if tier in ALL_TIERS
] or ALL_TIERS
selected_tiers = st.sidebar.multiselect(
    "预测榜线",
    options=ALL_TIERS,
    format_func=lambda t: f"T{t}",
    key='selected_tiers',
)

with st.sidebar.expander("Graph 实验", expanded=False):
    enable_graph_panel = st.checkbox(
        "生成 Graph 状态空间实验图",
        value=False,
        help="额外获取全量真实档线，绘制 graph rollout 与 pressure/affinity/mode 诊断；不影响正式 Skeleton+KF 预测。",
    )
    graph_max_tiers = st.number_input(
        "最多显示档线数",
        min_value=2,
        max_value=16,
        value=8,
        step=1,
        disabled=not enable_graph_panel,
    )

manual_btn = st.sidebar.button("⚡ 立即运行预测", type="primary")

st.sidebar.markdown("---")
with st.sidebar.expander("参数设置", expanded=False):
    st.caption("切换预设会重置下面控件；手动修改只影响当前会话，不会回写 JSON。")

    api_source_keys = list(API_SOURCE_CONFIGS.keys())
    selected_api_source_key = st.session_state.get(
        _config_state_key('api_source'),
        DEFAULT_CONFIG.get('api_source', api_source_keys[0])
    )
    if selected_api_source_key not in api_source_keys:
        selected_api_source_key = api_source_keys[0]
    selected_api_source = st.selectbox(
        "API 数据源",
        options=api_source_keys,
        index=api_source_keys.index(selected_api_source_key),
        format_func=lambda key: API_SOURCE_CONFIGS[key].get('label', key),
        help="切换活动元数据与榜线数据接口；T10 scale 会优先使用所选数据源，若历史 tier=10 缺失则自动回退 Bestdori eventtop。",
        key=_config_state_key('api_source'),
    )
    
    # 模型参数
    st.markdown("**模型参数**")
    weekend_mult = st.slider(
        "周末增强系数", 0.8, 1.5,
        value=float(st.session_state.get(_config_state_key('weekend_multiplier'), DEFAULT_CONFIG.get('weekend_multiplier', 1.0))),
        step=0.05,
        key=_config_state_key('weekend_multiplier')
    )
    panic_scaler = st.slider(
        "恐慌期最小加速倍数", 1.0, 3.0,
        value=float(st.session_state.get(_config_state_key('panic_scaler'), DEFAULT_CONFIG.get('panic_scaler', 1.1))),
        step=0.05,
        key=_config_state_key('panic_scaler')
    )
    panic_ease_power = st.slider(
        "恐慌期缓动指数", 0.1, 5.0,
        value=float(st.session_state.get(_config_state_key('panic_ease_power'), DEFAULT_CONFIG.get('panic_ease_power', 1.0))),
        step=0.1,
        key=_config_state_key('panic_ease_power')
    )
    refit_weight_scale = st.number_input(
        "拟合权重系数 (Log Scale)", 1.0, 100.0,
        value=float(st.session_state.get(_config_state_key('refit_weight_scale'), DEFAULT_CONFIG.get('refit_weight_scale', 10.0))),
        step=1.0,
        key=_config_state_key('refit_weight_scale')
    )
    similar_count = st.number_input(
        "参考历史活动数", 1, 10,
        value=int(st.session_state.get(_config_state_key('similar_count'), DEFAULT_CONFIG.get('similar_count', 5))),
        step=1,
        key=_config_state_key('similar_count')
    )
    
    ignore_ids_str = st.text_input(
        "忽略的活动 ID (逗号分隔)",
        value=st.session_state.get(IGNORE_IDS_TEXT_KEY, _format_ignore_ids(DEFAULT_CONFIG.get('ignore_event_ids', []))),
        help="例如: 297, 298",
        key=IGNORE_IDS_TEXT_KEY
    )
    ignore_ids = []
    if ignore_ids_str.strip():
        try:
            ignore_ids = _parse_ignore_ids(ignore_ids_str)
        except ValueError:
            st.sidebar.error("忽略 ID 格式错误，请使用逗号分隔的数字")

    # 阈值与限制
    st.markdown("**阈值与限制**")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        ratio_min = st.number_input(
            "Ratio Min",
            value=float(st.session_state.get(_config_state_key('ratio_min'), DEFAULT_CONFIG.get('ratio_min', 0.25))),
            step=0.05,
            key=_config_state_key('ratio_min')
        )
        scale_min = st.number_input(
            "Scale Min",
            value=float(st.session_state.get(_config_state_key('scale_min'), DEFAULT_CONFIG.get('scale_min', 0.5))),
            step=0.1,
            key=_config_state_key('scale_min')
        )
        t_start_cmp = st.number_input(
            "对比窗口起始",
            value=float(st.session_state.get(_config_state_key('t_start_cmp'), DEFAULT_CONFIG.get('t_start_cmp', 6.0))),
            step=0.5,
            key=_config_state_key('t_start_cmp')
        )
    with col_p2:
        ratio_max = st.number_input(
            "Ratio Max",
            value=float(st.session_state.get(_config_state_key('ratio_max'), DEFAULT_CONFIG.get('ratio_max', 4.0))),
            step=0.1,
            key=_config_state_key('ratio_max')
        )
        scale_max = st.number_input(
            "Scale Max",
            value=float(st.session_state.get(_config_state_key('scale_max'), DEFAULT_CONFIG.get('scale_max', 2.0))),
            step=0.1,
            key=_config_state_key('scale_max')
        )
        t_end_cap = st.number_input(
            "窗口结束上限",
            value=float(st.session_state.get(_config_state_key('t_end_cap'), DEFAULT_CONFIG.get('t_end_cap', 72.0))),
            step=1.0,
            key=_config_state_key('t_end_cap')
        )

    # 回测与平滑
    st.markdown("**回测与平滑**")
    corr_min = st.number_input(
        "24h 修正下限",
        value=float(st.session_state.get(_config_state_key('corr_min'), DEFAULT_CONFIG.get('corr_min', 0.6))),
        step=0.05,
        key=_config_state_key('corr_min')
    )
    corr_max = st.number_input(
        "24h 修正上限",
        value=float(st.session_state.get(_config_state_key('corr_max'), DEFAULT_CONFIG.get('corr_max', 1.6))),
        step=0.05,
        key=_config_state_key('corr_max')
    )
    smooth_thresh1 = st.number_input(
        "平滑阈值 1", 0.0, 1.0,
        value=float(st.session_state.get(_config_state_key('smooth_thresh1'), DEFAULT_CONFIG.get('smooth_thresh1', 0.5))),
        step=0.01,
        key=_config_state_key('smooth_thresh1')
    )
    smooth_thresh2 = st.number_input(
        "平滑阈值 2", 0.0, 1.0,
        value=float(st.session_state.get(_config_state_key('smooth_thresh2'), DEFAULT_CONFIG.get('smooth_thresh2', 0.65))),
        step=0.01,
        key=_config_state_key('smooth_thresh2')
    )
    smooth_hard_cap = st.number_input(
        "绝对硬顶", 0.0, 1.0,
        value=float(st.session_state.get(_config_state_key('smooth_hard_cap'), DEFAULT_CONFIG.get('smooth_hard_cap', 0.8))),
        step=0.01,
        key=_config_state_key('smooth_hard_cap')
    )

with st.sidebar.expander("高级参数", expanded=False):
    st.caption("当前主要开放在线重拟合相关参数，便于 preset 覆盖和精细微调。")

    refit_min_points = st.number_input(
        "Refit 最少点数", 1, 100,
        value=int(st.session_state.get(_config_state_key('refit_min_points'), DEFAULT_CONFIG.get('refit_min_points', 10))),
        step=1,
        key=_config_state_key('refit_min_points')
    )
    refit_lambda = st.number_input(
        "Refit 正则强度",
        value=float(st.session_state.get(_config_state_key('refit_lambda'), DEFAULT_CONFIG.get('refit_lambda', 0.3))),
        step=0.05,
        key=_config_state_key('refit_lambda')
    )

    col_refit_1, col_refit_2 = st.columns(2)
    with col_refit_1:
        refit_start_hours = st.number_input(
            "Refit 起始小时",
            value=float(st.session_state.get(_config_state_key('refit_start_hours'), DEFAULT_CONFIG.get('refit_start_hours', 6.0))),
            step=0.5,
            key=_config_state_key('refit_start_hours')
        )
        refit_conf_norm_hours = st.number_input(
            "置信度归一化时长",
            value=float(st.session_state.get(_config_state_key('refit_conf_norm_hours'), DEFAULT_CONFIG.get('refit_conf_norm_hours', 72.0))),
            step=1.0,
            key=_config_state_key('refit_conf_norm_hours')
        )
        refit_base_min_ratio = st.number_input(
            "Base 下界比例",
            value=float(st.session_state.get(_config_state_key('refit_base_min_ratio'), DEFAULT_CONFIG.get('refit_base_min_ratio', 0.6))),
            step=0.05,
            key=_config_state_key('refit_base_min_ratio')
        )
        refit_linear_bound_scale = st.number_input(
            "A 边界缩放",
            value=float(st.session_state.get(_config_state_key('refit_linear_bound_scale'), DEFAULT_CONFIG.get('refit_linear_bound_scale', 2.0))),
            step=0.05,
            key=_config_state_key('refit_linear_bound_scale')
        )
        refit_quad_min_ratio = st.number_input(
            "B 下界比例",
            value=float(st.session_state.get(_config_state_key('refit_quad_min_ratio'), DEFAULT_CONFIG.get('refit_quad_min_ratio', 0.1))),
            step=0.05,
            key=_config_state_key('refit_quad_min_ratio')
        )
    with col_refit_2:
        refit_recent_hours = st.number_input(
            "Refit 最近窗口",
            value=float(st.session_state.get(_config_state_key('refit_recent_hours'), DEFAULT_CONFIG.get('refit_recent_hours', 48.0))),
            step=1.0,
            key=_config_state_key('refit_recent_hours')
        )
        refit_conf_max = st.number_input(
            "Refit 最大权重",
            value=float(st.session_state.get(_config_state_key('refit_conf_max'), DEFAULT_CONFIG.get('refit_conf_max', 0.35))),
            step=0.01,
            key=_config_state_key('refit_conf_max')
        )
        refit_base_max_ratio = st.number_input(
            "Base 上界比例",
            value=float(st.session_state.get(_config_state_key('refit_base_max_ratio'), DEFAULT_CONFIG.get('refit_base_max_ratio', 1.6))),
            step=0.05,
            key=_config_state_key('refit_base_max_ratio')
        )
        refit_linear_zero_ratio = st.number_input(
            "A 向零收缩比例",
            value=float(st.session_state.get(_config_state_key('refit_linear_zero_ratio'), DEFAULT_CONFIG.get('refit_linear_zero_ratio', 0.25))),
            step=0.05,
            key=_config_state_key('refit_linear_zero_ratio')
        )
        refit_quad_max_ratio = st.number_input(
            "B 上界比例",
            value=float(st.session_state.get(_config_state_key('refit_quad_max_ratio'), DEFAULT_CONFIG.get('refit_quad_max_ratio', 2.0))),
            step=0.05,
            key=_config_state_key('refit_quad_max_ratio')
        )

# 调试模式
st.sidebar.markdown("---")
enable_debug = st.sidebar.checkbox("启用调试模式", value=False)
if enable_debug:
    debug_event_id = st.sidebar.number_input("目标 Event ID", min_value=1, value=312, step=1)
    
    # 冻结时间控制
    use_max_time = st.sidebar.checkbox("使用最大观测时间 (禁用冻结)", value=False)
    if not use_max_time:
        # 支持输入相对小时数 (float) 或 绝对时间 (UTC+8 string)
        debug_freeze_str = st.sidebar.text_input(
            "冻结时间 (相对小时 或 UTC+8时间)",
            value="60.0",
            help="支持输入数字(如 60.0)表示相对小时，或日期时间(如 2025-11-22 12:00:00)"
        )
    else:
        debug_freeze_str = None
    
    # 🔮 假设性干预
    manual_points_raw = []
    with st.sidebar.expander("假设性干预", expanded=False):
        st.caption("输入未来的假设性数据点，每行一个：`YYYY-MM-DD HH:MM 分数`, UTC+8 时间")
        st.caption("例如：`2025-11-23 18:00 1000000`")
        
        manual_text = st.text_area(
            "输入框",
            value="",
            height=100,
            placeholder="2025-11-23 18:00 1000000\n2025-11-24 09:00 1500000"
        )
        
        if manual_text.strip():
            for line in manual_text.strip().split('\n'):
                line = line.strip()
                if not line: continue
                # 尝试解析：最后一部分是分数，前面是时间
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        score_val = float(parts[-1])
                        time_str = " ".join(parts[:-1])
                        manual_points_raw.append({'time_str': time_str, 'score': score_val})
                    except ValueError:
                        st.error(f"无法解析行: {line}")

else:
    debug_event_id = None
    debug_freeze_str = None
    use_max_time = True
    manual_points_raw = []

# ==========================================
# 4. 核心逻辑
# ==========================================
should_run = False
trigger_reason = ""

if manual_btn:
    should_run = True
    trigger_reason = "手动触发"
elif not st.session_state['has_initialized']:
    should_run = True
    trigger_reason = "首次加载"
    st.session_state['has_initialized'] = True

if should_run:
    run_timings = {}
    timing_state = {'last': time.perf_counter(), 'start': time.perf_counter()}

    def _mark_timing(label: str):
        now = time.perf_counter()
        run_timings[label] = now - timing_state['last']
        timing_state['last'] = now

    # 构造本次运行的配置字典
    current_config = load_preset(selected_model, selected_preset)
    current_config.update({
        'api_source': selected_api_source,
        'weekend_multiplier': weekend_mult,
        'panic_scaler': panic_scaler,
        'panic_ease_power': panic_ease_power,
        'refit_weight_scale': refit_weight_scale,
        'similar_count': int(similar_count),
        'ignore_event_ids': ignore_ids,
        'ratio_min': ratio_min, 'ratio_max': ratio_max,
        'scale_min': scale_min, 'scale_max': scale_max,
        't_start_cmp': t_start_cmp, 't_end_cap': t_end_cap,
        'corr_min': corr_min, 'corr_max': corr_max,
        'smooth_thresh1': smooth_thresh1, 'smooth_thresh2': smooth_thresh2,
        'smooth_hard_cap': smooth_hard_cap,
        'refit_min_points': int(refit_min_points),
        'refit_lambda': refit_lambda,
        'refit_start_hours': refit_start_hours,
        'refit_recent_hours': refit_recent_hours,
        'refit_conf_norm_hours': refit_conf_norm_hours,
        'refit_conf_max': refit_conf_max,
        'refit_base_min_ratio': refit_base_min_ratio,
        'refit_base_max_ratio': refit_base_max_ratio,
        'refit_linear_bound_scale': refit_linear_bound_scale,
        'refit_linear_zero_ratio': refit_linear_zero_ratio,
        'refit_quad_min_ratio': refit_quad_min_ratio,
        'refit_quad_max_ratio': refit_quad_max_ratio,
    })

    with st.spinner(f"🐱 ({trigger_reason}) 正在计算中..."):
        ds = create_data_source(current_config.get('api_source'))
        try:
            # 1. 获取目标 ID
            if enable_debug and debug_event_id:
                target_eid = int(debug_event_id)
            else:
                target_eid = ds.get_current_event_id()

            if not target_eid:
                st.error("无法获取当前活动 ID，请检查网络或手动指定。")
            elif not selected_tiers:
                st.warning("请至少选择一个榜线。")
            else:
                # --- 先获取 meta（全局共享）---
                meta_raw = ds.fetch_event_meta(target_eid)
                if not meta_raw:
                    st.error(f"无法获取活动 {target_eid} 的元数据。")
                else:
                    meta_obj = EventMeta.from_dict(target_eid, meta_raw)
                    start_ts = meta_obj.start_at
                    tz_utc8 = timezone(timedelta(hours=8))

                    # --- 时间解析与转换（共享）---
                    target_debug_h = None
                    if enable_debug and not use_max_time and debug_freeze_str:
                        try:
                            target_debug_h = float(debug_freeze_str)
                        except ValueError:
                            try:
                                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
                                    try:
                                        dt_freeze = datetime.strptime(debug_freeze_str, fmt)
                                        break
                                    except ValueError:
                                        continue
                                else:
                                    raise ValueError("无法识别的时间格式")
                                dt_freeze = dt_freeze.replace(tzinfo=tz_utc8)
                                ts_freeze = dt_freeze.timestamp() * 1000
                                target_debug_h = (ts_freeze - start_ts) / 3600000.0
                            except Exception as e:
                                st.error(f"冻结时间解析失败: {e}")
                                target_debug_h = None
                        if target_debug_h is not None and target_debug_h < 0:
                            target_debug_h = 0.0

                    # --- 人工干预点解析（共享）---
                    manual_points = []
                    if manual_points_raw:
                        for mp in manual_points_raw:
                            try:
                                t_str = mp['time_str']
                                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
                                    try:
                                        dt_mp = datetime.strptime(t_str, fmt)
                                        break
                                    except ValueError:
                                        continue
                                else:
                                    continue
                                dt_mp = dt_mp.replace(tzinfo=tz_utc8)
                                ts_mp = dt_mp.timestamp() * 1000
                                h_mp = (ts_mp - start_ts) / 3600000.0
                                manual_points.append({'hours': h_mp, 'score': mp['score']})
                            except Exception:
                                pass

                    # --- 初始化引擎组件（所有层级共享）---
                    seasonality = SeasonalityHandler(
                        weekend_multiplier=float(current_config.get('weekend_multiplier', weekend_mult)),
                        panic_scaler=float(current_config.get('panic_scaler', panic_scaler)),
                        panic_ease_power=float(current_config.get('panic_ease_power', panic_ease_power))
                    )
                    modeler = CosineModeler()
                    engine = PredictionEngine(seasonality, modeler, config=current_config)

                    # --- 预取共享资源 ---
                    event_type = meta_raw.get('event_type', 'unknown')
                    target_debug_limit_ts = None
                    if target_debug_h is not None:
                        target_debug_limit_ts = int(start_ts + target_debug_h * 3600000)
                    scale_val = ds.fetch_top10_max_speed(target_eid, debug_limit_ts=target_debug_limit_ts)
                    ds.fetch_events_index()  # 预热缓存
                    _mark_timing("共享资源: meta/scale/index")

                    # --- 并行获取各线数据 ---
                    def _fetch_one_tier(tier):
                        tds = create_data_source(current_config.get('api_source'))
                        try:
                            tp = tds.fetch_event_data_pack(target_eid, tier=tier, meta=meta_raw, scale=scale_val)
                            if not tp:
                                return tier, None, None, "数据不可用"
                            sp = tds.find_similar_events(
                                target_eid, event_type,
                                count=int(current_config.get('similar_count', similar_count)),
                                ignore_ids=current_config.get('ignore_event_ids', ignore_ids),
                                tier=tier,
                            )
                            return tier, tp, sp, None
                        except Exception as exc:
                            return tier, None, None, str(exc)
                        finally:
                            tds.close()

                    tier_packs = {}
                    tier_similar = {}
                    tier_errors_dict = {}
                    tier_warnings_dict = {}
                    max_workers = min(len(selected_tiers), 4)
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {executor.submit(_fetch_one_tier, t): t for t in selected_tiers}
                        for f in as_completed(futures):
                            tier, tp, sp, err = f.result()
                            if err:
                                tier_errors_dict[tier] = err
                            else:
                                tier_packs[tier] = tp
                                tier_similar[tier] = sp
                    _mark_timing("API: 当前档线+相似活动")

                    # --- 顺序执行预测 ---
                    tier_data_dict = {}
                    tier_results_dict = {}

                    for tier in selected_tiers:
                        if tier not in tier_packs:
                            if tier not in tier_errors_dict:
                                tier_errors_dict[tier] = "数据不可用"
                            continue

                        tier_target = wrap_event_data(tier_packs[tier])
                        try:
                            tier_target = calculate_derived_columns(tier_target)
                        except Exception:
                            tier_errors_dict[tier] = "数据异常"
                            continue

                        tier_target.full_df = tier_target.df.copy()

                        if target_debug_h is not None:
                            limit_ts = tier_target.meta.start_at + (target_debug_h * 3600 * 1000)
                            tier_target.df = tier_target.df[tier_target.df['time'] <= limit_ts].copy()

                        history_events = []
                        for pack in tier_similar.get(tier, []):
                            if pack.get('is_interpolated_tier'):
                                source_tiers = pack.get('interpolated_from_tiers') or []
                                if source_tiers:
                                    tier_warnings_dict[tier] = (
                                        f"历史先验缺失，使用 T{source_tiers[0]}/T{source_tiers[1]} "
                                        f"相邻榜线合成 baseline"
                                    )
                            h_data = wrap_event_data(pack)
                            try:
                                h_data = calculate_derived_columns(h_data)
                                history_events.append(h_data)
                            except Exception:
                                pass

                        try:
                            result = engine.predict(
                                tier_target, history_events,
                                debug_hours=target_debug_h,
                                manual_points=manual_points
                            )
                        except Exception as e:
                            tier_errors_dict[tier] = f"预测失败: {e}"
                            continue

                        tier_data_dict[tier] = tier_target
                        tier_results_dict[tier] = result
                    _mark_timing("模型预测")

                    # --- 绘图 ---
                    if tier_data_dict:
                        fig = plot_prediction_plotly(
                            tier_data_dict,
                            tier_results_dict,
                            debug_hours=target_debug_h,
                            manual_points=manual_points,
                        )
                        st.session_state['img_bytes'] = fig
                    else:
                        st.session_state['img_bytes'] = None
                    _mark_timing("主图绘制")

                    # --- Graph 状态空间实验图（独立于正式预测）---
                    st.session_state['graph_fig'] = None
                    st.session_state['graph_cell_fig'] = None
                    st.session_state['graph_cell_table'] = None
                    st.session_state['graph_cell_frames'] = None
                    st.session_state['graph_curve_errors'] = None
                    st.session_state['graph_summary'] = None
                    st.session_state['graph_error'] = None
                    if enable_graph_panel:
                        try:
                            graph_rewards = ds.fetch_event_rewards(target_eid)
                            if not graph_rewards.get('target_tiers'):
                                reward_ds = create_data_source('bestdori')
                                try:
                                    graph_rewards = reward_ds.fetch_event_rewards(target_eid)
                                finally:
                                    reward_ds.close()
                            _mark_timing("Graph: 奖励线")

                            graph_tier_data = ds.fetch_all_tier_data(target_eid)
                            _mark_timing("Graph: 全量档线")
                            total_hours = (meta_raw['end_at'] - meta_raw['start_at']) / 3600000.0

                            observed_until_hour = target_debug_h
                            if observed_until_hour is None:
                                observed_hours = []
                                for graph_df in graph_tier_data.values():
                                    if graph_df is None or graph_df.empty or 'time' not in graph_df.columns:
                                        continue
                                    observed_hours.append(float(((graph_df['time'] - meta_raw['start_at']) / 3600000.0).max()))
                                observed_until_hour = max(observed_hours) if observed_hours else 0.0

                            graph_engine = GraphStateSpaceEngine(
                                seasonality,
                                GraphModelConfig(),
                            )
                            graph_observed_states, graph_observed_hours = graph_engine.replay_observed_states(
                                graph_tier_data,
                                meta_raw,
                                graph_rewards.get('target_tiers', []),
                                scale=scale_val,
                                observed_until_hour=observed_until_hour,
                                align_freq_hours=1.0,
                            )
                            _mark_timing("Graph: filtering")
                            graph_nodes = graph_engine.current_nodes()
                            graph_start_hour = float(graph_engine._last_hour)
                            graph_cell_rows = graph_engine.cell_snapshot()
                            graph_filter_metrics = graph_engine.filter_metrics()
                            st.session_state['graph_cell_table'] = pd.DataFrame(graph_cell_rows)
                            st.session_state['graph_cell_fig'] = plot_graph_cell_state_plotly(
                                graph_cell_rows,
                                reward_tiers=graph_rewards.get('target_tiers', []),
                                observed_tiers=graph_engine._observed_tiers,
                            )
                            _mark_timing("Graph: cell图")
                            graph_hours_forward = max(total_hours - graph_start_hour, 0.0)
                            graph_cell_frames = graph_engine.cell_rollout_snapshots(
                                start_hour=graph_start_hour,
                                hours_forward=graph_hours_forward,
                                scale=scale_val,
                                total_hours=total_hours,
                                reward_tiers=graph_rewards.get('target_tiers', []),
                            )
                            st.session_state['graph_cell_frames'] = graph_cell_frames
                            _mark_timing("Graph: cell帧")
                            graph_traj = {}
                            for frame in graph_cell_frames:
                                for tier, node in (frame.get('nodes') or {}).items():
                                    graph_traj.setdefault(int(tier), []).append(node)
                            _mark_timing("Graph: rollout")
                            graph_curve_errors = _graph_curve_error_diagnostics(
                                graph_traj,
                                graph_tier_data,
                                meta_raw,
                                graph_start_hour,
                                total_hours,
                                scale_val,
                            )
                            st.session_state['graph_curve_errors'] = graph_curve_errors
                            _mark_timing("Graph: 曲线误差")
                            st.session_state['graph_fig'] = plot_graph_rollout_plotly(
                                graph_traj,
                                start_hour=graph_start_hour,
                                total_hours=total_hours,
                                start_ts=meta_raw['start_at'],
                                scale=scale_val,
                                reward_tiers=graph_rewards.get('target_tiers', []),
                                actual_tier_data=graph_tier_data,
                                observed_state_trajectories=graph_observed_states,
                                observed_state_hours=graph_observed_hours,
                                max_tiers=int(graph_max_tiers),
                            )
                            _mark_timing("Graph: 绘图")
                            st.session_state['graph_summary'] = {
                                'nodes': len(graph_nodes),
                                'cells': len(graph_cell_rows),
                                'reward_tiers': graph_rewards.get('target_tiers', []),
                                'start_hour': graph_start_hour,
                                'hours_forward': graph_hours_forward,
                                'filter_metrics': graph_filter_metrics,
                            }
                        except Exception as graph_exc:
                            st.session_state['graph_error'] = f"{graph_exc}\n{traceback.format_exc()}"

                    st.session_state['tier_results'] = tier_results_dict
                    st.session_state['tier_targets'] = tier_data_dict
                    st.session_state['tier_errors'] = tier_errors_dict
                    st.session_state['tier_warnings'] = tier_warnings_dict
                    st.session_state['current_event_id'] = target_eid
                    st.session_state['is_debug_mode'] = enable_debug
                    run_timings["总计"] = time.perf_counter() - timing_state['start']
                    st.session_state['run_timings'] = run_timings
                    logger.warning(
                        "[perf] " + " | ".join(
                            f"{label}={elapsed:.2f}s" for label, elapsed in run_timings.items()
                        )
                    )

                    # 更新时间
                    beijing_tz = timezone(timedelta(hours=8))
                    st.session_state['last_update_str'] = datetime.now(beijing_tz).strftime('%H:%M:%S')

                    if manual_btn:
                        passed = len(tier_results_dict)
                        st.success(f"预测完成！Event {target_eid} | {passed}/{len(selected_tiers)} 线成功")

        except Exception as e:
            st.error(f"运行出错: {str(e)}")
            st.code(traceback.format_exc())
        finally:
            ds.close()

# ==========================================
# 5. 结果展示
# ==========================================
col_img, col_info = st.columns([3, 1])

with col_img:
    if st.session_state['img_bytes']:
        tier_results = st.session_state.get('tier_results', {})
        tier_targets = st.session_state.get('tier_targets', {})
        tier_errors = st.session_state.get('tier_errors', {})
        tier_warnings = st.session_state.get('tier_warnings', {})

        score_parts = []
        for tier in sorted(tier_results.keys()):
            r = tier_results[tier]
            t = tier_targets.get(tier)
            cur = int(t.df['value'].max()) if t is not None and not t.df.empty else 0
            score_parts.append(f"T{tier}: 当前 **{cur:,}** → 预测 **{int(r.final_score):,}**")

        for tier in sorted(tier_errors.keys()):
            score_parts.append(f"T{tier}: ⚠ {tier_errors[tier]}")

        for tier in sorted(tier_warnings.keys()):
            score_parts.append(f"T{tier}: ⚠ {tier_warnings[tier]}")

        if score_parts:
            st.markdown("  |  ".join(score_parts))

        st.plotly_chart(
            st.session_state['img_bytes'],
            width='stretch',
            config={
                'displaylogo': False,
                'modeBarButtonsToRemove': [
                    'select2d',
                    'lasso2d',
                    'autoScale2d',
                    'toggleSpikelines',
                ],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'event_prediction',
                    'height': 900,
                    'width': 1200,
                    'scale': 2,
                },
            },
        )
        st.caption(f"更新于: {st.session_state['last_update_str']}")

        if st.session_state.get('graph_error'):
            with st.expander("Graph 实验错误", expanded=False):
                st.code(st.session_state['graph_error'])

        if st.session_state.get('graph_fig') is not None:
            summary = st.session_state.get('graph_summary') or {}
            st.markdown("### Graph 状态空间实验")
            st.caption(
                "节点数: {nodes} | 奖励目标线: {reward_tiers} | 起点: +{start_hour:.1f}h | rollout: {hours_forward:.1f}h".format(
                    nodes=summary.get('nodes', 0),
                    reward_tiers=summary.get('reward_tiers', []),
                    start_hour=float(summary.get('start_hour', 0.0)),
                    hours_forward=float(summary.get('hours_forward', 0.0)),
                )
            )
            filter_metrics = summary.get('filter_metrics') or {}
            if filter_metrics:
                st.caption(
                    "filter fit: prior MAE {prior:,.0f} -> post MAE {post:,.0f} | last {last_prior:,.0f} -> {last_post:,.0f}".format(
                        prior=float(filter_metrics.get('prior_mae', 0.0)),
                        post=float(filter_metrics.get('post_mae', 0.0)),
                        last_prior=float(filter_metrics.get('last_prior_mae', 0.0)),
                        last_post=float(filter_metrics.get('last_post_mae', 0.0)),
                    )
                )
                prior_mae_by_tier = filter_metrics.get('prior_mae_by_tier') or {}
                prior_signed_by_tier = filter_metrics.get('prior_signed_by_tier') or {}
                last_prior_by_tier = filter_metrics.get('last_prior_error_by_tier') or {}
                residual_rows = []
                for tier in sorted({int(t) for t in prior_mae_by_tier.keys()} | {int(t) for t in prior_signed_by_tier.keys()}):
                    residual_rows.append({
                        'tier': int(tier),
                        'prior_mae': float(prior_mae_by_tier.get(tier, prior_mae_by_tier.get(str(tier), 0.0))),
                        'prior_signed': float(prior_signed_by_tier.get(tier, prior_signed_by_tier.get(str(tier), 0.0))),
                        'last_prior_error': float(last_prior_by_tier.get(tier, last_prior_by_tier.get(str(tier), 0.0))),
                    })
                if residual_rows:
                    with st.expander("Graph filter residuals", expanded=False):
                        st.dataframe(pd.DataFrame(residual_rows), width='stretch', hide_index=True)
            curve_errors = st.session_state.get('graph_curve_errors')
            if curve_errors is not None and not curve_errors.empty:
                with st.expander("Graph rollout curve errors", expanded=True):
                    st.caption(
                        "逐曲线误差：rollout 曲线按真实观测时间点插值后计算。score_signed = pred - actual；speed_*_norm 是速度误差 / T10 scale。"
                    )
                    phase_filter = st.segmented_control(
                        "误差阶段",
                        options=["all", "start_to_24h", "middle", "final_24h"],
                        default="all",
                        key="graph_curve_error_phase",
                    )
                    visible_errors = curve_errors[curve_errors["phase"] == phase_filter].copy()
                    metric_cols = [
                        "tier", "phase", "points",
                        "score_mae", "score_rmse", "score_signed",
                        "score_max_abs", "score_max_hour",
                        "score_max_pred", "score_max_actual",
                        "speed_mae_norm", "speed_signed_norm", "speed_max_abs_norm",
                    ]
                    st.dataframe(
                        visible_errors[[col for col in metric_cols if col in visible_errors.columns]],
                        width='stretch',
                        hide_index=True,
                    )
                    top_score = curve_errors.sort_values("score_max_abs", ascending=False).head(10)
                    top_speed = curve_errors.sort_values("speed_max_abs_norm", ascending=False).head(10)
                    err_col_a, err_col_b = st.columns(2)
                    with err_col_a:
                        st.markdown("**Score 最大误差点**")
                        st.dataframe(
                            top_score[[col for col in metric_cols if col in top_score.columns]],
                            width='stretch',
                            hide_index=True,
                        )
                    with err_col_b:
                        st.markdown("**Speed 最大误差段**")
                        st.dataframe(
                            top_speed[[col for col in metric_cols if col in top_speed.columns]],
                            width='stretch',
                            hide_index=True,
                        )
            st.plotly_chart(
                st.session_state['graph_fig'],
                width='stretch',
                config={
                    'displaylogo': False,
                    'modeBarButtonsToRemove': [
                        'select2d',
                        'lasso2d',
                        'autoScale2d',
                        'toggleSpikelines',
                    ],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'graph_state_rollout',
                        'height': 950,
                        'width': 1200,
                        'scale': 2,
                    },
                },
            )
            if st.session_state.get('graph_cell_fig') is not None:
                st.markdown("### Graph Cell 状态")
                st.caption(
                    "hidden cells: {cells} | anchor cells: observed/reward ranks are kept exact".format(
                        cells=summary.get('cells', 0),
                    )
                )
                st.plotly_chart(
                    st.session_state['graph_cell_fig'],
                    width='stretch',
                    config={
                        'displaylogo': False,
                        'modeBarButtonsToRemove': [
                            'select2d',
                            'lasso2d',
                            'autoScale2d',
                            'toggleSpikelines',
                        ],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'graph_cell_state',
                            'height': 920,
                            'width': 1200,
                            'scale': 2,
                        },
                    },
                )
                cell_table = st.session_state.get('graph_cell_table')
                if cell_table is not None and not cell_table.empty:
                    active_cell_table = cell_table
                    graph_cell_frames = st.session_state.get('graph_cell_frames') or []
                    if graph_cell_frames:
                        frame_hours = [float(frame.get('hour', 0.0)) for frame in graph_cell_frames]
                        min_hour = float(min(frame_hours))
                        max_hour = float(max(frame_hours))
                        selected_hour = st.slider(
                            "Node 状态时间 (+h)",
                            min_value=min_hour,
                            max_value=max_hour,
                            value=min_hour,
                            step=1.0,
                            key="graph_node_state_hour",
                        )
                        frame_idx = min(
                            range(len(graph_cell_frames)),
                            key=lambda idx: abs(float(graph_cell_frames[idx].get('hour', 0.0)) - selected_hour),
                        )
                        selected_frame = graph_cell_frames[frame_idx]
                        active_cell_table = pd.DataFrame(selected_frame.get('rows', []))
                        st.caption(
                            "当前查看 rollout frame: +{hour:.1f}h / {count} cells".format(
                                hour=float(selected_frame.get('hour', 0.0)),
                                count=len(active_cell_table),
                            )
                        )

                    with st.expander("Node 对比与状态说明", expanded=True):
                        st.caption("输入 rank 范围如 `990-1010`，或输入 cell id 如 `cell:37,38`。两组只读取当前 graph 结果，不会重新计算。")
                        selector_col_a, selector_col_b = st.columns(2)
                        with selector_col_a:
                            group_a_selector = st.text_input(
                                "Node 组 A",
                                value="990-1010",
                                key="graph_node_group_a",
                            )
                        with selector_col_b:
                            group_b_selector = st.text_input(
                                "Node 组 B",
                                value="1010-1030",
                                key="graph_node_group_b",
                            )

                        group_a = _parse_node_selector(group_a_selector, active_cell_table)
                        group_b = _parse_node_selector(group_b_selector, active_cell_table)
                        compare_col_a, compare_col_b = st.columns(2)
                        detail_cols = [
                            'cell_id', 'rank_start', 'rank_end', 'cell_size',
                            'score', 'speed_norm', 'base_norm_speed',
                            'capacity_norm_speed',
                            'pressure', 'target_pressure', 'neighbor_pressure',
                            'density_pressure', 'pressure_source',
                            'target_rank', 'target_probs_label',
                            'target_importance',
                            'target_score', 'target_gap_norm', 'target_surplus_norm',
                            'target_profile_norm', 'target_follow_norm',
                            'target_plan_lag_norm', 'follow_rank',
                            'target_boundary_proximity',
                            'speed_cruise_norm', 'speed_target_norm',
                            'speed_profile_norm', 'speed_follow_norm',
                            'speed_lag_norm', 'speed_panic_feature_norm',
                            'speed_committed_norm',
                            'behavior_coeffs_label',
                            'speed_defend_norm', 'speed_boundary_drive',
                            'speed_preseason_norm',
                            'speed_season_effect', 'speed_desired_norm',
                            'speed_limit_reason',
                            'target_affinity', 'dominant_mode',
                            'mode_idle', 'mode_cruising', 'mode_watching',
                            'mode_chasing', 'mode_defending', 'mode_panic_rushing',
                            'mode_dropped',
                        ]
                        with compare_col_a:
                            st.markdown("**组 A 汇总**")
                            if group_a.empty:
                                st.warning("组 A 没有匹配到 cell。")
                            else:
                                st.dataframe(_node_group_summary(group_a), width='stretch', hide_index=True)
                                st.dataframe(
                                    group_a[[col for col in detail_cols if col in group_a.columns]],
                                    width='stretch',
                                    hide_index=True,
                                )
                        with compare_col_b:
                            st.markdown("**组 B 汇总**")
                            if group_b.empty:
                                st.warning("组 B 没有匹配到 cell。")
                            else:
                                st.dataframe(_node_group_summary(group_b), width='stretch', hide_index=True)
                                st.dataframe(
                                    group_b[[col for col in detail_cols if col in group_b.columns]],
                                    width='stretch',
                                    hide_index=True,
                                )

                        with st.expander("状态字段含义", expanded=False):
                            st.dataframe(pd.DataFrame(GRAPH_STATE_MEANINGS), width='stretch', hide_index=True)

                    display_cols = [
                        'cell_id', 'rank_start', 'rank_end', 'cell_size',
                        'score', 'speed_norm', 'base_norm_speed',
                        'capacity_norm_speed',
                        'pressure', 'target_pressure', 'neighbor_pressure',
                        'density_pressure', 'pressure_source', 'target_affinity',
                        'target_rank', 'target_importance', 'target_score', 'target_gap_norm', 'target_surplus_norm',
                        'target_profile_norm', 'target_follow_norm',
                        'target_plan_lag_norm', 'follow_rank',
                        'target_boundary_proximity',
                        'speed_cruise_norm', 'speed_target_norm',
                        'speed_profile_norm', 'speed_follow_norm',
                        'speed_lag_norm', 'speed_panic_feature_norm',
                        'speed_committed_norm',
                        'behavior_coeffs_label',
                        'speed_defend_norm', 'speed_boundary_drive',
                        'speed_preseason_norm',
                        'speed_season_effect', 'speed_desired_norm',
                        'speed_limit_reason',
                        'target_probs_label',
                        'dominant_mode', 'nearest_target_tier', 'nearest_reward_tier',
                        'is_observed_anchor', 'is_reward_anchor',
                    ]
                    st.dataframe(
                        active_cell_table[[col for col in display_cols if col in active_cell_table.columns]],
                        width='stretch',
                        hide_index=True,
                    )
    else:
        tier_errors = st.session_state.get('tier_errors', {})
        if tier_errors:
            for tier, err in tier_errors.items():
                st.warning(f"T{tier}: {err}")
        else:
            st.info("🐱 暂无数据，正在等待初始化或手动触发...")

with col_info:
    st.markdown("### 状态面板")
    st.write(f"最后更新: **{st.session_state['last_update_str']}**")
    if st.session_state.get('run_timings'):
        st.markdown("**本轮耗时**")
        for label, elapsed in st.session_state['run_timings'].items():
            st.write(f"{label}: `{elapsed:.2f}s`")

    if st.session_state.get('img_bytes'):
        st.success("系统运行正常 喵！")
