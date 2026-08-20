import json
import numbers
from pathlib import Path
from typing import Dict, List, Optional

# ==========================================
# API 数据源配置
# ==========================================
DEFAULT_SERVER = 3  # 国服
DEFAULT_API_SOURCE = "hhwx"
DEFAULT_FALLBACK_API_SOURCE = "bestdori"

# HHWX 的实时 tracker 按分钟级刷新。实时缓存只用于合并同一轮
# Streamlit 重复请求，不能跨越下一个刷新窗口。
EVENTS_INDEX_CACHE_TTL_SECONDS = 60.0
LIVE_SCALE_CACHE_TTL_SECONDS = 60.0

# Public tracker contract shared by this project and the HHWX fixed-tier list.
# Individual events may expose only a subset; rank is not a continuous query.
ALL_TRACKER_TIERS = [
    1, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500,
    1000, 1500, 2000, 3000, 4000, 5000,
    10000, 20000, 30000, 40000, 50000, 70000, 100000,
]
TRACKER_TIER_SET = frozenset(ALL_TRACKER_TIERS)


def validate_tracker_tier(tier: int) -> int:
    """Return a supported public tracker tier, or fail before any HTTP call."""
    if isinstance(tier, bool):
        raise ValueError(f"Tracker tier must be an integer, got {tier!r}")
    if isinstance(tier, numbers.Integral):
        normalized = int(tier)
    elif isinstance(tier, str):
        text = tier.strip()
        if not text.isdecimal() or str(int(text)) != text:
            raise ValueError(
                f"Tracker tier must be a canonical integer, got {tier!r}"
            )
        normalized = int(text)
    else:
        raise ValueError(f"Tracker tier must be an integer, got {tier!r}")
    if normalized not in TRACKER_TIER_SET:
        raise ValueError(
            f"Unsupported tracker tier T{normalized}. Public APIs only expose "
            f"fixed tiers: {ALL_TRACKER_TIERS}"
        )
    return normalized


def canonicalize_tracker_tiers(tiers) -> list[int]:
    """Validate a tier collection and return it in the public-contract order."""
    if tiers is None:
        return list(ALL_TRACKER_TIERS)
    requested = [validate_tracker_tier(tier) for tier in tiers]
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate tracker tier request: {requested}")
    requested_set = set(requested)
    return [tier for tier in ALL_TRACKER_TIERS if tier in requested_set]

API_SOURCE_CONFIGS = {
    "hhwx": {
        "label": "HHWX Tracker Proxy",
        "event_index_url": "https://hhwx.org/api/bandori/master/events",
        "event_meta_url": "https://hhwx.org/api/bandori/master/events/{event_id}",
        "tracker_url": "https://hhwx.org/api/bandori/tracker/data?server={server}&event={event_id}&type=event&tier={tier}",
        "top10_url": "https://hhwx.org/api/bandori/tracker/data?server={server}&event={event_id}&type=event&tier=10",
    },
    "bestdori": {
        "label": "Bestdori Public API",
        "event_index_url": "https://bestdori.com/api/events/all.3.json",
        "event_meta_url": "https://bestdori.com/api/events/{event_id}.json",
        "tracker_url": "https://bestdori.com/api/tracker/data?server={server}&event={event_id}&tier={tier}",
        "top10_url": "https://bestdori.com/api/eventtop/data?server={server}&event={event_id}&mid=0&interval=3600000",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_REGISTRY_PATH = CONFIGS_DIR / "models.json"
DEFAULT_MODEL_ID = "skeleton_kf"

BUILTIN_MODELS = {
    DEFAULT_MODEL_ID: {
        "name": "Skeleton + Kalman Filter",
        "description": "基于历史骨架拟合 + 实时卡尔曼滤波修正",
        "engine_class": "PredictionEngine",
        "module": "prediction_engine",
        "config_dir": "configs/models/skeleton_kf",
    }
}

# ==========================================
# 预测器默认配置字典 (Default Configuration)
# ==========================================
DEFAULT_CONFIG = {
    # --- 数据源 ---
    'api_source': DEFAULT_API_SOURCE, # 默认使用的 API profile

    # --- 节律与模型参数 ---
    'weekend_multiplier': 1.1,      # 周末增强系数 (1.0 = 无增强, >1.0 = 周末加速)
    'panic_ease_power': 1.0,        # 恐慌期缓动指数 (控制最后时刻加速曲线的弯曲程度)
    'panic_scaler': 1.1,            # 恐慌期最小加速倍数
    
    # --- 相似活动搜索 ---
    'similar_count': 5,             # 寻找相似历史活动的数量
    'ignore_event_ids': [297, 298], # 忽略的活动ID列表 (e.g. [297, 298])

    # --- 预测逻辑: 对比窗口 ---
    't_start_cmp': 6.0,             # 对比起始时间 (小时，跳过开局数据不稳定)
    't_end_cap': 72.0,              # 对比结束时间上限 (小时，防止中后期数据干扰)
    'duration_align_log_full_weight': 0.18, # 时长差异达到约 18% 时，进度对齐/时长归一进入强介入区
    'duration_window_align_max_weight': 0.75, # Ratio 对比窗口最多有多少权重切到“按进度对齐”
    'duration_param_align_max_weight': 0.85, # 骨架参数时长归一的最大介入力度
    
    # --- 预测逻辑: 限制阈值 (Safety Bounds) ---
    'ratio_min': 0.25,              # 强度比率下限 
    'ratio_max': 4.0,               # 强度比率上限 
    'scale_min': 0.5,               # 观测Scaling下限
    'scale_max': 2.0,               # 观测Scaling上限
    'corr_min': 0.6,                # 24h回测修正下限
    'corr_max': 1.6,                # 24h回测修正上限
    
    # --- 顶部平滑参数 (Top Speed Smoothing) ---
    # 当预测速度达到 T10 极速的百分比时开始衰减，防止预测出超越物理极限的速度
    'smooth_thresh1': 0.5,          # 第一阶段轻微衰减阈值 (50% 极速)
    'smooth_thresh2': 0.65,         # 第二阶段强力衰减阈值 (65% 极速)
    'smooth_hard_cap': 0.8,         # 绝对硬顶 (80% 极速，不可逾越之墙)

    # --- 拟合权重参数 ---
    'refit_weight_scale': 10.0,     # 对数加权系数 (log1p(x * scale))，越大则低值区权重越低
    'refit_min_points': 10,         # Refit 至少需要的观测点数
    'refit_lambda': 0.3,            # Refit 正则强度
    'refit_start_hours': 6.0,       # Refit 起始时间，避免首日冷启动直接拉歪形状
    'refit_recent_hours': 48.0,     # Refit 仅关注最近一段观测窗口，避免过早历史过度主导
    'refit_conf_norm_hours': 72.0,  # Refit 置信度归一化时长
    'refit_conf_max': 0.35,         # Refit 最大混合权重，避免在线重拟合直接接管骨架
    'refit_base_min_ratio': 0.6,    # Base 对先验的下界比例
    'refit_base_max_ratio': 1.6,    # Base 对先验的上界比例
    'refit_linear_bound_scale': 2.0,# A 对先验的可扩张倍数（保留符号）
    'refit_linear_zero_ratio': 0.25,# A 向 0 收缩的最近边界比例（保留符号）
    'refit_quad_min_ratio': 0.1,    # B 对先验的下界比例，防止被直接压成 0
    'refit_quad_max_ratio': 2.0,    # B 对先验的上界比例

    # --- Kalman Filter 参数 ---
    'kf_dt_step': 1.0,              # KF 更新时间步长（小时）
    'kf_init_P_scale': 0.5,         # KF 初始 Scale 协方差
    'kf_init_P_trend': 0.01,        # KF 初始 Trend 协方差
    'kf_Q_scale': 1e-5,             # KF 过程噪声（Scale）
    'kf_Q_trend': 1e-7,             # KF 过程噪声（Trend）
    'kf_min_delta_model': 50.0,     # KF 观测生效所需最小模型增量
    'kf_z_clip_min': 0.1,           # KF 观测值下截断
    'kf_z_clip_max': 5.0,           # KF 观测值上截断
    'kf_base_R': 0.1,               # KF 基础测量噪声
    'kf_clip_penalty': 100.0,       # KF 观测截断惩罚
    'kf_adaptive_R_numerator': 2000.0, # KF 自适应测量噪声分子
    'kf_scale_clip_min': 0.5,       # KF 最终 Scale 下截断
    'kf_scale_clip_max': 3.0,       # KF 最终 Scale 上截断
    'kf_trend_clip_min': -0.03,     # KF 最终 Trend 下截断
    'kf_trend_clip_max': 0.05,      # KF 最终 Trend 上截断
    'kf_trend_half_life_hours': 2.0,# KF 趋势半衰期（小时）
}


def _read_json_file(path: Path) -> Optional[dict]:
    """安全读取 JSON 文件，失败时返回 None。"""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _resolve_model_registry() -> Dict[str, dict]:
    registry = _read_json_file(MODELS_REGISTRY_PATH)
    if registry:
        return registry
    return BUILTIN_MODELS.copy()


def _resolve_model_entry(model_id: str) -> Optional[dict]:
    registry = _resolve_model_registry()
    model_info = registry.get(model_id)
    if not isinstance(model_info, dict):
        return None
    return {"id": model_id, **model_info}


def _resolve_model_config_dir(model_info: dict) -> Optional[Path]:
    config_dir = model_info.get("config_dir")
    if not config_dir:
        return None

    config_path = Path(config_dir)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path


def list_models() -> List[dict]:
    """返回可用模型列表。"""
    models = []
    for model_id, model_info in _resolve_model_registry().items():
        if isinstance(model_info, dict):
            models.append({"id": model_id, **model_info})
    return models


def list_presets(model_id: str) -> List[dict]:
    """扫描模型目录下的预设 JSON，并返回展示用元数据。"""
    model_info = _resolve_model_entry(model_id)
    if not model_info:
        return []

    config_dir = _resolve_model_config_dir(model_info)
    if config_dir is None or not config_dir.exists():
        return []

    presets = []
    for preset_path in sorted(config_dir.glob("*.json")):
        preset_json = _read_json_file(preset_path) or {}
        meta = preset_json.get("_meta", {})
        if not isinstance(meta, dict):
            meta = {}

        presets.append({
            "id": preset_path.stem,
            "name": meta.get("name", preset_path.stem),
            "description": meta.get("description", ""),
            "author": meta.get("author", ""),
            "created_at": meta.get("created_at", ""),
            "path": str(preset_path),
        })

    presets.sort(key=lambda item: (item["id"] != "default", item["name"].lower(), item["id"]))
    return presets


def load_preset(model_id: str, preset_name: str) -> dict:
    """
    读取预设并与 DEFAULT_CONFIG 合并。
    缺失项或损坏文件会自动回退到默认配置。
    """
    merged = DEFAULT_CONFIG.copy()

    model_info = _resolve_model_entry(model_id)
    if not model_info:
        return merged

    config_dir = _resolve_model_config_dir(model_info)
    if config_dir is None:
        return merged

    preset_path = config_dir / f"{preset_name}.json"
    preset_json = _read_json_file(preset_path)
    if not preset_json:
        return merged

    params = preset_json.get("params", {})
    if isinstance(params, dict):
        merged.update(params)

    return merged
