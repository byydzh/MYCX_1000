# app.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import streamlit as st
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import API_SOURCE_CONFIGS, DEFAULT_CONFIG, list_models, list_presets, load_preset
from data_source import create_data_source
from domain_models import EventData, EventMeta
from math_models import SeasonalityHandler, CosineModeler
from prediction_engine import PredictionEngine
from plotly_viz import plot_prediction_plotly

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
if 'last_run_error' not in st.session_state:
    st.session_state['last_run_error'] = None
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
if 'run_timings' not in st.session_state:
    st.session_state['run_timings'] = {}
if 'data_source_status' not in st.session_state:
    st.session_state['data_source_status'] = []

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
        help=(
            "HHWX 为主源；其请求、结构或空数据失败时会自动切换 "
            "Bestdori，并在页面明示实际来源。直接选 Bestdori 时不会再切换。"
        ),
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

if should_run:
    st.session_state['last_run_error'] = None
    st.session_state['data_source_status'] = []
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
        ds = None
        try:
            source_key = current_config.get('api_source')
            fallback_enabled = str(source_key).lower() == 'hhwx'
            ds = create_data_source(source_key, allow_fallback=fallback_enabled)
            source_status = []

            def _capture_source_status(label, record):
                if not isinstance(record, dict):
                    return
                source_status.append({
                    'label': str(label),
                    'requested_source': record.get('requested_source', source_key),
                    'source': record.get('source', source_key),
                    'fallback_used': bool(record.get('fallback_used', False)),
                    'primary_error': record.get('primary_error'),
                })

            # 1. 获取目标 ID
            if enable_debug and debug_event_id:
                target_eid = int(debug_event_id)
            else:
                event_index_url = API_SOURCE_CONFIGS.get(source_key, {}).get('event_index_url', '未配置')
                try:
                    target_eid = ds.get_current_event_id()
                    _capture_source_status(
                        "活动列表/当前活动",
                        ds.get_provenance("current_event"),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"{str(source_key).upper()} 自动活动选择异常"
                        f"（事件列表 {event_index_url}）："
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

            if not target_eid:
                raise RuntimeError(
                    f"{str(source_key).upper()} 自动活动选择失败："
                    f"事件列表 {event_index_url} 已加载，"
                    "但 get_current_event_id() 未找到可用活动。"
                )
            elif not selected_tiers:
                st.warning("请至少选择一个榜线。")
            else:
                # --- 先获取 meta（全局共享）---
                meta_raw = ds.fetch_event_meta(target_eid)
                if not meta_raw:
                    raise RuntimeError(f"无法获取活动 {target_eid} 的元数据。")
                else:
                    _capture_source_status(
                        "活动元数据",
                        ds.get_provenance("event_meta"),
                    )
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
                    scale_observation = ds.fetch_top10_max_speed_observation(
                        target_eid,
                        debug_limit_ts=target_debug_limit_ts,
                        allow_fallback=fallback_enabled,
                    )
                    _capture_source_status(
                        "T10 scale",
                        {
                            'requested_source': source_key,
                            'source': scale_observation.source,
                            'fallback_used': scale_observation.fallback_used,
                            'primary_error': scale_observation.primary_error,
                        },
                    )
                    scale_val = scale_observation.value
                    if (
                        scale_val is None
                        or not np.isfinite(scale_val)
                        or scale_val <= 0
                    ):
                        scale_url = ds.api_config['top10_url'].format(
                            server=ds.server_index,
                            event_id=target_eid,
                        )
                        failure_details = [
                            scale_observation.primary_error or scale_url
                        ]
                        if fallback_enabled:
                            failure_details.append(
                                scale_observation.fallback_error
                                or 'Bestdori 回退未返回可计算速度'
                            )
                        raise RuntimeError(
                            "T10 scale 获取失败：" + "；".join(failure_details)
                        )
                    _mark_timing("共享资源: meta/scale")

                    # --- 并行获取各线数据 ---
                    def _fetch_one_tier(tier):
                        tds = create_data_source(
                            current_config.get('api_source'),
                            allow_fallback=fallback_enabled,
                        )
                        try:
                            tp = tds.fetch_event_data_pack(target_eid, tier=tier, meta=meta_raw, scale=scale_val)
                            if not tp:
                                return tier, None, None, "数据不可用", []
                            sp = tds.find_similar_events(
                                target_eid, event_type,
                                count=int(current_config.get('similar_count', similar_count)),
                                ignore_ids=current_config.get('ignore_event_ids', ignore_ids),
                                tier=tier,
                                allow_scale_fallback=fallback_enabled,
                                allow_tier_interpolation=False,
                            )
                            notices = []
                            tracker_provenance = (
                                tp.get('source_provenance', {}).get('tracker') or {}
                            )
                            if tracker_provenance:
                                notices.append({
                                    'label': f'T{tier} 当前榜线',
                                    **tracker_provenance,
                                })
                            history_sources = {}
                            for pack in sp:
                                tracker_source = str(pack.get('source') or '').lower()
                                if tracker_source:
                                    history_sources.setdefault(tracker_source, False)
                                if pack.get('fallback_used'):
                                    history_sources[str(tds.fallback_api_source).lower()] = True

                            for actual_source, used_fallback in sorted(history_sources.items()):
                                notices.append({
                                    'label': f'T{tier} 历史参考',
                                    'requested_source': source_key,
                                    'source': actual_source,
                                    'fallback_used': used_fallback,
                                })
                            return tier, tp, sp, None, notices
                        except Exception as exc:
                            return tier, None, None, str(exc), []
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
                            tier, tp, sp, err, notices = f.result()
                            for notice in notices:
                                _capture_source_status(notice.get('label'), notice)
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


                    st.session_state['tier_results'] = tier_results_dict
                    st.session_state['tier_targets'] = tier_data_dict
                    st.session_state['tier_errors'] = tier_errors_dict
                    st.session_state['tier_warnings'] = tier_warnings_dict
                    st.session_state['current_event_id'] = target_eid
                    st.session_state['data_source_status'] = source_status
                    st.session_state['is_debug_mode'] = enable_debug
                    if not tier_results_dict:
                        error_details = "；".join(
                            f"T{tier}: {message}"
                            for tier, message in sorted(tier_errors_dict.items())
                        ) or "未生成任何榜线结果"
                        raise RuntimeError(f"Event {target_eid} 预测失败：{error_details}")

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
                    st.session_state['has_initialized'] = True
                    st.session_state['last_run_error'] = None

                    if manual_btn:
                        passed = len(tier_results_dict)
                        st.success(f"预测完成！Event {target_eid} | {passed}/{len(selected_tiers)} 线成功")

        except Exception as e:
            st.session_state['has_initialized'] = False
            if 'source_status' in locals():
                st.session_state['data_source_status'] = source_status
            st.session_state['last_run_error'] = f"{type(e).__name__}: {e}"
            logger.exception("预测运行失败")
        finally:
            if ds is not None:
                ds.close()

# ==========================================
# 5. 结果展示
# ==========================================
if st.session_state.get('last_run_error'):
    st.error(f"上次运行失败：{st.session_state['last_run_error']}")

source_status = st.session_state.get('data_source_status', [])
if source_status:
    actual_sources = []
    for item in source_status:
        actual_source = str(item.get('source') or '').strip().upper()
        if actual_source and actual_source not in actual_sources:
            actual_sources.append(actual_source)

    if actual_sources:
        source_summary = " / ".join(actual_sources)
        if any(item.get('fallback_used') for item in source_status):
            source_summary += "（自动切换）"
        st.caption("本轮数据来源：" + source_summary)

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
