# app.py
import streamlit as st
import time
import traceback
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta, timezone
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 引入新架构的组件
from config import API_SOURCE_CONFIGS, DEFAULT_CONFIG, list_models, list_presets, load_preset
from data_source import create_data_source
from domain_models import EventData, EventMeta
from math_models import SeasonalityHandler, CosineModeler
from prediction_engine import PredictionEngine
from visualizer import Visualizer

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
        scale=data_pack['scale']
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
if PRESET_SIGNATURE_KEY not in st.session_state:
    st.session_state[PRESET_SIGNATURE_KEY] = None

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
    index=model_options.index(st.session_state[MODEL_STATE_KEY]),
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
    index=preset_options.index(st.session_state[PRESET_STATE_KEY]),
    format_func=lambda preset_id: preset_lookup[preset_id].get("name", preset_id),
    key=PRESET_STATE_KEY,
)

selected_preset_meta = preset_lookup[selected_preset]
if selected_preset_meta.get("description"):
    st.sidebar.caption(selected_preset_meta["description"])

current_signature = f"{selected_model}:{selected_preset}"
if st.session_state.get(PRESET_SIGNATURE_KEY) != current_signature:
    _apply_preset_to_session(selected_model, selected_preset)

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
        help="切换活动元数据与榜线数据接口；HHWX 配置默认启用，T10 scale 仍会保留 Bestdori 兜底。",
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
            else:
                # 2. 获取目标数据
                target_pack = ds.fetch_event_data_pack(target_eid)
                if not target_pack:
                    st.error(f"无法获取活动 {target_eid} 的详细数据。")
                else:
                    target_data = wrap_event_data(target_pack)
                    target_data = calculate_derived_columns(target_data)
                    target_data.full_df = target_data.df.copy() # 保存上帝视角副本

                    # --- 时间解析与转换逻辑 ---
                    start_ts = target_data.meta.start_at
                    tz_utc8 = timezone(timedelta(hours=8))
                    
                    # A. 处理冻结时间
                    target_debug_h = None
                    if enable_debug and not use_max_time and debug_freeze_str:
                        # 1. 尝试解析为纯数字 (相对小时)
                        try:
                            target_debug_h = float(debug_freeze_str)
                        except ValueError:
                            # 2. 尝试解析为 UTC+8 时间字符串
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
                                st.error(f"冻结时间解析失败: {e}，将使用最大观测时间")
                                target_debug_h = None
                        
                        if target_debug_h is not None and target_debug_h < 0:
                            st.warning("冻结时间早于活动开始时间，将使用 0.0 小时")
                            target_debug_h = 0.0

                    # B. 处理人工干预点
                    manual_points = []
                    if manual_points_raw:
                        for mp in manual_points_raw:
                            try:
                                # 同样尝试解析时间
                                t_str = mp['time_str']
                                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"]:
                                    try:
                                        dt_mp = datetime.strptime(t_str, fmt)
                                        break
                                    except ValueError:
                                        continue
                                else:
                                    st.warning(f"跳过无法解析的时间: {t_str}")
                                    continue
                                
                                dt_mp = dt_mp.replace(tzinfo=tz_utc8)
                                ts_mp = dt_mp.timestamp() * 1000
                                h_mp = (ts_mp - start_ts) / 3600000.0
                                
                                manual_points.append({'hours': h_mp, 'score': mp['score']})
                            except Exception as e:
                                st.warning(f"处理干预点出错: {mp} - {e}")

                    # --- 异常输入检查 (仅警告) ---
                    if manual_points:
                        # 1. 检查时间范围
                        total_h = target_data.meta.total_hours
                        for mp in manual_points:
                            if mp['hours'] < 0 or mp['hours'] > total_h:
                                st.warning(f"⚠️ 警告: 干预点时间 {mp['hours']:.1f}h 超出活动范围 (0~{total_h}h)")
                        
                        # 2. 检查分数倒退 (负速度)
                        # 需要结合当前最新数据
                        current_max_score = target_data.df['value'].max()
                        current_max_time = target_data.df['hours_elapsed'].max()
                        
                        # 按时间排序
                        sorted_mps = sorted(manual_points, key=lambda x: x['hours'])
                        
                        last_s = current_max_score
                        last_t = current_max_time
                        
                        for mp in sorted_mps:
                            if mp['hours'] <= last_t:
                                st.warning(f"⚠️ 警告: 干预点时间 {mp['hours']:.1f}h 早于或等于前一个点 ({last_t:.1f}h)，将被忽略")
                                continue
                                
                            if mp['score'] < last_s:
                                st.warning(f"⚠️ 警告: 干预点分数 {int(mp['score'])} 低于前一个点 ({int(last_s)})，意味着负增长")
                            
                            # 3. 检查超高速
                            # 粗略计算一下平均速度
                            delta_s = mp['score'] - last_s
                            delta_t = mp['hours'] - last_t
                            if delta_t > 0:
                                speed_val = delta_s / delta_t
                                # 归一化速度 > 1.0 意味着超过了理论最大速度 (scale)
                                if target_data.scale > 0:
                                    norm_spd = (speed_val / 60.0) / target_data.scale
                                    if norm_spd > 1.0:
                                        st.warning(f"⚠️ 警告: 干预区间速度超过理论极限 (Norm Speed ≈ {norm_spd:.2f} > 1.0)，请检查输入")

                            last_s = mp['score']
                            last_t = mp['hours']
                    # ---------------------------

                    # 截断逻辑
                    if target_debug_h is not None:
                        limit_ts = target_data.meta.start_at + (target_debug_h * 3600 * 1000)
                        target_data.df = target_data.df[target_data.df['time'] <= limit_ts].copy()

                    # 3. 获取历史数据
                    similar_packs = ds.find_similar_events(
                        target_eid,
                        target_data.meta.event_type,
                        count=int(current_config.get('similar_count', similar_count)),
                        ignore_ids=current_config.get('ignore_event_ids', ignore_ids)
                    )
                    history_events = []
                    for pack in similar_packs:
                        h_data = wrap_event_data(pack)
                        try:
                            h_data = calculate_derived_columns(h_data)
                            history_events.append(h_data)
                        except: pass
                    
                    # 4. 初始化引擎组件
                    seasonality = SeasonalityHandler(
                        weekend_multiplier=float(current_config.get('weekend_multiplier', weekend_mult)),
                        panic_scaler=float(current_config.get('panic_scaler', panic_scaler)),
                        panic_ease_power=float(current_config.get('panic_ease_power', panic_ease_power))
                    )
                    modeler = CosineModeler()
                    engine = PredictionEngine(seasonality, modeler, config=current_config)
                    visualizer = Visualizer()

                    # 5. 执行预测
                    result = engine.predict(
                        target_data,
                        history_events,
                        debug_hours=target_debug_h,
                        manual_points=manual_points
                    )

                    # 6. 绘图 (内存操作)
                    fig = visualizer.plot_prediction(
                        target_data,
                        result,
                        debug_hours=target_debug_h,
                        manual_points=manual_points,
                        save=False
                    )
                    
                    # 转 BytesIO
                    buf = BytesIO()
                    FigureCanvasAgg(fig).print_png(buf)
                    buf.seek(0)
                    st.session_state['img_bytes'] = buf
                    
                    # 更新时间
                    beijing_tz = timezone(timedelta(hours=8))
                    st.session_state['last_update_str'] = datetime.now(beijing_tz).strftime('%H:%M:%S')
                    
                    if manual_btn:
                        st.success(f"预测完成！Event {target_eid} | Final: {int(result.final_score):,}")

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
        st.image(
            st.session_state['img_bytes'],
            caption=f"预测趋势图 (更新于: {st.session_state['last_update_str']})",
            width="content"
        )
    else:
        st.info("🐱 暂无数据，正在等待初始化或手动触发...")

with col_info:
    st.markdown("### 状态面板")
    st.write(f"最后更新: **{st.session_state['last_update_str']}**")
    
    if st.session_state.get('img_bytes'):
         st.success("系统运行正常 喵！")
