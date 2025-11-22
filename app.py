import streamlit as st
import time
import traceback
from datetime import datetime, timedelta, timezone

# 引入后端逻辑
from config import DEFAULT_CONFIG
from predictor import DataHandler, fetch_recent_json, get_current_event_for_server

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="自动预测面板", page_icon="🐱", layout="wide")

st.title("🐱 实时预测面板")

# ==========================================
# 2. 初始化 Session State
# ==========================================
if 'img_bytes' not in st.session_state:
    st.session_state['img_bytes'] = None
if 'last_update_str' not in st.session_state:
    st.session_state['last_update_str'] = "暂无数据"

# 用于判断是否是首次加载的 Flag
if 'has_initialized' not in st.session_state:
    st.session_state['has_initialized'] = False

# ==========================================
# 3. 侧边栏控制
# ==========================================
st.sidebar.header("控制台 🎮")

manual_btn = st.sidebar.button("⚡ 立即运行预测", type="primary")

# --- 高级参数配置 (Advanced Config) ---
st.sidebar.markdown("---")
with st.sidebar.expander("参数设置"):
    st.caption("调整下列参数将覆盖 config.py 的默认值")
    
    # 1. 模型参数
    st.markdown("**模型参数**")
    weekend_mult = st.slider(
        "周末增强系数", 
        min_value=0.8, max_value=1.5, step=0.05,
        value=DEFAULT_CONFIG.get('weekend_multiplier', 1.0),
        help="大于1.0表示预测周末相较工作日会有额外增幅，注意并非一定会让预测值上升，这主要作用于模型预测速度分布的形状"
    )

    panic_scaler = st.slider(
        "恐慌期最小加速倍数",
        min_value=1.0, max_value=3.0, step=0.05,
        value=DEFAULT_CONFIG.get('panic_scaler', 1.1),
        help="恐慌期的最小加速倍数，数值越大表示加速效果越明显"
    )

    panic_ease_power = st.slider(
        "恐慌期缓动指数",
        min_value=0.1, max_value=5.0, step=0.1,
        value=DEFAULT_CONFIG.get('panic_ease_power', 1.0),
        help="控制恐慌期的缓动效果，数值越大“龙抬头”效果越晚"
    )
    
    similar_count = st.number_input(
        "参考历史活动数",
        min_value=1, max_value=10, step=1,
        value=DEFAULT_CONFIG.get('similar_count', 5),
        help="不建议调整，更不建议设置太少"
    )

    st.markdown("以下参数不建议轻易调整")

    # 2. 阈值与限制
    with st.sidebar.expander("阈值与限制"):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            ratio_min = st.number_input("Ratio Min", value=DEFAULT_CONFIG.get('ratio_min', 0.25), step=0.05)
            scale_min = st.number_input("Scale Min", value=DEFAULT_CONFIG.get('scale_min', 0.5), step=0.1)
            # 对比窗口起始时间 (小时)
            t_start_cmp = st.number_input(
                "对比窗口起始 (小时)", min_value=0.0, value=DEFAULT_CONFIG.get('t_start_cmp', 6.0), step=0.5,
                help="用于计算历史相似性时跳过开局不稳定（常被维护时间占用）的时间（小时）"
            )
        with col_p2:
            ratio_max = st.number_input("Ratio Max", value=DEFAULT_CONFIG.get('ratio_max', 4.0), step=0.1)
            scale_max = st.number_input("Scale Max", value=DEFAULT_CONFIG.get('scale_max', 2.0), step=0.1)
            # 对比窗口结束上限 (小时)
            t_end_cap = st.number_input(
                "窗口结束上限 (小时)", min_value=1.0, value=DEFAULT_CONFIG.get('t_end_cap', 72.0), step=1.0,
                help="历史对比时考虑的最大小时数，上限用于避免中后期数据干扰"
            )

    # 3. 24h 回测修正与顶部平滑阈值
    with st.sidebar.expander("回测与平滑设置"):
        corr_min = st.number_input("24h 回测修正下限", value=DEFAULT_CONFIG.get('corr_min', 0.6), step=0.05)
        corr_max = st.number_input("24h 回测修正上限", value=DEFAULT_CONFIG.get('corr_max', 1.6), step=0.05)

        st.markdown("**顶部平滑阈值**")
        smooth_thresh1 = st.number_input("轻微衰减阈值 (比例)", min_value=0.0, max_value=1.0, value=DEFAULT_CONFIG.get('smooth_thresh1', 0.5), step=0.01)
        smooth_thresh2 = st.number_input("强力衰减阈值 (比例)", min_value=0.0, max_value=1.0, value=DEFAULT_CONFIG.get('smooth_thresh2', 0.65), step=0.01)
        smooth_hard_cap = st.number_input("绝对硬顶 (比例)", min_value=0.0, max_value=1.0, value=DEFAULT_CONFIG.get('smooth_hard_cap', 0.8), step=0.01)

# --- 调试回测 ---
st.sidebar.markdown("---")
st.sidebar.header("调试回测 🛠️")
enable_debug = st.sidebar.checkbox("启用调试/回测模式", value=False)

if enable_debug:
    debug_event_id = st.sidebar.number_input("目标 Event ID", min_value=1, value=312, step=1)
    debug_hours_input = st.sidebar.number_input("冻结时间 (小时)", min_value=0.0, value=60.0, step=1.0, format="%.1f")
else:
    debug_event_id = None
    debug_hours_input = None


# ==========================================
# 4. 核心逻辑 (首次自动 + 手动触发)
# ==========================================

# 判定逻辑：如果是(手动点击) 或者 (当前Session还没初始化过)
# 注意：Streamlit 每次交互都会重跑脚本，所以要用 session_state 锁住自动运行
should_run = False
trigger_reason = ""

if manual_btn:
    should_run = True
    trigger_reason = "手动触发"
elif not st.session_state['has_initialized']:
    should_run = True
    trigger_reason = "首次加载自动运行"
    # 标记为已初始化，防止后续只要刷新页面就重跑（除非彻底刷新浏览器Tab）
    st.session_state['has_initialized'] = True

if should_run:
    with st.spinner(f"🐱 ({trigger_reason}) 正在获取数据并绘图..."):
        try:
            if enable_debug and debug_event_id is not None:
                target_event_id = int(debug_event_id)
                target_debug_hours = float(debug_hours_input) if debug_hours_input is not None else None
            else:
                recent = fetch_recent_json()
                target_event_id = get_current_event_for_server(recent, server_index=3)
                target_debug_hours = None

            if target_event_id is None:
                st.error("未找到活动 ID！")
            else:
                # --- 构建配置覆盖字典 ---
                user_config_overrides = {
                    'weekend_multiplier': weekend_mult,
                    'panic_scaler': float(panic_scaler),
                    'panic_ease_power': float(panic_ease_power),
                    'similar_count': int(similar_count),
                    'ratio_min': float(ratio_min),
                    'ratio_max': float(ratio_max),
                    'scale_min': float(scale_min),
                    'scale_max': float(scale_max),
                    't_start_cmp': float(t_start_cmp),
                    't_end_cap': float(t_end_cap),
                    'corr_min': float(corr_min),
                    'corr_max': float(corr_max),
                    'smooth_thresh1': float(smooth_thresh1),
                    'smooth_thresh2': float(smooth_thresh2),
                    'smooth_hard_cap': float(smooth_hard_cap)
                }

                # --- 传入 config_overrides ---
                handler = DataHandler(
                    target_event_id, 
                    debug_hours=target_debug_hours,
                    config_overrides=user_config_overrides
                )
                
                handler.load_target_data()
                handler.find_similar_events()

                new_img = handler.run_prediction(return_type='bytes')

                if new_img:
                    st.session_state['img_bytes'] = new_img
                    
                    # --- 时间处理部分 ---
                    # 定义北京时区 (UTC+8)
                    beijing_tz = timezone(timedelta(hours=8))
                    # 获取当前UTC时间并转换为北京时间
                    now_bj = datetime.now(beijing_tz)
                    st.session_state['last_update_str'] = now_bj.strftime('%H:%M:%S (北京时间)')
                    
                    if manual_btn:
                        st.success(f"刷新成功！(Event {target_event_id})")
                else:
                    st.warning("计算完成，但没有生成图片数据。")

        except Exception as e:
            st.error(f"运行出错: {str(e)}")
            print(traceback.format_exc())

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
        # 如果首次运行出错导致没有图片，这里会显示
        st.info("🐱 似乎没有数据呢，请检查网络或点击按钮重试...")

with col_info:
    st.markdown("### 状态面板")
    # 这里会显示明确的北京时间
    st.write(f"最后更新: **{st.session_state['last_update_str']}**")
    
    st.caption("机制说明：首次进入自动刷新，后续需手动点击按钮。")