import streamlit as st
import time
import traceback
from datetime import datetime, timedelta, timezone

# 引入后端逻辑
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
                handler = DataHandler(target_event_id, debug_hours=target_debug_hours)
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