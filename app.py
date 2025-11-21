import streamlit as st
import time
import traceback
from datetime import datetime
# 务必确保安装了此库: pip install streamlit-autorefresh
from streamlit_autorefresh import st_autorefresh

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
# 记录上一次【实际执行计算】的时间戳
if 'last_run_ts' not in st.session_state:
    st.session_state['last_run_ts'] = 0.0

# ==========================================
# 3. 自动刷新配置 (心跳包)
# ==========================================
# 这里的 interval 是 300000 毫秒 = 300 秒 = 5 分钟
# 它的作用仅仅是每5分钟让脚本从头到尾“空跑”一次，以便触发下面的时间检查
st_autorefresh(interval=300_000, limit=None, key="auto_refresher_5min")

# ==========================================
# 4. 侧边栏控制
# ==========================================
st.sidebar.header("控制台 🎮")

# 手动按钮
manual_btn = st.sidebar.button("⚡ 立即运行预测", type="primary")

st.sidebar.markdown("---")
st.sidebar.header("调试回测 🛠️")
enable_debug = st.sidebar.checkbox("启用调试/回测模式", value=False, help="勾选后将强制使用下方指定的活动ID和时间进度，填好后手动运行。虽然一般可以中途修改，但还是请尽量避免在已经运行时使用以防止潜在的线程冲突")

# 只有勾选时才显示输入框，保持界面整洁喵
if enable_debug:
    debug_event_id = st.sidebar.number_input("目标 Event ID", min_value=1, value=312, step=1)
    debug_hours_input = st.sidebar.number_input("冻结时间 (小时)", min_value=0.0, value=60.0, step=1.0, format="%.1f")
else:
    debug_event_id = None
    debug_hours_input = None

# ==========================================
# 5. 核心逻辑 (触发条件判断)
# ==========================================

# 获取当前时间
now_ts = time.time()
# 计算距离上次运行过去了多久
elapsed = now_ts - st.session_state['last_run_ts']
# 设定自动运行的阈值 (30分钟 = 1800秒)
AUTO_INTERVAL_SEC = 1800

# 判定：如果是 (手动点击) 或者 (距离上次运行超过了30分钟)
should_run = False
trigger_reason = ""

if manual_btn:
    should_run = True
    trigger_reason = "手动触发"
elif elapsed >= AUTO_INTERVAL_SEC:
    should_run = True
    trigger_reason = "自动定时触发"

# ==========================================
# 6. 执行计算
# ==========================================
if should_run:
    with st.spinner(f"🐱 ({trigger_reason}) 正在获取数据并绘图..."):
        try:
            # Determine target event and optional debug hours depending on debug mode
            if enable_debug and debug_event_id is not None:
                target_event_id = int(debug_event_id)
                target_debug_hours = float(debug_hours_input) if debug_hours_input is not None else None
            else:
                # --- A. 获取数据 ---
                recent = fetch_recent_json()
                target_event_id = get_current_event_for_server(recent, server_index=3)
                target_debug_hours = None

            if target_event_id is None:
                st.error("未找到活动 ID！")
            else:
                # --- B. 运行计算 ---
                # If debug hours is provided, pass it to DataHandler to freeze progress
                handler = DataHandler(target_event_id, debug_hours=target_debug_hours)
                handler.load_target_data()
                handler.find_similar_events()

                # --- C. 获取图片 ---
                new_img = handler.run_prediction(return_type='bytes')

                if new_img:
                    # --- D. 更新状态 ---
                    st.session_state['img_bytes'] = new_img
                    st.session_state['last_update_str'] = datetime.now().strftime('%H:%M:%S')
                    # 关键：更新时间戳
                    st.session_state['last_run_ts'] = time.time()

                    if manual_btn:
                        st.success(f"刷新成功！(Event {target_event_id})")
                else:
                    st.warning("计算完成，但没有生成图片数据。")

        except Exception as e:
            st.error(f"运行出错: {str(e)}")
            print(traceback.format_exc())

# ==========================================
# 7. 结果展示
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
        st.info("🐱 正在等待首次数据加载...")

with col_info:
    st.markdown("### 状态面板")
    st.write(f"最后更新: **{st.session_state['last_update_str']}**")
    
    # 显示倒计时
    curr_elapsed = time.time() - st.session_state['last_run_ts']
    next_run = max(0, AUTO_INTERVAL_SEC - curr_elapsed)
    
    st.progress(min(1.0, curr_elapsed / AUTO_INTERVAL_SEC), text=f"下一次自动刷新: {int(next_run)}秒后")
    
    st.caption("机制说明：每30分钟自动刷新，或点击按钮立即刷新。")