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
from config import DEFAULT_CONFIG
from data_source import BestdoriDataSource
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

# ==========================================
# 3. 侧边栏控制
# ==========================================
st.sidebar.header("控制台 🎮")
manual_btn = st.sidebar.button("⚡ 立即运行预测", type="primary")

st.sidebar.markdown("---")
with st.sidebar.expander("参数设置", expanded=False):
    st.caption("调整下列参数将覆盖 config.py 的默认值")
    
    # 模型参数
    st.markdown("**模型参数**")
    weekend_mult = st.slider("周末增强系数", 0.8, 1.5, DEFAULT_CONFIG.get('weekend_multiplier', 1.0), 0.05)
    panic_scaler = st.slider("恐慌期最小加速倍数", 1.0, 3.0, DEFAULT_CONFIG.get('panic_scaler', 1.1), 0.05)
    panic_ease_power = st.slider("恐慌期缓动指数", 0.1, 5.0, DEFAULT_CONFIG.get('panic_ease_power', 1.0), 0.1)
    similar_count = st.number_input("参考历史活动数", 1, 10, DEFAULT_CONFIG.get('similar_count', 5))

    # 阈值与限制
    st.markdown("**阈值与限制**")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        ratio_min = st.number_input("Ratio Min", value=DEFAULT_CONFIG.get('ratio_min', 0.25), step=0.05)
        scale_min = st.number_input("Scale Min", value=DEFAULT_CONFIG.get('scale_min', 0.5), step=0.1)
        t_start_cmp = st.number_input("对比窗口起始", value=DEFAULT_CONFIG.get('t_start_cmp', 6.0), step=0.5)
    with col_p2:
        ratio_max = st.number_input("Ratio Max", value=DEFAULT_CONFIG.get('ratio_max', 4.0), step=0.1)
        scale_max = st.number_input("Scale Max", value=DEFAULT_CONFIG.get('scale_max', 2.0), step=0.1)
        t_end_cap = st.number_input("窗口结束上限", value=DEFAULT_CONFIG.get('t_end_cap', 72.0), step=1.0)

    # 回测与平滑
    st.markdown("**回测与平滑**")
    corr_min = st.number_input("24h 修正下限", value=DEFAULT_CONFIG.get('corr_min', 0.6), step=0.05)
    corr_max = st.number_input("24h 修正上限", value=DEFAULT_CONFIG.get('corr_max', 1.6), step=0.05)
    smooth_thresh1 = st.number_input("平滑阈值 1", 0.0, 1.0, DEFAULT_CONFIG.get('smooth_thresh1', 0.5), 0.01)
    smooth_thresh2 = st.number_input("平滑阈值 2", 0.0, 1.0, DEFAULT_CONFIG.get('smooth_thresh2', 0.65), 0.01)
    smooth_hard_cap = st.number_input("绝对硬顶", 0.0, 1.0, DEFAULT_CONFIG.get('smooth_hard_cap', 0.8), 0.01)

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
    current_config = DEFAULT_CONFIG.copy()
    current_config.update({
        'weekend_multiplier': weekend_mult,
        'panic_scaler': panic_scaler,
        'panic_ease_power': panic_ease_power,
        'similar_count': int(similar_count),
        'ratio_min': ratio_min, 'ratio_max': ratio_max,
        'scale_min': scale_min, 'scale_max': scale_max,
        't_start_cmp': t_start_cmp, 't_end_cap': t_end_cap,
        'corr_min': corr_min, 'corr_max': corr_max,
        'smooth_thresh1': smooth_thresh1, 'smooth_thresh2': smooth_thresh2,
        'smooth_hard_cap': smooth_hard_cap
    })

    with st.spinner(f"🐱 ({trigger_reason}) 正在计算中..."):
        ds = BestdoriDataSource()
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
                        target_eid, target_data.meta.event_type, count=int(similar_count)
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
                        weekend_multiplier=float(weekend_mult),
                        panic_scaler=float(panic_scaler),
                        panic_ease_power=float(panic_ease_power)
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