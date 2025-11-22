# config.py

# ==========================================
# 预测器默认配置字典 (Default Configuration)
# ==========================================
DEFAULT_CONFIG = {
    # --- 节律与模型参数 ---
    'weekend_multiplier': 1.1,      # 周末增强系数 (1.0 = 无增强, >1.0 = 周末加速)
    'panic_ease_power': 1.0,        # 恐慌期缓动指数 (控制最后时刻加速曲线的弯曲程度)
    'panic_scaler': 1.1,            # 恐慌期最小加速倍数
    
    # --- 相似活动搜索 ---
    'similar_count': 5,             # 寻找相似历史活动的数量
    
    # --- 预测逻辑: 对比窗口 ---
    't_start_cmp': 6.0,             # 对比起始时间 (小时，跳过开局数据不稳定)
    't_end_cap': 72.0,              # 对比结束时间上限 (小时，防止中后期数据干扰)
    
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
}