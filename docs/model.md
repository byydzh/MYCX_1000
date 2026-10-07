# Skeleton + Kalman Filter

返回 [README](../README.md)。本文说明线上 `skeleton_kf` 模型；运行、训练和回放
入口见 [开发说明](development.md)。

## 输入与输出

当前活动输入包括起止时间、类型、目标档 Tracker 和 T10 Tracker；历史参考使用
较早活动的同类型、同档数据。静态输入为参数预设与
[`base_speed_distribution.json`](../base_speed_distribution.json) 中的工作日/周末小时分布。
输出为结束分数、累计分曲线、速度曲线、强度 Ratio 和拟合参数。

T500、T1000、T1500、T2000 各自寻找历史并运行预测，生产路径采用精确档位数据。
单档失败不影响其他档；共享元数据或 T10 尺度不可用会阻止本轮预测。

## 计算过程

### 数据与速度尺度

目标档累计分为 `S(t)`，相邻记录转为每分钟速度：

```text
v(t_i) = max(0, ΔS / Δminutes)
V10 = mean(top-3 valid T10 interval speeds)
u(t) = v(t) / V10
```

目标速度中的非有限值与负值清零；T10 有效速度范围为 `(0, 1,000,000)`。
`V10` 是归一化速度尺度。若首个正分记录在活动开始后 24 小时内，预处理会将其
向下取整到整点作为有效起点，用于近似修正维护延迟。

归一化速度再除以 UTC+8 小时分布、中国节假日和周末因子，得到去节律骨架速度。
预测阶段乘回节律，并在终局阶段叠加渐进活跃度增益。

### 历史形状与当前强度

历史选择条件为 `event_id < target_event_id`、类型相同、不在 `ignore_event_ids`、
具有目标档 Tracker 和有效 T10 尺度。按 ID 从新到旧扫描至多 `similar_count + 3`
个候选，保留至多 `similar_count` 个，默认 5 个。

每场历史独立拟合，成功参数等权平均：

```text
g(t) = Base + A·t + B·t² + B_end·E_panic(t; T_panic, T_total)
```

`E_panic` 是在活动末段上升的非负终局项。没有可拟合历史时预测失败。

当前强度 Ratio 混合“去节律骨架速度比”和“原始归一化速度均值比”，随着观测
增加提高后者权重。比较窗口内先做 2σ 清理，再限制到 `ratio_min..ratio_max`。
不同时长活动之间混合绝对小时窗口与相对进度窗口，并对 `A/B` 做时长归一化。

```text
Base, A, B *= Ratio
B_end      *= Ratio^1.1
```

### 局部重拟合与实时修正

在线重拟合只更新 `Base/A/B`，以历史参数为正则中心，保留有符号参数边界。
默认从第 6 小时开始，取最近 48 小时，融合权重随观测增加，最高 35%。

Kalman 状态为 `[scale, trend]`，观测为“真实分数增量 / 骨架预测增量”。观测做
上下截断并使用自适应噪声，未来趋势按半衰期衰减；观测不足时修正系数保持为 1。
终局阶段将 scale 平滑拉回 1，控制它与骨架追分项的叠加。

### 速度约束与积分

预测速度相对 T10 尺度经过三段压缩：默认从 0.50 开始衰减，0.65 后加强，
硬顶为 0.80。从最后可见分数向活动结束积分，保持曲线在锚点连续：

```text
S_hat(T) = S(t_anchor) + integral[t_anchor, T] v_hat(t) dt
```

## 参数入口

基础值定义于 [`config.py`](../config.py)，网页通过 preset 覆盖；当前网页默认
[`learned_notebook`](../configs/models/skeleton_kf/learned_notebook.json)。
以下字段说明用途，具体数值以所选配置为准。

| 参数组 | 字段 | 作用 |
|---|---|---|
| 节律与终局 | `weekend_multiplier`, `panic_ease_power`, `panic_scaler` | 周末与终局增益 |
| 历史选择 | `similar_count`, `ignore_event_ids` | 参考活动数量和排除项 |
| 强度比较 | `t_start_cmp`, `t_end_cap`, `ratio_min`, `ratio_max` | 比较窗口和强度范围 |
| 时长对齐 | `duration_*` | 窗口映射与骨架参数归一化 |
| 局部拟合 | `refit_*` | 观测窗口、正则、参数边界和融合权重 |
| Kalman | `kf_*` | 状态、噪声、截断与趋势衰减 |
| 速度压缩 | `smooth_thresh1`, `smooth_thresh2`, `smooth_hard_cap` | 速度上限 |

## 适用限制

模型独立预测各条固定档线，不约束跨档排序，也不提供概率预测区间。昼夜分布、
终局函数、维护起点修正和 T10 尺度均为结构假设。历史活动目前按 ID 选择；国服
活动 ID 不总与真实时间顺序一致，严格回测还需检查历史结束时间早于目标开始时间。
