# mycx_1000

BangDream 国服活动档线实时预测系统。项目当前唯一的线上正式模型是
**Skeleton + Kalman Filter**：它从同类型、同档位的历史活动学习完整速度骨架，
再利用当前活动已经发生的 Tracker 数据估计强度、重拟合局部形状，并通过卡尔曼
滤波修正实时偏差，最终给出活动结束分数与完整预测曲线。

- 在线面板：[mycx1000.streamlit.app](https://mycx1000.streamlit.app)
- 服务区：国服（server `3`）
- 网页可选档位：T500、T1000、T1500、T2000
- 正式模型：`skeleton_kf`
- 网页默认参数：`learned_notebook`

`behavior_pace_model` 是离线研究用的奖励线实验模型。目前它没有通过替换线上
基线的门槛，不参与网页正式预测。

## 项目要解决的问题

活动档线不是匀速增长：开局热度、昼夜作息、活动类型、活动长度和终局追分都会
改变速度。仅拿当前平均速度线性外推，通常无法同时处理这些结构。正式基线将问题
拆成四层：

1. **历史形状**：同类型活动通常共享可复用的开局、平稳期和终局加速结构；
2. **当前强度**：当前活动相对历史活动可能整体更热或更冷；
3. **实时偏差**：活动进行中会持续偏离原始骨架；
4. **物理尺度**：目标档速度不能脱离同场 T10 活跃度无限增长。

输入与输出如下：

| 类别 | 内容 |
|---|---|
| 当前活动输入 | 活动起止时间、活动类型、目标档 Tracker、T10 Tracker |
| 历史输入 | Event ID 更早的同类型活动、相同目标档 Tracker、各活动 T10 scale |
| 静态输入 | 工作日/周末 24 小时速度分布、模型 preset |
| 输出 | 最终档线分数、逐时累计分曲线、预测速度曲线、强度 Ratio、拟合参数 |

## 正式基线：Skeleton + Kalman Filter

### 1. 数据清洗与时间轴

目标档累计分记作 `S_r(t)`。Tracker 相邻记录先换算为每分钟速度：

```text
v_r(t_i) = max(0, ΔS_r / Δminutes)
```

非有限值与负速度会被清零。若活动开始后的首个正分记录落在 24 小时内，系统会
把该记录向下取整到整点，并将它作为维护延迟后的有效起点。这是处理开服维护造成
的时间轴偏移的启发式规则。

### 2. T10 速度尺度

T10 scale 不是 T10 最终档线，而是同一活动头部玩家的可达速度尺度。系统从 T10
相邻记录计算 `point/minute`，保留 `0 < speed < 1,000,000` 的有效增量，并取最
大的三个增量均值：

```text
V10 = mean(top-3 valid T10 interval speeds)
u_r(t) = v_r(t) / V10
```

`u_r(t)` 是无量纲速度。没有正且有限的 T10 scale 时，正式预测会停止，不会拿
常数代替。

### 3. 去除昼夜节律

`base_speed_distribution.json` 提供工作日与周末的 24 小时活动分布。系统结合
UTC+8、本地星期与中国节假日规则，将目标档和历史档的归一化速度先除去昼夜波动，
使骨架拟合关注活动进程本身，而不是“凌晨比晚上慢”这一重复现象。

终局预测时会重新乘回节律；进入 panic 阶段后，再叠加渐进的终局活跃度增益。

### 4. 同档历史骨架

线上预测对每个目标档独立寻找历史：

- `event_id < target_event_id`；
- 活动类型相同；
- 不在 `ignore_event_ids`；
- 必须有该**精确档位**的 Tracker 与有效 T10 scale；
- 按活动 ID 从新到旧扫描至多 `similar_count + 3` 个候选，返回至多
  `similar_count` 个（默认 5 个）；
- 生产路径禁用相邻排名插值。

每个成功历史活动分别拟合下式，再对历史参数等权平均：

```text
g(t) = Base + A·t + B·t² + B_end·E_panic(t; T_panic, T_total)
```

其中 `E_panic` 是只在活动末段显著上升的非负终局项，参数为
`Base / A / B / B_end / T_panic`。如果没有任何可拟合的合法历史，模型直接报告
失败；它不会输出固定默认终值。

### 5. 当前活动强度 Ratio

系统在当前可见比较窗口内，同时估计两种强度比：

- 去节律后的骨架速度强度比；
- 原始 T10 归一化速度均值比。

随着可见数据增加，权重从历史骨架强度逐渐转向当前归一化速度；每个活动比较
窗口内的骨架速度先做 2σ 清理，最终 Ratio 再限制在 `ratio_min..ratio_max`。
目标活动与历史活动长度不同时，绝对小时窗口和相对进度窗口会连续混合，`A/B`
也会按时长差异归一化。

Ratio 对参数的主要作用为：

```text
Base, A, B  *= Ratio
B_end       *= Ratio^1.1
```

### 6. 在线局部重拟合

历史骨架提供先验以后，系统只用当前活动已经可见的数据重拟合 `Base/A/B`。默认
从第 6 小时以后开始，关注最近 48 小时，以历史参数为正则中心，并限制参数只能在
有意义的有符号区间内移动。

重拟合结果按可见窗口长度逐渐增加权重，默认最多占 35%，因此它负责纠正局部形状，
不会直接推翻历史终局结构。

### 7. 实时 Kalman 修正

卡尔曼滤波状态为：

```text
x = [scale, trend]
```

观测量是每个时间步的“真实分数增量 / 骨架预测增量”。测量会做上下截断，并按
增量大小自适应调整噪声；`trend` 在未来按半衰期衰减，避免短时异常被永久外推。
可见数据过少、时间过早或观测不合法时，实时修正保持为 `1`。

进入终局 panic 阶段后，系统会把 KF 的 scale 平滑拉回 `1`，避免实时乘数与
`B_end`、panic 节律重复放大同一次终局冲刺。

### 8. 速度约束与终值积分

预测速度相对 T10 scale 经过三段式压缩；默认从 `0.50` 开始轻度衰减，`0.65`
后加强衰减，最终硬顶为 `0.80`。最后从目标档最后可见分数开始积分到活动结束：

```text
S_hat(T) = S(t_origin) + integral[t_origin, T] v_hat(t) dt
```

这保证预测曲线与当前真实分数连续，而不是重新从零生成一条无关曲线。

## 多档预测语义

T500、T1000、T1500、T2000 会分别抓取目标曲线、寻找同档历史并独立运行完整
Skeleton+KF。当前线上基线不会用 T1000 冒充其他档，也不会在不同档位之间做
连续排名插值。

因此单个档位缺数据只会让该档失败；其他档位仍可继续。共享活动元数据、T10 scale
失败，或所有档位都失败时，本轮预测才整体失败。

## 数据源与正常路由

项目支持 HHWX 与 Bestdori 两套公开 API。默认来源是 HHWX。

### Streamlit 网页

网页选择 HHWX 时，活动索引、当前活动、活动详情、目标档 Tracker、历史 Tracker
和 T10 scale 都按组件执行：

```text
HHWX -> Bestdori
```

HHWX 的 HTTP、JSON、结构或有效数据失败后，才读取 Bestdori；两者都不能提供该
组件时才报错。一次预测可以由不同来源共同提供不同组件，页面只中性显示本轮实际
来源；成功切换来源不是模型质量警告。

直接选择 Bestdori 时不会反向请求 HHWX。成功的活动索引和实时 T10 scale 使用
60 秒进程内缓存，失败结果不缓存。生产路径不启用档位插值，也不会在模型失败后
改用另一种预测模型。

### CLI

`main_pipeline.py` 当前是所选 provider 的严格模式，没有启用网页端的
HHWX→Bestdori 自动路由。它也直接使用 `DEFAULT_CONFIG`，不会自动加载网页默认的
`learned_notebook` preset；初始化节律处理器时目前也没有传入
`panic_ease_power`。两种入口适合不同用途，不应把结果视为完全同构。

## 运行项目

### 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

主要依赖为 NumPy、Pandas、SciPy、Requests、Matplotlib、Plotly、Streamlit 与
`chinesecalendar`。

### 启动网页

```powershell
python -m streamlit run app.py
```

页面首次加载会运行一次预测，之后由“立即运行预测”按钮触发；当前没有后台定时
自动刷新。配置修改只保存在本次 Streamlit 会话，不会改写 preset JSON。

### 运行 CLI

```powershell
python main_pipeline.py --event_id 312 --debug_hours 60 --api-source hhwx --tiers "500,1000,1500,2000"
```

参数：

| 参数 | 说明 |
|---|---|
| `-e, --event_id` | 目标活动 ID；省略时尝试选择当前活动 |
| `-d, --debug_hours` | 从有效活动起点起保留多少小时的目标数据 |
| `--api-source` | `hhwx` 或 `bestdori` |
| `-t, --tiers` | 逗号分隔的固定档位；默认 `1000` |

CLI 会计算所有成功档位，但当前 Matplotlib 输出只绘制第一个成功档位。

## 网页 Debug 与历史回放

网页 Debug 模式支持指定历史 Event ID，并按相对小时或 UTC+8 时刻冻结。目标档
Tracker 与目标 T10 都会裁掉各自截止时间之后的记录：

```text
tracker.time <= cutoff
```

还可以添加未来人工点，生成 what-if 虚拟路径。人工点只属于明确的干预模拟，不会
混入普通预测。

当前网页先按官方开始时间计算 T10 cutoff，随后目标档时间轴可能经过维护起点修正；
存在维护延迟时，两者的绝对截止时刻可能不完全相同。因此网页 Debug 适合交互诊断，
不作为严格 benchmark。公共 Tracker archive 也没有逐行 `available_at`，这里的
遮罩只表达 API 时间戳，不是第三方保存的“当时首次公开时间”。

CLI 的 `--debug_hours` 还会在遮罩目标档以前预取整场 T10 scale，同样不应用于
严格因果比较。后述统一评估脚本会先确定一个绝对 origin，再对所有目标输入执行
同一 `tracker.time <= origin` 规则。

## 配置与 preset

模型注册表位于 [`configs/models.json`](configs/models.json)，当前只注册
`skeleton_kf`。参数文件位于 `configs/models/skeleton_kf/`：

| Preset | 用途 |
|---|---|
| `learned_notebook.json` | Streamlit 默认；由 `tuner/train.py` 生成的学习参数 |
| `default.json` | 早期人工稳定参数，也是 `DEFAULT_CONFIG` 的兼容基线 |

`learned_notebook` 自带的训练元数据显示 73 个训练活动、25 个测试活动和 5 个
external holdout 活动；其 external-holdout relative MSE 为 `0.059616`，相对
当时人工配置改善 `1.37%`。这是 preset 文件保存的训练记录，不等同于 MAPE，也
不替代后续历史回放。

真正参与引擎计算的主要参数组：

| 参数组 | 代表字段 | 作用 |
|---|---|---|
| 节律与终局 | `weekend_multiplier`, `panic_ease_power`, `panic_scaler` | 恢复昼夜/周末节律与终局活跃度 |
| 历史选择 | `similar_count`, `ignore_event_ids` | 控制同类、同档历史样本 |
| 比较窗口 | `t_start_cmp`, `t_end_cap` | 跳过开局噪声并限制强度比较范围 |
| 时长对齐 | `duration_*` | 混合绝对小时与相对进度，归一化骨架参数 |
| Ratio | `ratio_min`, `ratio_max` | 限制当前/历史强度倍数 |
| 在线 refit | `refit_*` | 控制当前数据的拟合窗口、正则、边界和融合权重 |
| Kalman | `kf_*` | 控制 scale/trend 状态、过程噪声、测量噪声与截断 |
| 速度压缩 | `smooth_thresh1`, `smooth_thresh2`, `smooth_hard_cap` | 限制目标档速度相对 T10 的比例 |

`scale_min/max` 与 `corr_min/max` 仍存在于旧配置和网页控件中，但当前
`PredictionEngine` 没有消费它们；不要把这四个字段当成已生效的预测约束。

## 训练与评估

### Skeleton 参数训练

`tuner/` 包含数据缓存、特征、离线评估、优化器、训练脚本与全局 benchmark。
一个明确指定输出文件的训练示例：

```powershell
python -m tuner.train --tier 1000 --history-count 80 --use-formal-holdout-split --output configs/models/skeleton_kf/learned_candidate.json
```

比较现有 preset：

```powershell
python -m tuner.global_benchmark --prepare-cache --model-id skeleton_kf --preset-id learned_notebook
```

训练产物不会自动成为网页默认 preset；需要显式审阅、命名并更新入口选择。

### 完成活动的严格 rolling-origin 回放

统一评估器从每个活动声明的真实奖励档生成 origin，用 `tracker.time <= origin`
遮罩目标输入，并要求历史活动在目标活动开始前结束。评估缓存不会随 Git 仓库分发；
首次运行前需要用 `collect_tier_surface_cache.py` 采集目标事件范围：

```powershell
python scripts/collect_tier_surface_cache.py --api-source hhwx --min-event-id 284 --max-event-id 319
python scripts/evaluate_reward_tiers_318_319.py --event-id-range 284 319
```

尽管评估脚本文件名保留了最初的 `318_319`，当前 CLI 支持显式事件列表与闭区间
范围。评估会并列输出线上同档 Skeleton、奖励行为换算 Skeleton、实验 pace、
最后两点斜率、累计均速与 persistence；聚合顺序是 origin 等权、活动内奖励档
等权、活动等权。

“奖励行为换算 Skeleton”只是一种评估历史选择变体：它按奖励类别把目标奖励线
映射到历史活动自己的精确奖励档。网页生产路径仍是同档历史 Skeleton。

在 284–319 的固定回放中，线上同档 Skeleton 在其 418 个成功 origin 上 MAPE
为 `15.99%`；同一支持集的实验 pace 为 `21.15%`。完整 466 个共同输入上，pace
为 `21.64%`，最后两点斜率为 `25.09%`，累计均速为 `28.84%`。因此正式模型继续
使用 Skeleton+KF；这些历史回放可以立即重复，不需要等待未来活动才能评价模型。

## 行为 pace 实验

[`behavior_pace_model.py`](behavior_pace_model.py) 研究另一条路线：直接以活动
master 数据中的真实奖励线为目标，用 `sustain / launch / deadline` 三个非负
分量描述共享活动节奏，再为每个档位拟合独立幅度。它支持异步 Tracker 时间点，
不把 T1000 固定为目标或尺度。

训练先验由 [`behavior_pace_prior.py`](behavior_pace_prior.py) 在事件 `192..283`
的固定六档支持上按事件等权构建；实现方程、识别假设和数据契约见
[`BEHAVIOR_MODEL_THEORY.md`](BEHAVIOR_MODEL_THEORY.md)。该模型当前只用于离线
实验，不在 `configs/models.json` 注册，也不替代线上结果。

## 代码结构

| 路径 | 职责 |
|---|---|
| [`app.py`](app.py) | Streamlit 多档实时预测、Debug 与交互图入口 |
| [`main_pipeline.py`](main_pipeline.py) | Skeleton+KF 命令行流水线 |
| [`data_source.py`](data_source.py) | HHWX/Bestdori API、来源路由、Tracker 与历史活动选择 |
| [`domain_models.py`](domain_models.py) | `EventMeta`、`EventData`、`PredictionResult` |
| [`math_models.py`](math_models.py) | 昼夜节律处理与历史骨架函数 |
| [`prediction_engine.py`](prediction_engine.py) | Ratio、时长对齐、refit、KF、平滑与积分 |
| [`plotly_viz.py`](plotly_viz.py) | 网页交互式多档图 |
| [`visualizer.py`](visualizer.py) | CLI Matplotlib 图 |
| [`config.py`](config.py) | 数据源、模型注册、默认参数与 preset 加载 |
| [`configs/models/skeleton_kf`](configs/models/skeleton_kf) | 正式模型参数文件 |
| [`base_speed_distribution.json`](base_speed_distribution.json) | 工作日/周末 24 小时速度分布 |
| [`tuner`](tuner) | Skeleton 参数训练、离线评估与 benchmark |
| [`tier_surface.py`](tier_surface.py) | 固定档位 as-of 快照与质量检查工具 |
| [`scripts/collect_tier_surface_cache.py`](scripts/collect_tier_surface_cache.py) | 多档 Tracker 与奖励元数据缓存采集 |
| [`scripts/evaluate_reward_tiers_318_319.py`](scripts/evaluate_reward_tiers_318_319.py) | 多模型 rolling-origin 统一评估 |
| [`behavior_pace_model.py`](behavior_pace_model.py) | 实验 pace 点预测模型 |
| [`tests`](tests) | 数据源、基线、配置、回放与实验模型的聚焦测试 |

主调用关系：

```text
app.py / main_pipeline.py
    -> config.py + configs/models/skeleton_kf/*.json
    -> data_source.py
    -> domain_models.py
    -> math_models.py
    -> prediction_engine.py
    -> plotly_viz.py / visualizer.py
```

## 聚焦测试

数据源与网页 provider 路由：

```powershell
python -m pytest tests/test_api_sources.py tests/test_baseline_fallback.py
```

Skeleton 数学、配置与离线评估：

```powershell
python -m pytest tests/test_duration_alignment.py tests/test_refit_guardrails.py tests/test_config_loading.py tests/test_backward_compat.py tests/test_offline_evaluator.py tests/test_global_benchmark.py
```

实验 pace 与统一评估器：

```powershell
python -m pytest tests/test_behavior_pace_model.py tests/test_behavior_pace_prior.py tests/test_collect_tier_surface_cache.py tests/test_reward_tier_evaluator.py
```

## 已知限制

- 公共 API 只提供固定档位，不支持任意连续排名；网页目前只暴露四条常用档线。
- Skeleton 必须有有效 T10 scale 和至少一个可拟合同档历史；缺失时明确失败。
- 各档独立建模，当前线上基线不表达跨档竞争或活动奖励之间的联动。
- 模型没有概率预测区间；输出是点预测与确定性曲线。
- 昼夜分布、终局函数、维护起点修正和 T10 硬顶仍是结构性假设。
- 公共 archive 没有逐行 `available_at`，历史 Debug 只保证 Tracker 时间戳遮罩。
- 网页与 CLI 在 preset、provider 路由和 `panic_ease_power` 传递上尚未完全同构。
- 在线数据只有短期进程缓存；若没有冻结原始 API 响应，历史数值不保证逐字节复现。

仓库当前未声明开源许可证。
