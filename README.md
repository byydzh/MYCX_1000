# mycx_1000

**[Launch Dashboard](https://mycx1000.streamlit.app)**

`mycx_1000` 是一个面向 BangDream / Bestdori 国服活动 T1000 线的实时预测面板。当前仓库已经进入“单模型、多配置、可继续扩展新模型”的结构。

## Current State

- 当前正式模型：`Skeleton + Kalman Filter`
- 当前正式配置：
  - `早期手动参数配置`
  - `最新基于学习的参数配置`
- 前端默认进入：`最新基于学习的参数配置`
- `301+` 事件只作为 external holdout，不进入训练

## Benchmark

当前正式 benchmark 约定：

- 训练池：`<=300`
- external holdout：`301+`
- 当前 holdout：`301, 302, 303, 312, 313`
- 忽略事件：`[297, 298]`

基于最近一次正式对比，`最新基于学习的参数配置` 已经优于 `早期手动参数配置`：

| 配置 | Holdout Relative MSE | Holdout Curve Relative MSE | Holdout Objective Loss |
|---|---:|---:|---:|
| 早期手动参数配置 | 0.04582 | 0.01747 | 0.04757 |
| 最新基于学习的参数配置 | 0.03223 | 0.01602 | 0.03383 |

说明：

- 最终值误差更低
- 曲线拟合误差也更低
- 综合目标同样更优

## Main Files

- `app.py`: Streamlit 前端入口
- `config.py`: 默认配置、模型注册与 preset 加载
- `prediction_engine.py`: 当前主预测引擎
- `main_pipeline.py`: CLI 入口
- `configs/models.json`: 模型注册表
- `configs/models/skeleton_kf/default.json`: 早期手动参数配置
- `configs/models/skeleton_kf/learned_notebook.json`: 最新基于学习的参数配置

训练与评估相关：

- `tuner/train.py`: 离线训练入口
- `tuner/global_benchmark.py`: 跨配置 benchmark
- `tuner/train_workbench.ipynb`: 主训练 notebook
- `tuner/train_validation_light.ipynb`: 轻量验证 notebook

## Future Direction

当前仓库不再把自己限定成“只有一个模型的一组参数”。后续可以继续在 `configs/models.json` 下注册新模型，并为每个模型维护各自的配置、训练入口和 benchmark。

也就是说，未来这里可以同时容纳：

- 当前 `Skeleton + Kalman Filter`
- 新的混合模型
- 更偏机器学习的纯数据驱动模型

## Run

本地前端：

```bash
streamlit run app.py
```

CLI：

```bash
python main_pipeline.py --event_id 312 --debug_hours 60
```

测试：

```bash
python -m pytest tests
```
