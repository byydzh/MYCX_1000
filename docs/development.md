# 开发说明

返回 [README](../README.md)。模型方程见 [Skeleton 模型说明](model.md)。

## 模块边界

`app.py` 和 `main_pipeline.py` 各自编排数据读取和预测，共用 `PredictionEngine`、
`SeasonalityHandler`、`CosineModeler` 和 `EventData`。目前两入口与
`tuner/offline_evaluator.py` 仍分别维护数据预处理，修改时需要核对三个调用方。
`scripts/evaluate_reward_tiers_318_319.py` 还复用了 CLI 的私有预处理方法。

`data_source.py` 管理 HHWX/Bestdori 请求、来源路由、短期进程缓存、历史选择与
T10 尺度。`plotly_viz.py` 和 `visualizer.py` 分别用于网页和 CLI 输出。

实验模型为 `behavior_pace_model.py`、`behavior_cohort_model.py`，先验构建位于
`behavior_pace_prior.py`；`tier_surface.py` 处理多档观测，`scripts/` 提供收集、
评估和报告入口。这些实验没有注册为线上模型。

## 参数训练

训练与 benchmark 会读取或准备历史数据，首次运行可能联网。明确指定候选输出，
保留现有线上 preset：

```powershell
python -m tuner.train --tier 1 --history-count 80 --use-formal-holdout-split --output configs/models/skeleton_kf/learned_candidate.json
python -m tuner.global_benchmark --prepare-cache --model-id skeleton_kf --preset-id learned_notebook
```

训练命令的 `--tier` 指参数搜索层级，当前仅支持 `1`；目标榜线仍为 T1000。
`tuner/offline_evaluator.py` 负责快照评估，`tuner/global_benchmark.py` 比较 preset。
新候选经评估后再决定是否使用；网页默认 preset 由 `app.py` 的
`DEFAULT_PRESET_BY_MODEL` 指定。新增 JSON 可供网页选择，但不会自动成为默认项。

## 历史回放

网页 Debug 支持指定 Event ID、相对小时或 UTC+8 时间，以及未来假设点。
假设点用于 what-if 模拟，评估时应关闭。

现有入口的时间口径需要注意：

- CLI 的 `--debug_hours` 仅截断目标档，T10 仍取全场数据。
- 网页会截断目标档和 T10，但相对小时的 T10 截止按原始开始时间计算，目标档
  截止按维护修正后的起点计算；发生起点修正时两者可能不同。
- `tuner/offline_evaluator.py` 使用缓存中预先算好的 scale，仅截断目标档数据。
- 生产历史搜索按活动 ID 过滤。严格回测应另按真实起止时间排除尚未结束的历史。

因此上述 Debug 和 tuner 指标不能直接当作所有输入都严格按时刻冻结的回测。
正式评估需要统一目标档、T10 及其他输入的截止时间，并固定历史样本和原始响应。

## 行为模型实验

[Pace v1](../BEHAVIOR_MODEL_THEORY.md) 与 [Cohort 实验](../EXPERIMENT_20260908.md)
分别说明模型方程和训练边界。Cohort 的[冻结报告](../event_data/cohort_experiment_20260908/report.md)
包含候选选择、逐阶段结果和失败记录；其整体误差改善不足以支持替换线上模型。

从已提交结果重新生成报告：

```powershell
python scripts/report_behavior_cohorts.py
```

完整重跑还需要未入库的 `event_data/tier_surface_cache/192.json` 至 `319.json`、
`event_data/reward_tier_evaluation_284_319_36events_224cea9f.json`，以及已入库的
`configs/behavior_model/pace_prior_train192_283.json`。已有匹配数据时按顺序运行：

```powershell
python scripts/experiment_behavior_cohorts.py screen
python scripts/experiment_behavior_cohorts.py repair
python scripts/experiment_behavior_cohorts.py confirm
python scripts/experiment_behavior_cohorts.py replay
python scripts/report_behavior_cohorts.py
```

这些命令会覆盖对应实验产物。冻结结果的原始源码版本为 `bfd5259`，结果中的
`source_sha256` 对应生成时文件字节；后续代码修改不应改写旧结果的来源摘要。
新克隆可阅读冻结结果并重建报告，完整重跑需要上述本地输入。旧 Skeleton 对照
复用原评估结果，部分原始 T10 响应未保存，因此其从头复算仍有边界。

## 测试

先安装开发所需的 pytest：`python -m pip install pytest`。
只选择与本次改动相关的文件，例如：

| 修改范围 | 测试文件（位于 `tests/`） |
|---|---|
| 数据源与路由 | `test_api_sources.py`, `test_baseline_fallback.py` |
| Skeleton 数学 | `test_duration_alignment.py`, `test_refit_guardrails.py` |
| 配置 | `test_config_loading.py`, `test_backward_compat.py` |
| 离线评估 | `test_offline_evaluator.py`, `test_global_benchmark.py` |
| Cohort 模型与缓存 | `test_behavior_cohort_model.py` |

```powershell
python -m pytest tests/test_behavior_cohort_model.py -q
```
