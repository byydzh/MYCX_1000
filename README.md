# mycx_1000

**[Launch Dashboard](https://mycx1000.streamlit.app)**

BangDream 国服活动档线预测项目。线上正式结果仍由 `Skeleton + Kalman
Filter` 生成；`behavior_pace_model` 是独立实验模型，当前不会替换线上默认
结果。旧的玩家粒子/mean-field 实验不再定义新模型路线。

## 线上数据源

线上选择 HHWX 时，数据层按 **HHWX → Bestdori** 自动路由活动索引、活动
详情、Tracker 和 T10 scale。任一来源返回了可计算数据就继续预测，页面只用
中性文字标出本轮实际来源；正常的自动切源不是预测警告。只有两个来源都无法
提供该组件时才报错并停止本次预测。

直接选择 Bestdori 时只读取 Bestdori。成功缓存按实际来源和路由隔离，失败
结果不缓存。

```bash
streamlit run app.py
python main_pipeline.py --event_id 312 --debug_hours 60
```

## 行为 pace 实验

实验目标是当前活动 master 数据声明的**真实奖励线**。调用方必须显式传入
这些奖励档位及其各自 Tracker 前缀；模型不把 T1000 固定为目标、尺度或参考
档，也不拿相邻档位代替缺失目标。

模型用三个非负行为分量描述累计 pace：

- `sustain`：随昼夜可用性持续游玩；
- `launch`：开局较强、随活动进程耗尽的常规追线投入；
- `deadline`：由剩余时间紧迫度与当前活动真实奖励边界共同调制的终局投入。

三个权重在单个活动内共享且和为 1，每个档位有自己的非负幅度。各档 Tracker
时间点可以完全不同，不要求为了组成横截面而对齐或取整。预测时，每个目标档
只使用它在 origin 之前的最后可见分数作为连续锚点。

完整方程和数据契约见
[`BEHAVIOR_MODEL_THEORY.md`](BEHAVIOR_MODEL_THEORY.md)。实现入口是
[`behavior_pace_model.py`](behavior_pace_model.py)，训练先验构建器是
[`behavior_pace_prior.py`](behavior_pace_prior.py)。

## 因果回放

任意历史事件都可以立刻按下式做 debug replay：

```text
tracker.time <= origin
```

模型在解析分数之前就丢弃 origin 之后的行；修改被遮罩后缀不得改变预测。
这项验证不要求等待新活动，也不要求额外的预测收据。`available_at` 可以用于
研究“本地采集器当时看见了什么”这一不同问题，但不能替代上述 API 时间戳
遮罩。

每个 origin 必须对所有候选模型使用同一事件、同一真实奖励档、同一可见
前缀和同一终值。实验结果至少应与线上 `Skeleton + Kalman Filter`、按历史
奖励线行为换算的 Skeleton 基线以及简单 causal persistence 并列报告；不能
只给孤立的 sMAPE，也不能用一个固定 T1000 预测冒充其他奖励档结果。

### 284–319 完成活动回放

2026-08-21 的固定回放覆盖 36 个完成活动、57 个真实奖励档和 468 个预定
origin。Event 291 的两个早期 origin 没有共同目标输入，因此 466 行可评估。
聚合顺序固定为：先对 origin 等权，再对活动内奖励档等权，最后对活动等权。

完整覆盖方法在同一 466 行上的结果如下；MAE 列单位为万游戏分：

| 方法 | 可评估覆盖 | MAPE | MAE | sMAPE | 平均偏差 |
|---|---:|---:|---:|---:|---:|
| Behavior pace | 466/466 | 21.64% | 194.26 万 | 25.16% | -20.85% |
| 最后两点非负斜率 | 466/466 | 25.09% | 214.91 万 | 28.69% | -15.60% |
| 累计均速外推 | 466/466 | 28.84% | 256.30 万 | 34.60% | -27.95% |
| Persistence（分数保持不变） | 466/466 | 61.45% | 506.50 万 | 95.40% | -61.45% |

Skeleton 依赖合法历史曲线，不能把失败行从 pace 中删掉后直接比较。因此另在
各 Skeleton 成功的完全相同 origin 上配对：

| 配对基线 | 成功 origin | Skeleton MAPE / MAE | 同支持集 pace MAPE / MAE |
|---|---:|---:|---:|
| 同档历史 Skeleton + KF | 418/466 | 15.99% / 128.52 万 | 21.15% / 194.02 万 |
| 奖励行为换算 Skeleton + KF | 371/466 | 18.85% / 142.94 万 | 22.85% / 207.49 万 |

这说明三分量结构确实增加了超过简单均速/斜率外推的信息：它在 36/36 个活动
上都胜累计均速，整体 MAPE 相对降低 24.97%。但它仍明显输给有合法历史可用
时的线上 Skeleton，并存在系统性低估，因此保持实验态，不能替换线上默认。
这项判断来自已经结束活动的严格时间遮罩，不需要等待未来活动。

pace 与朴素基线的目标前缀均来自带 SHA 的冻结 HHWX 缓存。Skeleton 另有
187 个 target-T10 origin 通过已授权的实时 HHWX→Bestdori 路由取得；结果记录
了 cutoff、数值和实际来源，但没有保存原始 API 响应，因此这些 scale 不能只
靠结果 JSON 离线逐字节重建。

可复现入口：

```bash
python scripts/evaluate_reward_tiers_318_319.py --event-id-range 284 319
```

## 事件等权训练先验

`behavior-pace-prior-v1` 的默认训练边界是明确冻结的 92 个事件
`192..283`。构建器只读取 `cache_dir/<event_id>.json`，不扫描目录、不猜测
替代文件名。所有事件使用共同测量支持：

```text
T50, T100, T300, T500, T1000, T2000
```

六档地位完全对称；T1000 只是其中一条。每个事件先独立拟合一次 simplex
权重，再对 92 个事件做算术平均，因此 Tracker 行数更多或档位分数更高的
事件不会获得更大先验权重。缓存中后来新增的其他档位不会改变这套固定支持。

构建器会把输入文件 SHA、源码 SHA、逐事件权重与诊断、排除原因和覆盖门槛
写入 JSON。示例：

```bash
python behavior_pace_prior.py \
  --cache-dir event_data/tier_surface_cache \
  --output configs/behavior_model/pace_prior_train192_283.json
```

该先验和模型仍处于实验态；284–319 的同目标、同 origin 对比没有通过替换
Skeleton 的门槛，因此不接入线上默认模型。历史遮罩回放已经足以作出这项
判断，不以“再等几个未来活动”作为日常开发前置条件。

## 聚焦测试

```bash
python -m pytest tests/test_behavior_pace_model.py tests/test_behavior_pace_prior.py
python -m pytest tests/test_api_sources.py tests/test_baseline_fallback.py
```
