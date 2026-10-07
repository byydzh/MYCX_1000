# mycx_1000

预测 BanG Dream! 国服活动结束时的档线分数，提供累计分和速度曲线。

**在线使用：[mycx1000.streamlit.app](https://mycx1000.streamlit.app)**

支持 T500、T1000、T1500、T2000。线上模型为 **Skeleton + Kalman Filter**：
从同类型、同档位的历史活动拟合速度形状，再用当前观测修正强度和趋势。

## 本地运行

在仓库根目录执行。以下为 Windows PowerShell 命令：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

页面首次加载自动预测，之后点击“立即运行预测”更新。侧栏可选档位、参数预设和
Debug 模式；参数修改仅保存在当前会话。

命令行运行：

```powershell
python main_pipeline.py --api-source hhwx --tiers "500,1000,1500,2000"
```

| 参数 | 用途 |
|---|---|
| `--event_id`, `-e` | 指定活动；省略时选择当前活动 |
| `--tiers`, `-t` | 逗号分隔的固定档位，默认 `1000` |
| `--api-source` | `hhwx`（默认）或 `bestdori` |
| `--debug_hours`, `-d` | 只保留目标档前若干小时的数据，用于调试 |

CLI 计算全部请求档位，绘图仅显示第一个成功档位。
`--debug_hours` 尚未同步截断 T10 输入，不能直接用作严格历史回测。

## 数据与配置

网页默认读取 HHWX，数据不可用时转向 Bestdori，并显示实际来源；选择 Bestdori
时只读取该来源。CLI 按 `--api-source` 指定来源读取。

| 入口 | 默认参数 |
|---|---|
| 网页 | [`learned_notebook.json`](configs/models/skeleton_kf/learned_notebook.json) |
| CLI | [`config.py`](config.py) 中的 `DEFAULT_CONFIG` |

网页可切换到 [`default.json`](configs/models/skeleton_kf/default.json)。
预设文件位于 `configs/models/skeleton_kf/`，模型注册表位于
[`configs/models.json`](configs/models.json)。

Debug 模式可指定历史活动、冻结时间和未来假设点。回放的时间口径与注意事项见
[开发说明](docs/development.md#历史回放)。

## 预测边界

- 各档独立建模，需要有效 T10 速度尺度和至少一个可拟合的同档历史活动；
  单档数据不足会报告失败。
- 输出是点预测，没有概率预测区间；昼夜节律、终局追分和速度上限属于模型假设。
- 数据来自公共 API，在线仅有短期进程缓存；严格复现需要保存输入数据。

模型方程、参数与数据处理见 [Skeleton 模型说明](docs/model.md)。

## 开发与实验

| 代码 | 职责 |
|---|---|
| `app.py` / `main_pipeline.py` | 网页 / CLI 入口 |
| `data_source.py` / `domain_models.py` | 数据读取与活动数据对象 |
| `math_models.py` / `prediction_engine.py` | 节律、骨架拟合与预测 |
| `plotly_viz.py` / `visualizer.py` | 网页 / CLI 绘图 |
| `tuner/` | 参数训练与离线评估 |
| `behavior_*.py` / `tier_surface.py` / `scripts/` | 离线行为模型、多档数据与实验工具 |
| `tests/` | 按模块组织的测试 |

训练命令、测试入口和现有代码边界见 [开发说明](docs/development.md)。
行为模型仍处于实验阶段，未接入网页；数学定义见
[Pace v1](BEHAVIOR_MODEL_THEORY.md)，后续方案见
[Cohort 实验](EXPERIMENT_20260908.md)及[结果报告](event_data/cohort_experiment_20260908/report.md)。
