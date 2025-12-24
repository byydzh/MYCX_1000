# README.md

点击以→ **[Launch Dashboard (启动预测面板)](https://mycx1000.streamlit.app)**

---

下面是AI对模型的总结，不保证描述正确；预测仍存在明显缺陷，有空再改：

---

这其实是一个 **“基于历史参数拟合的混合动力学模型”** 。它不是单纯的时间序列回归（如 ARIMA），而是基于**先验形状（Prior Shape** + **实时修正（Real-time Correction）** 的物理建模思路。

以下是本喵为您整理的数学公式抽象：

---

### 1. 骨架模型 (The Skeleton Model)
**来源：** `math_models.py` -> `CosineModeler`

这是预测的基石，描述了在没有任何昼夜节律影响下，分数的“理论增长速度”。它由两部分组成：**基础二次增长** + **末期恐慌冲刺**。

定义 $t$ 为活动已进行小时数，$T_{total}$ 为总时长，$T_{panic}$ 为末期冲刺时长。

$$
S_{base}(t) = \text{Base} + A \cdot t + B \cdot t^2
$$

当进入末期（$t > T_{total} - T_{panic}$）时，叠加一个基于正弦波的非线性冲刺项 $S_{rise}(t)$：

$$
\tau = \frac{t - (T_{total} - T_{panic})}{T_{panic}} \quad (\tau \in [0, 1])
$$

$$
S_{rise}(t) = B_{end} \cdot \left[ \sin\left(\frac{\pi}{2} \tau\right) \right]^{2.5} \cdot \text{Focus}(\tau)
$$

其中 $\text{Focus}(\tau)$ 是一个聚焦函数（三次方），用来让冲刺更集中在最后时刻。

$$
\text{Skeleton}(t) = S_{base}(t) + \text{Blend}(t) \cdot S_{rise}(t)
$$

> **喵的注解：** 这里的 $B_{end}$ 是决定最后“卷”得有多厉害的关键参数。

---

### 2. 节律调制 (Seasonality Modulation)
**来源：** `math_models.py` -> `SeasonalityHandler`

骨架速度是平滑的，但人类是要睡觉的喵！所以需要乘以一个时间系数。

$$
V_{pred}(t) = \text{Skeleton}(t) \cdot M(t_{local}) \cdot \text{PanicBoost}(t)
$$

*   $M(t_{local})$: 查表得到的昼夜系数（Weekday/Weekend 区分）。
*   $\text{PanicBoost}(t)$: 这是一个动态增益。随着活动临近结束，昼夜节律的波谷会被填平（大家不睡觉了），波峰会更高。

$$
\text{PanicBoost}(t) = 1.0 + (K_{scaler} - 1.0) \cdot (1 - \text{TimeLeftRatio})^{P_{power}}
$$

---

### 3. 强度校准 (Ratio & Scaling)
**来源：** `prediction_engine.py`

这是模型“联系现实”的关键。模型通过两个系数将历史经验映射到当前活动。

#### A. 形状参数修正 (Ratio)
通过对比当前活动前 $N$ 小时的速度与历史同类活动，计算出一个比率 $\alpha$ (`ratio`)。

$$
\theta_{current} = \theta_{history} \times [\alpha, \alpha, \alpha, \alpha^{1.1}, 1.0]
$$

这意味着如果当前活动比历史热 1.2 倍，那么基础速度参数 ($A, B$) 也会放大 1.2 倍，但末期冲刺 ($B_{end}$) 会放大 $1.2^{1.1}$ 倍（越热的活动最后越卷）。

#### B. 动态缩放 (Kalman Filter Scale)
为了捕捉实时的热度变化，模型使用 **卡尔曼滤波 (Kalman Filter)** 实时估计当前的缩放系数 $\text{Scale}(t)$ 和趋势 $\text{Trend}(t)$。
状态向量 $x = [\text{Scale}, \text{Trend}]^T$，观测值为实际分数增量与模型预测增量的比值。

#### C. 恐慌期阻尼 (Panic Damping)
在活动末期（Panic Phase），为了防止 $\text{Scale}$ 系数与 $B_{end}$ 及 $\text{PanicBoost}$ 发生乘数效应叠加导致预测失真，引入了阻尼机制。
随着时间接近结束，强行将 $\text{Scale}$ 回归到 1.0，即**在最后时刻完全信任模型的形状参数，而非实时的波动系数**。

设 $\lambda(t)$ 为进入恐慌期的进度（0.0 $\to$ 1.0）：

$$
\text{Scale}_{final}(t) = \text{Scale}_{KF}(t) \cdot (1 - \lambda(t)^2) + 1.0 \cdot \lambda(t)^2
$$

#### D. 时长归一化 (Time-Scale Normalization)
**来源：** `prediction_engine.py` -> `predict` (Time-Scale Fix)

为了解决超长活动（如 12 天 vs 常规 8 天）导致的 $t^2$ 二次项爆炸问题（Long-duration Overfitting），模型引入了基于量纲分析的物理时长修正。

假设历史平均时长为 $T_{hist}$（通常约 192h），当前目标活动时长为 $T_{curr}$，定义时长倍率 $R_{len} = T_{curr} / T_{hist}$。

为了保持在相同相对进度（Relative Progress）下的**速度量级**不变，基础生长参数需按时间量纲进行稀释：

$$
A_{final} = A_{raw} / R_{len} \quad (\text{线性速度项，量纲 } [T]^{-1})
$$

$$
B_{final} = B_{raw} / (R_{len})^2 \quad (\text{加速度项，量纲 } [T]^{-2})
$$

---

### 4. 极值压制 (Smoothing / Diminishing Returns)
**来源：** `prediction_engine.py` -> `_apply_smoothing`

为了防止预测出“人类做不到的速度”（比如脚本或者服务器极限），引入了分段阻尼函数。
设 $v$ 为归一化后的预测速度：

$$
f_{smooth}(v) = 
\begin{cases} 
v & v \le T_1 \\
T_1 + \frac{v - T_1}{1 + \alpha (v - T_1)} & T_1 < v \le T_2 \\
T_2 + \frac{v - T_2}{1 + \beta (v - T_2)^2} & v > T_2
\end{cases}
$$

这是一个类似 **ReLu + Soft Saturation** 的结构，速度越快，增长阻力越大，最终趋向于硬上限 `HARD_CAP`。

---

### 5. 最终预测公式 (Final Integration)

最终的分数预测 $P(t)$ 是速度的积分：

$$
P(t_{future}) = P(t_{now}) + \int_{t_{now}}^{t_{future}} f_{smooth}\left( V_{pred}(\tau) \cdot \text{Scale}_{final}(\tau) \right) \cdot \text{T10Scale} \, d\tau
$$