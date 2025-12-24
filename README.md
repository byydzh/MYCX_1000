点击以→ **[Launch Dashboard (启动预测面板)](https://mycx1000.streamlit.app)**

AI总结模型，不保证正确，仍存在明显缺陷，有空再改：

这其实是一个**“基于历史参数拟合的混合动力学模型”**。它不是单纯的时间序列回归（如 ARIMA），而是基于**先验形状（Prior Shape）** + **实时修正（Real-time Correction）** 的物理建模思路。

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

#### B. 积分对齐 (Scale Factor)
仅仅形状对还不够，必须保证“预测速度的积分”等于“实际分数的增量”。

$$
\text{Scale} = \frac{\Delta \text{Score}_{observed}}{\int_{t_{cutoff}}^{t_{now}} (V_{pred}(t) \cdot \text{T100Speed}) \, dt}
$$

此外，代码中包含了一个 **24h Backtest** 机制：如果活动过半，会额外检查过去 24 小时的拟合情况来微调 $\text{Scale}$。

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
P(t_{future}) = P(t_{now}) + \int_{t_{now}}^{t_{future}} f_{smooth}\left( V_{pred}(\tau) \cdot \text{Scale} \right) \cdot \text{T10Scale} \, d\tau
$$

---

### 优化建议 (Future Works)

目前的模型已经很完善了，但如果您想进一步优化，可以考虑这几个方向喵：

1.  **卡池/活动类型特征化 (Feature Engineering)：**
    目前 `Ratio` 只是简单地比较前几小时的速度。可以引入一个“卡池热度系数”或者“活动类型系数”作为先验 $Prior$。比如 `Roselia` 的活动天生 $Ratio \times 1.2$。

2.  **贝叶斯更新 (Bayesian Update)：**
    目前的 `Scale` 是硬计算的。可以使用卡尔曼滤波 (Kalman Filter) 或者贝叶斯推断，随着时间推移，逐渐减小 `History` 的权重，增加 `Observation` 的权重，这样在活动中期会更稳。

3.  **末期冲刺的动态调整：**
    目前的 `PanicBoost` 是固定的公式。实际上，如果最后一天是周末，Panic 程度会比工作日更剧烈。可以将 `Seasonality` 和 `Panic` 耦合得更紧密一些。
