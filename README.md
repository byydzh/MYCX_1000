# README.md

点击以→ **[Launch Dashboard (启动预测面板)](https://mycx1000.streamlit.app)**

---

下面是AI对模型的总结，不保证描述正确；预测仍存在明显缺陷，有空再改：

---

总体上这是一个“基于历史参数的物理骨架模型 + 实时修正”的混合预测器。核心思想是：用可解释的低维骨架形状描述长期速度趋势（Skeleton），在此基础上通过短期观测做强度校准（Ratio / Scale）并在观测充足时对部分形状参数做带先验的重拟合（MAP / 正则化）。

主要实现文件与职责：

---

### 1. 骨架模型 (The Skeleton Model)
**来源：** `math_models.py` -> `CosineModeler`


骨架表达式由两部分组成：基础增长项和末期冲刺项。

基础项（示例量级、可扩展到其它基函数）：

$$
S_{base}(t) = \mathrm{Base} + A\,t + B\,t^2
$$

末期冲刺项（Panic / end surge）：令 $T_{total}$ 为活动总时长，$T_{panic}$ 为冲刺期长度，定义局部进度

$$
	au(t)=\frac{t-(T_{total}-T_{panic})}{T_{panic}},\quad \tau\in[0,1]
$$

使用平滑升起函数（示例实现为 $\bigl[\sin(\tfrac{\pi}{2}\tau)\bigr]^{2.5}$ 和一个三次聚焦函数 $\mathrm{Focus}(\tau)$）：

$$
S_{rise}(t)=B_{end}\cdot\bigl[\sin(\tfrac{\pi}{2}\tau)\bigr]^{2.5}\cdot\mathrm{Focus}(\tau)
$$

将两者混合得到骨架速度（unit: normalized speed）

$$
\mathrm{Skeleton}(t)=S_{base}(t)+\mathrm{Blend}(t)\cdot S_{rise}(t)
$$

实现注意点：`CosineModeler.shape_function` 给出了以上组合的具体系数化实现，含 `Base, A, B, B_{end}, T_{panic}` 等参数。

> **喵的注解：** 这里的 $B_{end}$ 是决定最后“卷”得有多厉害的关键参数。

---

### 2. 节律调制 (Seasonality Modulation)
**来源：** `math_models.py` -> `SeasonalityHandler`


骨架为“无节律”的理论速度，实际预测需要乘以昼夜/周末权重与 panic 增益：

$$
V_{pred}(t)=\mathrm{Skeleton}(t)\times M(t_{local})\times \mathrm{PanicBoost}(t)
$$

其中 $M(t_{local})$ 来自 `SeasonalityHandler` 的查表/插值，表示本地小时（或工作日/周末）对活跃度的缩放。

示例 PanicBoost 形式：

$$
\mathrm{PanicBoost}(t)=1+(K_{scaler}-1)\cdot(1-\mathrm{TimeLeftRatio})^{P}
$$

实现细节在 `SeasonalityHandler.apply_seasonality` 中，包含对局部时区偏移的处理与 panic 权重的平滑上升。

---

### 3. 强度校准 (Ratio & Scaling)
**来源：** `prediction_engine.py`


此部分是将历史先验映射到当前活动的桥梁，包含两个层面：

1) 全局形状缩放（`ratio`）——通过对比短期窗口内的骨架速率（`skeleton_speed`）与历史同类窗口的平均值计算得到：

$$
\mathrm{skeleton\_ratio}=\frac{I_{curr}}{\overline{I}_{hist}},\quad I=\text{mean skeleton speed in } [t_{start}, t_{end}]
$$

系统还计算基于 `norm_speed` 的归一化比（`norm_ratio`），并根据已观测时间占比 $s$ 对两者加权混合：

$$
\mathrm{chosen\_ratio}=(1-w(s))\cdot\mathrm{skeleton\_ratio}+w(s)\cdot\mathrm{norm\_ratio}
$$

最终对历史参数做元素层缩放：

$$
	\theta_{init}=\theta_{hist}\times[\alpha,\alpha,\alpha,\alpha^{1.1},1]
$$

（实现映射见 `_calculate_ratio` 与 `predict` 中对 `pred_params` 的缩放逻辑）。

#### B. 动态缩放 (Kalman Filter Scale)
为了捕捉实时的热度变化，模型使用卡尔曼滤波估计 `Scale` 与 `Trend`：

状态与观测模型：

$$
x_{k+1}=F\,x_k+w_k,\quad w_k\sim\mathcal{N}(0,Q)
$$

$$
z_k=H\,x_k+v_k,\quad v_k\sim\mathcal{N}(0,R)
$$

其中通常取 $x=[\mathrm{Scale},\mathrm{Trend}]^\top$，测量 $z_k$ 来自实测得分增量与模型预测增量的比值：

$$
z_k=\frac{\Delta P_{obs}}{\Delta P_{model}}\quad(\text{在代码中以每小时区间构造})
$$

卡尔曼滤波的预测/更新步骤在 `_run_kalman_filter` 中实现；`_calculate_scale_factor` 会对 KF 输出做截断、阻尼并产生用于未来每点的 `scale_curve`。


#### C. 恐慌期阻尼 (Panic Damping)
为防止 `Scale` 与末期项叠加产生爆炸性误差，预测在进入 panic 区间时对 `Scale` 做平滑回归到 1.0：

$$
\mathrm{Scale}_{final}(t)=\mathrm{Scale}_{KF}(t)\cdot(1-\lambda(t)^2)+1\cdot\lambda(t)^2
$$

其中 $\lambda(t)$ 为进入 panic 的进度函数（代码通过 $T_{panic}$ 与时间插值计算），实现位于 `_calculate_scale_factor`。


#### D. 时长归一化 (Time-Scale Normalization)
基于量纲分析，当将时间尺度按 $R_{len}$ 放大时，线性项 $A$ 与二次项 $B$ 的单位分别为 $[T]^{-1}$ 和 $[T]^{-2}$，因此需要调整：

令 $R_{len}=T_{curr}/T_{hist}$，则

$$
A_{adj}=A_{raw}/R_{len},\quad B_{adj}=B_{raw}/R_{len}^2
$$

代码在 `predict` 中对显著不同的总时长应用稀释策略以避免 $t^2$ 在长活动中主导输出。

---

### 4. 极值压制 (Smoothing / Diminishing Returns)
**来源：** `prediction_engine.py` -> `_apply_smoothing`


为防止输出不可实现的高速度，引入三阶段压制（mild -> strong -> hard cap），实现细节见 `_apply_smoothing`：

设 $v$ 为归一化速度，阈值 $T_1<T_2$，则

$$
f_{smooth}(v)=\begin{cases}
v,&v\le T_1\\
T_1+\dfrac{v-T_1}{1+\alpha (v-T_1)},&T_1<v\le T_2\\
T_2+\dfrac{v-T_2}{1+\beta (v-T_2)^2},&v>T_2
\end{cases}
$$

此外对结果做 `min(..., HARD\_CAP)` 限制以避免极端数值。

---

### 5. 最终预测公式 (Final Integration)

最终的分数预测 $P(t)$ 是速度的积分：

$$
P(t_{future})=P(t_{now})+\int_{t_{now}}^{t_{future}} f_{smooth}\bigl(V_{pred}(\tau)\cdot\mathrm{Scale}_{final}(\tau)\bigr)\cdot\mathrm{T10Scale}\,d\tau
$$

其中 `T10Scale`（代码中的 `target.scale`）将归一化速度转换为实际点/分钟或点/小时的量级。

---

## 8. 实现映射（代码位置与说明）

- `math_models.py` (`CosineModeler`): 骨架函数与 panic 叠加的具体实现。可在 `shape_function` 中查看参数化表达。
- `math_models.py` (`SeasonalityHandler`): 本地化昼夜系数、panic 增益与节律乘法的实现。
- `prediction_engine.py`:
	- `_calculate_ratio`: 负责短期窗口内对比，输出 `ratio`。
	- `_fit_history_params`: 从历史事件中拟合/取平均骨架参数作为先验。
	- `_refit_shape_params`: 当观测数据充足时，用带先验的最小二乘（MAP）对 `A,B` 等形状参数做重拟合（正则化项以历史参数为中心）。
	- `_run_kalman_filter` / `_calculate_scale_factor`: 基于观测增量与模型增量比的卡尔曼滤波器，输出 `Scale,Trend` 并构造未来 `scale_curve`。
	- `_apply_smoothing`: 对归一化速度进行三阶段压制并上限截断。

---

## 9. 设计合理性与已知局限

- 设计上将“形状修正（参数层）”与“幅度修正（观测层）”分离，提升了在稀疏数据下的稳健性，同时允许在观测充分时改变曲率（A/B）。
- 当前 KF 主要用于幅度/趋势估计；若希望对参数向量做在线滤波，可考虑对 `\theta` 本身使用扩展卡尔曼/粒子滤波或在线贝叶斯更新。
- 对于 `B_{end}`（末期冲刺）与 panic 的拟合，建议保持保守策略（历史先验 + ratio 缩放），仅在接近结束并有大量观测时才尝试直接拟合。