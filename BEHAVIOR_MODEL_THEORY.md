# Behavior Pace 模型：数学定义与实验边界

## 1. 当前范围

新模型预测活动中固定排名 \(r\) 的累计分数 \(S_r(t)\)。实验层只把当前活动
master 数据实际声明的奖励档位作为目标，并为每条目标线读取它自己的 Tracker
前缀。T1000 没有特殊地位：它既不是默认目标，也不是幅度尺度或 rank 参考
锚。

公共 API 观测的是固定排名的群体档线，而不是玩家 UID 和完整个体轨迹。因此
当前实现是一个从玩家行为机制出发的低维 pace 模型，不声称已经识别或模拟了
每个具体玩家。可以直接检验的是奖励档线预测本身。

线上正式模型仍是 `Skeleton + Kalman Filter`。本文描述的
`behavior_pace_model.py` 处于实验态，只输出确定性 pace 曲线。

## 2. 可见信息与 origin 遮罩

对预测时点 \(o\)，档位 \(r\) 的唯一可见前缀是

\[
\mathcal D_r(o)=\{(t_k,S_{r,k}):t_k\le o\}.
\]

实现先按 `tracker.time <= origin` 生成布尔遮罩，随后才解析和验证分数列。
因此 origin 后的分数即使被修改为非法值，也不能进入计算。不同档位可以在
完全不同的时间被采样；模型不要求共同时间戳、整点对齐或容差拼接。

令最后一个可见时间为

\[
a_r(o)=\max\{t_k:t_k\le o\}.
\]

预测以 \((a_r,S_r(a_r))\) 为每档自己的连续锚点。若 Tracker 在 origin 前已有
一段时间没有更新，则模型会显式累计 \(a_r\) 到 \(o\) 的增长，而不是把这段
时间当成零速度。

训练缓存中的 Tracker 回退被视为 revision：先保留原始诊断，再按每档因果
`cummax` 构造训练视图。公开预测函数则要求传入的可见累计分数非递减，遇到
下降会明确失败，不会暗中改写运行时输入。

## 3. 昼夜可用性、奖励压力与终局紧迫度

活动时长为 \(T\)，\(t\in[0,T]\) 表示自活动开始后的小时数。活动开始的本地
小时由绝对时间戳和 `utc_offset_hours` 计算，不能把所有活动默认成午夜开场。

昼夜可用性为

\[
A(t)=\max\left\{
1+\alpha_A\cos\frac{2\pi(t+h_0-h_{\mathrm{peak}})}{24},
A_{\min}
\right\},
\qquad
H(t)=\int_0^t A(s)\,ds.
\]

开局投入的剩余比例定义为

\[
D(t)=1-\frac{H(t)}{H(T)}.
\]

它从 1 单调降到 0，表示一批常规追线投入随活动进程逐渐耗尽。终局紧迫度为

\[
U(t)=
\left[
\frac{\varepsilon}
{\operatorname{clip}((T-t)/T,0,1)+\varepsilon}
\right]^p,
\]

并在活动结束时等于 1。

令当前活动真实奖励线集合为

\[
\mathcal R_E=\{\rho_1,\ldots,\rho_m\}.
\]

基础排名竞争度与奖励覆盖比例分别为

\[
c(r)=\operatorname{clip}\left(
\frac{\log(R_{\max}/r)}
{\log(R_{\max}/R_{\min})},0,1
\right),
\]

\[
J_E(r)=
\begin{cases}
\frac1m\sum_{j=1}^{m}\mathbf 1[r\le\rho_j],&m>0,\\
0,&m=0.
\end{cases}
\]

两者以无额外拟合参数的饱和并集合成

\[
P_0(r)=p_{\min}+(1-p_{\min})c(r),
\qquad
P_E(r)=1-(1-P_0(r))(1-J_E(r)).
\]

\(\mathcal R_E\) 必须来自该活动的真实元数据，而不是从预测误差反推的行为
分类。空集合只表示没有特殊奖励加成，此时 \(P_E=P_0\)；字段缺失或结构错误
属于数据失败。

## 4. 三分量 pace 方程

对每个档位 \(r\)，瞬时得分速度为

\[
v_r(t)=q_r A(t)\left[
w_s+w_lD(t)+w_dP_E(r)U(t)
\right],
\]

其中 \(q_r\ge0\) 是该档位独立的幅度，事件内共享权重满足

\[
w_s,w_l,w_d\ge0,
\qquad
w_s+w_l+w_d=1.
\]

三项依次对应 `sustain`、`launch` 和 `deadline`。定义

\[
G(t)=\int_0^t A(s)U(s)\,ds,
\]

则三个累计 pace 分量为

\[
C_s(t)=H(t),
\qquad
C_l(t)=H(t)-\frac{H(t)^2}{2H(T)},
\qquad
C_{d,r}(t)=P_E(r)G(t).
\]

记

\[
C_r(t;w)=w_sC_s(t)+w_lC_l(t)+w_dC_{d,r}(t),
\]

完整累计曲线就是 \(S_r(t)=q_rC_r(t;w)\)。所有分量和权重均非负，所以
单档预测分数非递减。不同档位拥有不同幅度且独立锚定，当前实现没有跨档位
排序投影，因此不能额外宣称所有输出线天然保持横截面顺序。

## 5. 异步档位拟合

单个训练事件中，档位 \(r\) 的相邻观测区间记为

\[
\Delta t_{rj}=t_{r,j}-t_{r,j-1},
\qquad
y_{rj}=\frac{S_{r,j}-S_{r,j-1}}{\Delta t_{rj}}.
\]

该区间的三分量平均设计向量是

\[
x_{rj}=\frac{1}{\Delta t_{rj}}
\begin{bmatrix}
\Delta C_s & \Delta C_l & \Delta C_{d,r}
\end{bmatrix}^{\!\top}.
\]

给定共享权重 \(w\) 后，每档幅度被解析地 profile 掉：

\[
\hat q_r(w)=\max\left\{
\frac{\sum_j\Delta t_{rj}(x_{rj}^{\top}w)y_{rj}}
{\sum_j\Delta t_{rj}(x_{rj}^{\top}w)^2},
0
\right\}.
\]

每档使用 duration-weighted 相对平方损失

\[
L_r(w)=
\frac{
\sum_j\Delta t_{rj}
\left[y_{rj}-\hat q_r(w)x_{rj}^{\top}w\right]^2
}{
\sum_j\Delta t_{rj}y_{rj}^2
},
\]

事件权重为 simplex 上的

\[
\hat w_E=\arg\min_{w\in\Delta^2}
\frac1{|\mathcal T_E|}\sum_{r\in\mathcal T_E}L_r(w).
\]

因此每档先按自己的观测区间拟合，再在事件内等权平均相对损失；不需要同步
时间网格，也不会让高分档仅因数值量级大而支配目标。当前实现使用多起点
SLSQP 直接求解这个三分量 simplex；运行时使用拟合后的事件等权权重生成
确定性点预测。

## 6. 192–283 事件等权先验

`behavior_pace_prior.py` 把训练边界固定为完整整数区间 `192..283`，共 92 个
事件。它只构造并读取精确路径 `<cache_dir>/<event_id>.json`，不会扫描目录、
猜测文件名或自动吸收 284 之后的新事件。

为了避免历史采集 schema 扩张改变事件权重，所有 92 个事件只使用共同支持

\[
\mathcal T_{\mathrm{train}}=
\{50,100,300,500,1000,2000\}.
\]

六档完全对称，每档都必须至少有两个活动窗口内观测。缓存中的额外档位会被
记录但不进入拟合。档位之间仍不要求时间同步。

训练边界处理遵循以下固定规则：

1. 活动开始时注入确定性边界 \(S_r(0)=0\)；如果 Tracker 恰在开始时给出
   非零分数，则拒绝该事件。
2. 每档训练视图使用因果 `cummax` 修复 revision，并记录修复次数。
3. 若活动结束后 20 分钟内存在 Tracker 行，只把**第一条** post-end 行映射
   到 `end_at`；后续行仅作不一致诊断，不能改写终值。
4. 缺失或损坏的事件显式进入排除列表，覆盖门槛失败时不发布先验。

此外，构建器要求每个缓存携带完整的 `reward_tier_provenance`：来源必须为
HHWX、服务器必须为国服 3，且 `last_appearance` 推导出的档位集合必须与
`reward_tiers` 精确一致。只有一个外观合法的档位列表而没有来源证据时，事件
会失败而不是进入训练。

每个合格事件只贡献一个 \(\hat w_E\)，最终先验是算术平均

\[
\bar w=\frac1{N_{\mathrm{included}}}
\sum_E\hat w_E.
\]

这是真正的事件等权，而不是按 Tracker 行数、活动时长或档位分数加权。
`behavior-pace-prior-v1` 同时记录输入 SHA、模型源码 SHA、逐事件拟合诊断、
排除原因、覆盖率和算法配置，使先验来源可以复核。

## 7. Last-visible 连续锚定

运行时不重新拟合 simplex 权重。给定事件等权先验 \(\bar w\)，对目标档位
\(r\) 和它在 origin 前的最后可见点 \(a_r\)，幅度为

\[
\hat q_r=\frac{S_r(a_r)}{C_r(a_r;\bar w)}.
\]

任意预测时点 \(t\ge o\) 的分数为

\[
\widehat S_r(t)=
S_r(a_r)
\frac{C_r(t;\bar w)}{C_r(a_r;\bar w)}.
\]

所以曲线严格穿过最后可见点；当 \(a_r<o\) 时，origin 处预测可以高于该
锚点，这是对 Tracker 新鲜度缺口的显式积分。目标档缺少正的可见锚点时必须
失败，不允许换成 T1000、相邻档或其他数据源的合成值。

## 8. 验证与生产边界

模型判断应直接在已完成历史活动上做 rolling-origin 回放。每个 origin 的
候选方法必须接收完全相同的 `tracker.time <= origin` 前缀，并在所有预测完成
后才读取真实终值。修改任意 origin 后缀时，输入和预测应逐位不变。这种因果
遮罩可以立即验证模型，不需要等待新的活动发生。

结果必须按真实奖励档与基线并列：至少包括线上 `Skeleton + Kalman Filter`、
按历史奖励线行为换算的 Skeleton 基线和简单 causal persistence。应同时给出
逐事件/逐 origin 误差与聚合指标，不能只报告一个脱离基线的 sMAPE，也不能
用固定 T1000 的结果替代其他奖励档。

固定的 284–319 回放包含 36 个活动、57 个真实奖励档和 466 个共同可评估
origin。按 origin、活动内奖励档、活动依次等权后，pace 的 MAPE/MAE 为
21.64%/194.26 万，优于累计均速外推的 28.84%/256.30 万和最后两点斜率的
25.09%/214.91 万。pace 在 36/36 个活动上都胜累计均速，说明三分量形状不是
无效装饰。

但在同档 Skeleton 成功的 418 个相同 origin 上，pace 的 MAPE/MAE 为
21.15%/194.02 万，Skeleton 为 15.99%/128.52 万；pace 的完整样本平均偏差也
达到 -20.85%。因此当前结构仍保持实验态，线上默认继续使用
`Skeleton + Kalman Filter`。下一轮理论工作首先要解释并消除这种系统性低估，
不能靠 318/319 或其他已评分事件直接调紧迫度参数。
