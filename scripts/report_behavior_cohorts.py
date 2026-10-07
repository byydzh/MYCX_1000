"""Render the bounded cohort experiment's paired comparison and development log."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_behavior_cohorts import OUT, aggregate


def main():
    documents = {s:json.loads((OUT/f"{s}.json").read_text(encoding="utf-8"))
                 for s in ("screen","repair","confirm","replay")}
    selection = json.loads((OUT/"selection.json").read_text(encoding="utf-8"))
    model = selection["name"]
    rows = documents["replay"]["rows"]
    baseline = "skeleton_kf_same_rank_history"
    labels = {model:"新行为 Cohort", "pace_v1":"原 Pace", baseline:"Skeleton+KF",
        "skeleton_kf_reward_behavior_history":"奖励换线 Skeleton",
        "last_two_nonnegative_slope":"最后两点斜率",
        "planned_duration_average_speed":"累计均速", "persistence":"Persistence"}
    paired = {m:aggregate(rows,m,support=baseline) for m in (model,"pace_v1",baseline)}
    event_ids = sorted(paired[model]["event_mape"],key=int)
    delta = np.array([paired[model]["event_mape"][e]-paired[baseline]["event_mape"][e] for e in event_ids])
    rng = np.random.default_rng(20260908)
    ci = np.quantile(rng.choice(delta,size=(2000,len(delta)),replace=True).mean(axis=1),[.025,.975])
    lines = ["# 2026-09-08 行为群体投入模型实验结果", "",
        "本轮明显改善了旧行为模型，整体 MAPE 接近 Skeleton 并略低；"
        "但 MAE、sMAPE 和核心 T1000 仍较差，配对差值也没有显著优势。"
        "保留为实验模型，生产继续使用 Skeleton+KF。", "",
        f"选定模型：`{model}`。主比较为 284–319 的同 origin、同档位支持集。", "",
        "| 方法 | MAPE | MAE（万） | sMAPE | 平均偏差 | origin |",
        "|---|---:|---:|---:|---:|---:|"]
    def metric_line(name,m):
        return f"| {name} | {m['mape']:.2f}% | {m['mae']/10000:.2f} | {m['smape']:.2f}% | {m['bias']:+.2f}% | {m['origins']} |"
    lines += [metric_line(labels[m],paired[m]) for m in (baseline,"pace_v1",model)]
    lines += ["", f"新模型减 Skeleton 的活动等权 MAPE 差为 {delta.mean():+.2f} 个百分点；"
        f"36 场范围内实际参与配对 {len(delta)} 场，新模型胜 {int((delta<0).sum())} 场。"
        f"按活动整块 bootstrap 2,000 次，差值 95% 区间 [{ci[0]:+.2f}, {ci[1]:+.2f}] 个百分点。",
        "这描述本组历史活动的抽样不确定性，不是未来预测区间，也不能消除过去开发曝光的影响。", "",
        "![配对比较与逐活动误差差异](comparison.png)", "", "## 全部可用输入与失败", "",
        "每行使用该方法自身可用支持；方法强弱以上方配对表为准。", "",
        "| 方法 | MAPE | MAE（万） | sMAPE | 平均偏差 | origin |",
        "|---|---:|---:|---:|---:|---:|"]
    for method in (model,"pace_v1",baseline,"skeleton_kf_reward_behavior_history",
                   "last_two_nonnegative_slope","planned_duration_average_speed","persistence"):
        lines.append(metric_line(labels[method],aggregate(rows,method)))
    input_failures = [r for r in rows if "input" in r["failures"]]
    model_failures = [r for r in rows if model in r["failures"]]
    lines += ["", f"共计划 {len(rows)} 个 origin，输入失败 {len(input_failures)} 个，"
        f"新模型运行失败 {len(model_failures)} 个。Skeleton 缺失输出保留原报告状态。",
        "", "## 分档位与阶段", "",
        "均在 Skeleton 同支持集上聚合；早/中/晚为真实活动进度的三等分。", "",
        "| 切片 | 新模型 MAPE | 原 Pace MAPE | Skeleton MAPE | 配对 origin |",
        "|---|---:|---:|---:|---:|"]
    slices = [(f"T{t}",dict(tier=t)) for t in sorted({r['tier'] for r in rows})]
    slices += [(p,dict(phase=p)) for p in ("early","middle","late")]
    for label,filters in slices:
        values = [aggregate(rows,m,support=baseline,**filters) for m in (model,"pace_v1",baseline)]
        if all(values):
            lines.append(f"| {label} | {values[0]['mape']:.2f}% | {values[1]['mape']:.2f}% | {values[2]['mape']:.2f}% | {values[0]['origins']} |")
    lines += ["", "## 开发选择与内部确认", "",
        "候选训练池为 192–239；每次预测再按真实时间剔除尚未结束的历史。"
        "240–263 先选择下表预先列明的 12 个候选，再检验一次奖励迁移机制修正；"
        "共 13 个开发候选。264–283 在选定后一次确认。"
        "历史复验前把选定结构用 192–283 重拟合。没有按 284–319 的分数再搜索。", "",
        "| 开发候选 | MAPE | 偏差 | origin |", "|---|---:|---:|---:|"]
    for name,m in sorted(documents["screen"]["metrics"].items(),key=lambda kv:kv[1]["mape"]):
        if name in documents["screen"]["configs"] or name == "pace_v1":
            lines.append(f"| {name}{'（选定）' if name==model else ''} | {m['mape']:.2f}% | {m['bias']:+.2f}% | {m['origins']} |")
    for name in documents["repair"]["configs"]:
        m=documents["repair"]["metrics"][name]
        lines.append(f"| {name}{'（最终选定）' if name==model else ''} | {m['mape']:.2f}% | {m['bias']:+.2f}% | {m['origins']} |")
    lines += ["", "| 阶段 | 选定新模型 MAPE | 原 Pace MAPE | origin |",
              "|---|---:|---:|---:|"]
    for stage,title in (("repair","开发 240–263"),("confirm","确认 264–283"),("replay","历史复验 284–319")):
        d=documents[stage]["metrics"]
        lines.append(f"| {title} | {d[model]['mape']:.2f}% | {d['pace_v1']['mape']:.2f}% | {d[model]['origins']} |")
    for stage in ("repair","confirm","replay"):
        stage_rows = documents[stage]["rows"]
        stage_model = next(iter(documents[stage]["configs"]))
        input_count = sum("input" in r["failures"] for r in stage_rows)
        failure_count = sum(stage_model in r["failures"] for r in stage_rows)
        lines.append(f"\n{stage}：已加载活动的 {len(stage_rows)} 个计划 origin 中，"
            f"输入不足 {input_count} 个、模型拒绝非法前缀 {failure_count} 个。")
        for failure in documents[stage].get("excluded_events",[]):
            lines.append(f"\n{stage} 排除事件 {failure['event_id']}：{failure['excluded_reason']}。不计作成功或零误差。")
    quality=documents['replay']['fit_quality']
    excluded=[q for q in quality if 'excluded' in q]
    included_events = {q['event_id'] for q in quality if 'excluded' not in q}
    lines += ["", "## 数学与数据解释", "",
        "每条历史档线先拟合终局投入份额 m，满足 m≥0、各项之和为 1。三分量版本"
        "使用归一到终值 1 的旧 pace 基函数；六分量版本使用持续参与、两类指数退出"
        "与三类截止激活。形状定义在累计昼夜可用时间上，自动随活动时长伸缩。", "",
        "每个历史事件先贡献同等先验质量，事件内按 log-rank 距离加权；前缀每 6 小时"
        "的相对投入分布通过 KL 广义似然更新权重。各模板先在目标档末次可见分数"
        "处定标，再平均预测曲线。强度 0 是取消前缀更新的消融。", "",
        f"最终拟合保留 {len(quality)-len(excluded)}/{len(quality)} 条历史档线，"
        f"来自 {len(included_events)}/92 个候选训练活动。"
        "缺少结束后 20 分钟内首个终值或终值与此前最高分冲突的档线显式排除；"
        "其余训练 revision 沿用既有因果 cummax 规则。运行时输入下降仍明确失败。", "",
        "旧 pace 的速度系数平均、终局投入份额平均和预测增长倍数平均并不等价。"
        "本轮同时改变拟合损失、有效终值支持、rank 条件化及前缀更新，因此不能把"
        "全部成绩差异归因于其中单一因素。开发集的 k=0/16/64 对照能单独比较"
        "固定其余结构时前缀更新的作用。", "",
        "首轮开发失败后发现终值归一化没有把历史奖励条件迁移到目标档。唯一追加"
        "版本保留已选参数，用旧 pace 原有压力函数把 deadline 份额乘"
        " P_target/P_history 后归一。它不增加待调参数，开发误差单独列在表中。", "",
        "这仍是行为启发的固定排名群体模型，不是可识别的个体玩家模拟器。"
        "六分量的退出与激活比例也不能当成真实玩家统计。新奖励档使用自己的观测"
        "定标；历史 rank 平滑传递的是曲线形状，未读取目标其他榜线作为分数替身。", "",
        "## 复现与边界", "",
        "```powershell", ".\\venv\\Scripts\\python.exe scripts/experiment_behavior_cohorts.py screen",
        ".\\venv\\Scripts\\python.exe scripts/experiment_behavior_cohorts.py repair",
        ".\\venv\\Scripts\\python.exe scripts/experiment_behavior_cohorts.py confirm",
        ".\\venv\\Scripts\\python.exe scripts/experiment_behavior_cohorts.py replay",
        ".\\venv\\Scripts\\python.exe scripts/report_behavior_cohorts.py", "```", "",
        "需要本地既有 192–319 缓存、原 pace 先验及原 284–319 评估 JSON。"
        "新结果冻结逐 origin 预测、真值、输入摘要、选型配置和最终先验。"
        "复验核对目标缓存 SHA、终值及旧 pace 数值后复用原 Skeleton 预测，"
        "没有重新联网请求 T10。原报告部分 T10 原始响应未冻结，仍是旧 baseline"
        "从头复算的边界；本轮新模型不用 T10。", "",
        "284–319 已被上一轮开发查看，不能称作全新盲测。内部确认发生在更早的"
        "活动和旧奖励体系下，跨阶段差异是判断模型稳定性必须同时看的证据。"
        "本轮不改生产入口和 README，不启动 DL，也不把实验胜负自动变成上线结论。"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False,
                         "axes.spines.right":False, "font.size":10})
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={"width_ratios":[1,2]})
    names=["Skeleton+KF","Pace v1","Cohort"]
    values=[paired[m]['mape'] for m in (baseline,'pace_v1',model)]
    colors=['#64748b','#d97706','#0f766e']
    bars=axes[0].bar(names,values,color=colors,width=.6)
    axes[0].bar_label(bars,labels=[f"{v:.2f}%" for v in values],padding=5)
    axes[0].set_ylim(0,max(values)*1.2)
    axes[0].set_ylabel('Event-equal MAPE (%)')
    axes[0].set_title(f"Same support: {paired[model]['origins']} origins")
    x=np.arange(len(delta))
    axes[1].bar(x,delta,color=np.where(delta<0,'#0f766e','#b91c1c'))
    axes[1].axhline(0,color='#334155',linewidth=.8)
    axes[1].set_xticks(x[::2],np.array(event_ids)[::2],rotation=55)
    axes[1].set_ylabel('Cohort minus Skeleton MAPE (percentage points)')
    axes[1].set_xlabel('Event ID (historical replay; previously exposed)')
    axes[1].set_title(f"Wins: {int((delta<0).sum())}/{len(delta)} events; lower is better")
    fig.suptitle(f"Behavior cohort experiment | 284-319 | {model}",weight='bold')
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png',dpi=160)
    plt.close(fig)
    print(json.dumps({"model":model,"paired":{m:{k:v for k,v in s.items() if k!='event_mape'} for m,s in paired.items()},
        "mape_delta_pp":float(delta.mean()),"bootstrap95_pp":ci.tolist(),"wins":int((delta<0).sum())},indent=2))


if __name__ == '__main__':
    main()
