import requests
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import os
from chinese_calendar import is_workday
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Module-level session to enable connection reuse and avoid FD leaks
HTTP_SESSION = requests.Session()
try:
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=3)
    HTTP_SESSION.mount('http://', adapter)
    HTTP_SESSION.mount('https://', adapter)
except Exception:
    pass

# ================= 配置区域 =================
EVENT_RANGE = range(200, 300) 
BASE_URL = "https://bestdori.com/api/"
SERVER = 3 # 国服
OUTPUT_FILE = "base_speed_distribution.json"

# ================= 核心工具函数 =================

def fetch_event_meta(event_id):
    """获取活动元数据"""
    try:
        meta_url = f"{BASE_URL}events/{event_id}.json"
        r = HTTP_SESSION.get(meta_url, timeout=5)
        r.raise_for_status()
        metadata = r.json()
        return {
            "event_id": event_id,
            "start_at": int(metadata["startAt"][SERVER]),
            "end_at": int(metadata["endAt"][SERVER]),
            "event_type": metadata.get("eventType", "unknown")
        }
    except:
        return None

def fetch_tier_1000_data(event_id):
    """获取 T1000 分数线数据 (Tracker API)"""
    tracker_url = f"{BASE_URL}tracker/data?server={SERVER}&event={event_id}&tier=1000"
    try:
        r = HTTP_SESSION.get(tracker_url, timeout=10)
        r.raise_for_status()
        tracker_data = r.json()
        if not tracker_data["result"]:
            return None
        return pd.DataFrame(tracker_data["cutoffs"])
    except:
        return None

def fetch_top10_max_speed(event_id):
    """
    获取 T10 数据并计算该活动理论最大速度 (Scale Factor)
    API: eventtop
    """
    # 使用 1小时 (3600000ms) 的间隔来获取较为平滑的极速，避免瞬时爆发的噪声
    url = f"{BASE_URL}eventtop/data?server={SERVER}&event={event_id}&mid=0&interval=3600000"
    
    try:
        r = HTTP_SESSION.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data or "points" not in data:
            return None
        
        # 转换为 DataFrame
        # 只需要前 200 条数据通常足够覆盖开局爆发期，但为了保险起见，
        # 我们这里取前 500 条以确保覆盖到至少几小时的数据量，
        # 因为 eventtop 返回的是所有 top10 玩家的点，10个玩家每小时1个点，100条只够10小时。
        df = pd.DataFrame(data["points"]).head(500)
        
        if df.empty:
            return None
            
        # 优化：仅处理前 N 条数据以节省性能
        # df = df.head(500) 
        
        # === 核心逻辑：按 UID 分组计算速度 ===
        # 1. 排序
        df = df.sort_values(by=["uid", "time"])
        
        # 2. 计算差分 (每个 UID 内部计算)
        df["pt_diff"] = df.groupby("uid")["value"].diff()
        df["time_diff"] = df.groupby("uid")["time"].diff()
        
        # 3. 计算速度 (EP / 分钟)
        # time_diff 单位是毫秒，所以要 / 1000 / 60
        df["speed"] = df["pt_diff"] / (df["time_diff"] / 1000 / 60)
        
        # 4. 清洗数据
        # 去除 NaN (第一条数据没有差分)
        # 去除 速度 < 0 (可能是掉档或数据错误)
        # 去除 速度过大
        valid_speeds = df[(df["speed"] > 0) & (df["speed"] < 1000000)]["speed"]
        
        if valid_speeds.empty:
            return None
            
        # 5. 获取极速
        # 取最大的前几个值的平均，或者直接取最大值（需排除极端异常值）
        # 这里采用取 Top 3 的平均值作为 Scale Factor，比单一最大值更稳定
        top_speeds = valid_speeds.nlargest(3).values
        if len(top_speeds) > 0:
            return np.mean(top_speeds)
        else:
            return valid_speeds.max()
            
    except Exception as e:
        print(f"Error fetching T10 for {event_id}: {e}")
        return None

def calculate_speed_tracker(df):
    """计算 Tracker 数据 (T1000) 的速度"""
    df = df.sort_values("time")
    df["ep_diff"] = df["ep"].diff()
    df["time_diff"] = df["time"].diff() / 1000 / 60
    df["speed"] = df["ep_diff"] / df["time_diff"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["speed"])
    df = df[df["speed"] >= 0]
    return df

def get_day_type(dt):
    """判断日期类型 (工作日 vs 周末)"""
    # 1. 优先判断特定时间段的模式切换
    # 周五 17:00 后 -> 周末模式
    if dt.weekday() == 4 and dt.hour >= 17:
        return "weekend"
    # 周日 23:00 后 -> 工作日模式
    if dt.weekday() == 6 and dt.hour >= 23:
        return "weekday"

    # 2. 使用 chinesecalendar (如果可用)
    if is_workday is not None:
        try:
            if is_workday(dt.date()):
                return "weekday"
            else:
                return "weekend"
        except:
            pass

    # 3. Fallback
    if dt.weekday() >= 5:
        return "weekend"
    return "weekday"

# ================= 主逻辑 =================

def main():
    distribution_data = {
        "weekday": {h: [] for h in range(24)},
        "weekend": {h: [] for h in range(24)}
    }
    
    valid_event_count = 0
    print("🐱 CatGPT 正在启动分析引擎喵...")

    for event_id in EVENT_RANGE:
        # 1. 获取基础信息
        meta = fetch_event_meta(event_id)
        if not meta:
            continue
            
        # 2. 获取 T1000 数据
        df_1000 = fetch_tier_1000_data(event_id)
        if df_1000 is None or df_1000.empty:
            continue
            
        # 3. 【新逻辑】获取 T10 极速作为 Scale Factor
        scale_factor = fetch_top10_max_speed(event_id)
        
        if not scale_factor or scale_factor < 100: # 速度太小说明数据有问题
            print(f"⚠️ Event {event_id}: 无法计算有效的 T10 极速，跳过。")
            continue
            
        # 4. 处理 T1000 数据
        df_1000 = calculate_speed_tracker(df_1000)
        
        # 归一化
        df_1000["norm_speed"] = df_1000["speed"] / scale_factor
        
        # 5. 筛选有效时间段 (排除首日24h，尾日48h)
        start_ts = meta["start_at"]
        end_ts = meta["end_at"]
        valid_start = start_ts + 24 * 3600 * 1000
        valid_end = end_ts - 48 * 3600 * 1000
        
        df_valid = df_1000[(df_1000["time"] >= valid_start) & (df_1000["time"] <= valid_end)].copy()
        
        # 转换时间
        df_valid["dt"] = pd.to_datetime(df_valid["time"], unit="ms") + timedelta(hours=8)
        
        # 填入数据桶
        for _, row in df_valid.iterrows():
            hour = row["dt"].hour
            day_type = get_day_type(row["dt"])
            
            # 过滤异常归一化值 (T1000 速度不应超过 T10 极速太多)
            if 0 <= row["norm_speed"] <= 1.2: 
                distribution_data[day_type][hour].append(row["norm_speed"])
        
        valid_event_count += 1
        print(f"✅ Event {event_id} | T10极速: {scale_factor:.0f} EP/min | 已归档")
        time.sleep(0.5)

    # ================= 聚合输出 =================
    final_distribution = {"weekday": {}, "weekend": {}}
    
    for dtype in ["weekday", "weekend"]:
        for h in range(24):
            speeds = distribution_data[dtype][h]
            if speeds:
                final_distribution[dtype][h] = {
                    "mean": float(np.mean(speeds)),
                    "median": float(np.median(speeds)),
                    "std": float(np.std(speeds)),
                    "count": len(speeds)
                }
            else:
                final_distribution[dtype][h] = None

    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_distribution, f, indent=4)

    # 绘图部分保持不变 (省略以节省篇幅，逻辑同上)
    print(f"\n🎉 分析结束！共处理 {valid_event_count} 个活动。")
    print(f"数据已保存至 {OUTPUT_FILE}，Key 0-23 代表每天的小时段喵！")
    # 生成可视化输出（如果 matplotlib 可用）
    try:
        def plot_distribution(dist, out_prefix="distribution"):
            hours = list(range(24))

            def extract_stats(dtype):
                medians = []
                means = []
                counts = []
                for h in hours:
                    val = dist[dtype].get(h)
                    if val:
                        medians.append(val["median"])
                        means.append(val["mean"])
                        counts.append(val["count"])
                    else:
                        medians.append(float('nan'))
                        means.append(float('nan'))
                        counts.append(0)
                return medians, means, counts

            for dtype in ["weekday", "weekend"]:
                medians, means, counts = extract_stats(dtype)

                fig = Figure(figsize=(10, 4))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(hours, medians, marker='o', label='median')
                ax1.plot(hours, means, marker='x', label='mean', alpha=0.7)
                ax1.set_xlabel('Hour')
                ax1.set_ylabel('Normalized Speed')
                ax1.set_title(f'{dtype.capitalize()} hourly normalized-speed distribution')
                ax1.set_xticks(hours)
                ax1.grid(axis='y', linestyle='--', alpha=0.3)
                ax1.legend(loc='upper left')

                ax2 = ax1.twinx()
                ax2.bar(hours, counts, color='gray', alpha=0.2, label='count')
                ax2.set_ylabel('Count')
                ax2.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)

                out_dir = os.path.dirname(OUTPUT_FILE) or '.'
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{out_prefix}_{dtype}.png")
                try:
                    fig.tight_layout()
                except Exception:
                    pass
                # Prefer fig.savefig; if that fails in headless env, fallback to Agg canvas
                try:
                    fig.savefig(out_path)
                except Exception:
                    try:
                        buf = BytesIO()
                        FigureCanvasAgg(fig).print_png(buf)
                        with open(out_path, 'wb') as f:
                            f.write(buf.getvalue())
                        buf.close()
                    except Exception as e:
                        print(f"Failed to save plot {out_path}: {e}")
                finally:
                    try:
                        fig.clf()
                        del fig
                    except Exception:
                        pass
                print(f"Saved plot: {out_path}")

        plot_distribution(final_distribution, out_prefix=os.path.splitext(OUTPUT_FILE)[0])
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()