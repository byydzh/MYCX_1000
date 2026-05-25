# main_pipeline.py
import logging
import sys
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 引入所有组件
from data_source import create_data_source
from domain_models import EventData, EventMeta
from math_models import SeasonalityHandler, CosineModeler
from prediction_engine import PredictionEngine
from visualizer import Visualizer
from config import API_SOURCE_CONFIGS, DEFAULT_CONFIG

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('pipeline')

class PredictionPipeline:
    def __init__(self, config=None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        self.data_source = create_data_source(self.config.get('api_source'))
        
        self.seasonality = SeasonalityHandler(
            weekend_multiplier=float(self.config.get('weekend_multiplier', 1.0)),
            panic_scaler=float(self.config.get('panic_scaler', 1.1))
        )
        self.modeler = CosineModeler()
        self.engine = PredictionEngine(self.seasonality, self.modeler, self.config)
        self.visualizer = Visualizer()

    def _wrap_event_data(self, data_pack) -> EventData:
        if not data_pack: return None
        meta_obj = data_pack['meta']
        if isinstance(meta_obj, dict):
            meta_obj = EventMeta.from_dict(data_pack['event_id'], meta_obj)
            
        return EventData(
            meta=meta_obj,
            df=data_pack['dataframe'],
            scale=data_pack['scale'],
            tier=data_pack.get('tier', 1000),
        )

    def _calculate_derived_columns(self, event_data: EventData) -> EventData:
        """
        计算派生列：hours_elapsed, speed, norm_speed
        """
        df = event_data.df
        event_data.clean_data()
        
        # --- 维护延迟自动修正逻辑 ---
        original_start = event_data.meta.start_at
        valid_points = df[df['value'] > 0]
        if not valid_points.empty:
            first_valid_ts = valid_points.iloc[0]['time']
            if first_valid_ts > original_start and (first_valid_ts - original_start) < 86400000:
                from datetime import datetime, timezone
                dt_first = datetime.fromtimestamp(first_valid_ts / 1000, timezone.utc)
                dt_corrected = dt_first.replace(minute=0, second=0, microsecond=0)
                corrected_start = int(dt_corrected.timestamp() * 1000)
                
                if corrected_start > original_start:
                    diff_h = (corrected_start - original_start) / 3600000.0
                    logger.info(f"检测到维护延迟，修正 start_at: +{diff_h:.1f}h")
                    event_data.meta.start_at = corrected_start
        # ------------------------------------

        start_ts = event_data.meta.start_at
        df['hours_elapsed'] = (df['time'] - start_ts) / 3600000.0
        
        if 'speed' not in df.columns:
            diff_val = df['value'].diff()
            diff_time = df['time'].diff() / 60000.0 
            speed = diff_val / diff_time
            df['speed'] = speed.fillna(0.0)
            df.loc[~np.isfinite(df['speed']), 'speed'] = 0.0
            df.loc[df['speed'] < 0, 'speed'] = 0.0
            
        if 'norm_speed' not in df.columns:
            if event_data.scale is None or event_data.scale <= 0:
                raise ValueError("无法获取有效的 T10 scale，请检查所选 API 数据源是否提供对应的 tier=10 / eventtop 数据。")
            df['norm_speed'] = df['speed'] / event_data.scale
            
        event_data.df = df
        return event_data

    def _fetch_tier_data(self, tier, target_event_id, meta_raw, scale_val, event_type):
        """在独立线程中获取单线数据（每个线程有自己的 ds 实例）"""
        ds = create_data_source(self.config.get('api_source'))
        try:
            tier_pack = ds.fetch_event_data_pack(target_event_id, tier=tier, meta=meta_raw, scale=scale_val)
            if not tier_pack:
                return tier, None, None, "数据不可用"

            similar_packs = ds.find_similar_events(
                target_event_id, event_type,
                count=self.config.get('similar_count', 5),
                ignore_ids=self.config.get('ignore_event_ids', []),
                tier=tier,
            )
            return tier, tier_pack, similar_packs, None
        except Exception as e:
            return tier, None, None, str(e)
        finally:
            ds.close()

    def run(self, target_event_id=None, debug_hours=None, tiers=None):
        if tiers is None:
            tiers = [1000]

        logger.info("启动预测流水线...")

        if target_event_id is None:
            target_event_id = self.data_source.get_current_event_id()
            if not target_event_id:
                logger.error("无法自动获取当前活动 ID")
                return

        logger.info(f"目标活动 ID: {target_event_id}, 层级: {tiers}")

        # --- 预取共享资源 ---
        meta_raw = self.data_source.fetch_event_meta(target_event_id)
        if not meta_raw:
            logger.error("无法获取目标活动元数据")
            return
        scale_val = self.data_source.fetch_top10_max_speed(target_event_id)
        event_type = meta_raw.get('event_type', 'unknown')
        self.data_source.fetch_events_index()  # 预热缓存

        # --- 并行获取各线数据 ---
        tier_packs = {}
        tier_similar = {}
        max_workers = min(len(tiers), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_tier_data, tier, target_event_id, meta_raw, scale_val, event_type): tier
                for tier in tiers
            }
            for f in as_completed(futures):
                tier, pack, similar, err = f.result()
                if err:
                    logger.warning(f"T{tier}: {err}")
                else:
                    tier_packs[tier] = pack
                    tier_similar[tier] = similar

        # --- 顺序执行预测（CPU 密集，无需并行）---
        tier_results = {}
        for tier in tiers:
            if tier not in tier_packs:
                continue
            logger.info(f"--- 处理 T{tier} ---")

            target_data = self._wrap_event_data(tier_packs[tier])
            target_data = self._calculate_derived_columns(target_data)
            target_data.full_df = target_data.df.copy()

            if debug_hours:
                limit_ts = target_data.meta.start_at + (debug_hours * 3600 * 1000)
                target_data.df = target_data.df[target_data.df['time'] <= limit_ts].copy()

            history_events = []
            for pack in tier_similar.get(tier, []):
                h_data = self._wrap_event_data(pack)
                try:
                    h_data = self._calculate_derived_columns(h_data)
                    history_events.append(h_data)
                except Exception as e:
                    logger.warning(f"跳过历史活动 {h_data.meta.event_id}: {e}")

            logger.info(f"找到 {len(history_events)} 个有效历史活动 T{tier}")

            result = self.engine.predict(target_data, history_events, debug_hours=debug_hours)
            tier_results[tier] = result
            logger.info(f"T{tier} 预测完成: {int(result.final_score):,} (Ratio: {result.ratio:.4f})")

        logger.info("正在绘图...")
        if tier_results:
            first_tier = next(iter(tier_results))
            first_target = tier_packs.get(first_tier)
            if first_target:
                target_data = self._wrap_event_data(first_target)
                target_data = self._calculate_derived_columns(target_data)
                self.visualizer.plot_prediction(target_data, tier_results[first_tier],
                                                debug_hours=debug_hours)

        self.data_source.close()
        logger.info("流水线运行结束 喵！")
        return tier_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_id', '-e', type=int, help='Target Event ID')
    parser.add_argument('--debug_hours', '-d', type=float, help='Debug hours limit')
    parser.add_argument(
        '--api-source',
        choices=sorted(API_SOURCE_CONFIGS.keys()),
        help='API source profile'
    )
    parser.add_argument(
        '--tiers', '-t',
        type=str,
        default='1000',
        help='Comma-separated tier numbers, e.g. "500,1000,1500,2000" (default: 1000)'
    )
    args = parser.parse_args()

    runtime_config = DEFAULT_CONFIG.copy()
    if args.api_source:
        runtime_config['api_source'] = args.api_source

    tiers = [int(x.strip()) for x in args.tiers.split(',')]

    pipeline = PredictionPipeline(config=runtime_config)
    
    pipeline.run(target_event_id=args.event_id, debug_hours=args.debug_hours, tiers=tiers)
