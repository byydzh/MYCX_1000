# main_pipeline.py
import logging
import sys
import pandas as pd
import numpy as np

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
            scale=data_pack['scale']
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

    def run(self, target_event_id=None, debug_hours=None):
        logger.info("启动预测流水线...")
        
        if target_event_id is None:
            target_event_id = self.data_source.get_current_event_id()
            if not target_event_id:
                logger.error("无法自动获取当前活动 ID")
                return
        
        logger.info(f"目标活动 ID: {target_event_id}")

        target_pack = self.data_source.fetch_event_data_pack(target_event_id)
        if not target_pack:
            logger.error("无法获取目标活动数据")
            return
        
        target_data = self._wrap_event_data(target_pack)
        
        # 先在完整数据上计算派生列 (speed, hours_elapsed)
        target_data = self._calculate_derived_columns(target_data)
        
        # 保存一份完整数据的副本到 full_df
        target_data.full_df = target_data.df.copy()

        # 然后再进行截断，传给 Engine 的 df 是残缺的
        if debug_hours:
            limit_ts = target_data.meta.start_at + (debug_hours * 3600 * 1000)
            target_data.df = target_data.df[target_data.df['time'] <= limit_ts].copy()
            logger.info(f"Debug 模式: 数据截断至 {debug_hours} 小时 (Full data retained in .full_df)")

        logger.info(f"正在扫描同类活动 (Type: {target_data.meta.event_type})...")
        similar_packs = self.data_source.find_similar_events(
            target_event_id, target_data.meta.event_type, count=5
        )
        
        history_events = []
        for pack in similar_packs:
            h_data = self._wrap_event_data(pack)
            try:
                h_data = self._calculate_derived_columns(h_data)
                history_events.append(h_data)
            except Exception as e:
                logger.warning(f"跳过历史活动 {h_data.meta.event_id}: 数据异常 - {e}")

        logger.info(f"找到 {len(history_events)} 个有效历史活动用于对比")

        logger.info("开始核心计算...")
        result = self.engine.predict(target_data, history_events, debug_hours=debug_hours)
        
        logger.info(f"预测完成! 最终预测分数: {int(result.final_score):,}")
        logger.info(f"使用的修正系数 (Ratio): {result.ratio:.4f}")

        logger.info("正在绘图...")
        self.visualizer.plot_prediction(target_data, result, debug_hours=debug_hours)
        
        self.data_source.close()
        logger.info("流水线运行结束 喵！")

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
    args = parser.parse_args()

    runtime_config = DEFAULT_CONFIG.copy()
    if args.api_source:
        runtime_config['api_source'] = args.api_source

    pipeline = PredictionPipeline(config=runtime_config)
    
    if args.event_id:
        pipeline.run(target_event_id=args.event_id, debug_hours=args.debug_hours)
    else:
        # 默认行为
        pipeline.run()
