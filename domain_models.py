# domain_models.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

@dataclass
class EventMeta:
    """
    存储活动的基础元数据。
    """
    event_id: int
    event_type: str
    start_at: int       # timestamp ms
    end_at: int         # timestamp ms
    aggregate_at: int   # timestamp ms
    
    @property
    def total_hours(self) -> float:
        return (self.end_at - self.start_at) / 3600000.0

    @classmethod
    def from_dict(cls, eid: int, data: Dict[str, Any]) -> 'EventMeta':
        start = data.get('startAt') or data.get('start_at')
        end = data.get('endAt') or data.get('end_at')
        agg = data.get('aggregateAt') or data.get('aggregate_at') or end
        if agg is None: agg = end
            
        return cls(
            event_id=int(eid),
            event_type=data.get('eventType') or data.get('event_type') or 'unknown',
            start_at=int(start) if start else 0,
            end_at=int(end) if end else 0,
            aggregate_at=int(agg) if agg else 0
        )

@dataclass
class EventData:
    """
    存储一个活动的所有相关数据。
    """
    meta: EventMeta
    df: pd.DataFrame
    scale: float
    window_intensity: Optional[float] = None
    fit_params: Optional[np.ndarray] = None
    # [新增] 用于存储未截断的完整数据（上帝视角），默认为 None
    full_df: Optional[pd.DataFrame] = None 

    def __post_init__(self):
        """初始化后立即清洗列名"""
        self._standardize_columns()

    def _standardize_columns(self):
        if self.df is None or self.df.empty:
            return

        rename_map = {
            'ep': 'value',      
            'points': 'value',  
            'score': 'value',   
            'pt': 'value',
            'timestamp': 'time'
        }
        self.df = self.df.rename(columns=rename_map)

        if 'time' not in self.df.columns:
            self.df = self.df.reset_index()
            if 'index' in self.df.columns:
                self.df = self.df.rename(columns={'index': 'time'})

    def clean_data(self):
        self._standardize_columns()
        required = ['time', 'value']
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"EventData Error: Missing column '{col}'. Available: {list(self.df.columns)}")
        self.df = self.df.sort_values('time')

@dataclass
class PredictionResult:
    event_id: int
    future_t: np.ndarray
    future_score: np.ndarray
    future_speed: np.ndarray
    final_score: float
    used_params: np.ndarray
    ratio: float
    scale_factor: float
    full_t_score: np.ndarray
    full_score: np.ndarray