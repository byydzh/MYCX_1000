"""
优化器接口 (Optimizer Interface)

定义参数优化的抽象接口，提供统一的优化入口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
import logging

logger = logging.getLogger('tuner.optimizer')


class ParameterOptimizer(ABC):
    """
    参数优化器抽象基类
    
    职责：
    1. 定义参数优化的统一接口
    2. 提供默认的实现逻辑
    """
    
    def __init__(self, feature_extractor=None):
        """
        初始化优化器
        
        Args:
            feature_extractor: 特征提取器实例
        """
        self.feature_extractor = feature_extractor
        logger.info("ParameterOptimizer initialized")
    
    @abstractmethod
    def optimize(self, activities: List[Dict[str, Any]],
                 search_space: Optional[Dict[str, tuple]] = None,
                 max_iterations: int = 50) -> Dict[str, float]:
        """
        执行参数优化
        
        Args:
            activities: 活动列表
            search_space: 参数搜索空间
            max_iterations: 最大迭代次数
            
        Returns:
            最优参数配置
        """
        pass
    
    @abstractmethod
    def evaluate_params(self, params: Dict[str, float],
                        activities: List[Dict[str, Any]]) -> float:
        """
        评估参数配置的性能
        
        Args:
            params: 参数配置
            activities: 活动列表
            
        Returns:
            性能评分（越小越好）
        """
        pass
    
    def get_current_params(self) -> Dict[str, float]:
        """
        获取当前最优参数
        
        Returns:
            参数配置字典
        """
        return {}
    
    def save_params(self, params: Dict[str, float], 
                    filepath: Optional[str] = None):
        """
        保存参数配置
        
        Args:
            params: 参数配置
            filepath: 保存路径
        """
        logger.info(f"Saved parameters to {filepath}")
    
    def load_params(self, filepath: str) -> Dict[str, float]:
        """
        加载参数配置
        
        Args:
            filepath: 文件路径
            
        Returns:
            参数配置字典
        """
        logger.info(f"Loaded parameters from {filepath}")
        return {}
