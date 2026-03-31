"""
参数学习器 (Parameter Learner)

统一的参数学习接口，整合偏差数据库、特征提取器和优化器。
"""

import logging
from typing import Dict, List, Any, Optional
from .deviation_db import DeviationDatabase
from .features import FeatureExtractor
from .optimizer import ParameterOptimizer
from .bayesian_optimizer import BayesianOptimizer
from config_learning import get_learning_config, DEFAULT_LEARNING_CONFIG

logger = logging.getLogger('tuner.learner')


class ParameterLearner:
    """
    参数学习器类
    
    职责：
    1. 整合偏差数据库、特征提取器和优化器
    2. 提供统一的参数学习接口
    3. 管理学习过程和结果
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化参数学习器
        
        Args:
            config: 学习配置字典
        """
        self.config = get_learning_config(config)
        self.enabled = self.config.get('enabled', True)
        
        # 初始化组件
        self.deviation_db = DeviationDatabase(
            data_dir=self.config['data_source']['data_dir'],
            max_activities=self.config['data_source']['history_count']
        )
        
        self.feature_extractor = FeatureExtractor()
        
        # 创建优化器
        optimizer_config = self.config.get('optimizer', {})
        algorithm = optimizer_config.get('algorithm', 'bayesian')
        
        if algorithm == 'bayesian':
            self.optimizer = BayesianOptimizer(
                feature_extractor=self.feature_extractor,
                random_seed=optimizer_config.get('random_seed', 42)
            )
        else:
            # 默认使用贝叶斯优化
            self.optimizer = BayesianOptimizer(
                feature_extractor=self.feature_extractor,
                random_seed=optimizer_config.get('random_seed', 42)
            )
        
        logger.info(f"ParameterLearner initialized with algorithm={algorithm}")
    
    def record_prediction(self, activity_id: int,
                         actual_final_score: float,
                         predicted_final_score: float,
                         prediction_params: Dict[str, float],
                         activity_features: Optional[Dict[str, Any]] = None):
        """
        记录预测结果
        
        Args:
            activity_id: 活动 ID
            actual_final_score: 实际最终分数
            predicted_final_score: 预测的最终分数
            prediction_params: 使用的参数配置
            activity_features: 活动特征（可选）
        """
        if not self.enabled:
            return
        
        # 提取或获取活动特征
        if activity_features is None:
            activity_features = {}
        
        # 记录偏差
        self.deviation_db.record_prediction(
            activity_id=activity_id,
            actual_final_score=actual_final_score,
            predicted_final_score=predicted_final_score,
            prediction_params=prediction_params,
            activity_features=activity_features
        )
    
    def get_optimized_params(self, target: Any, history: List[Any]) -> Dict[str, float]:
        """
        获取优化后的参数
        
        Args:
            target: 目标活动
            history: 历史活动列表
            
        Returns:
            最优参数配置
        """
        if not self.enabled:
            logger.info("Parameter learning is disabled")
            return {}
        
        # 检查是否有足够的数据
        activities = self.deviation_db.get_recent_activities(
            count=self.config['data_source']['history_count']
        )
        
        if len(activities) < self.config['data_source']['min_activities']:
            logger.warning(f"Not enough activities for optimization (have {len(activities)}, "
                          f"need at least {self.config['data_source']['min_activities']})")
            return {}
        
        # 执行优化
        search_space = self.feature_extractor.get_param_search_space()
        optimized_params = self.optimizer.optimize(
            activities=activities,
            search_space=search_space,
            max_iterations=self.config['optimizer']['max_iterations']
        )
        
        return optimized_params
    
    def train(self) -> Dict[str, Any]:
        """
        训练参数学习器
        
        Returns:
            训练结果字典
        """
        if not self.enabled:
            logger.info("Parameter learning is disabled")
            return {}
        
        # 获取训练数据集
        training_set = self.deviation_db.get_training_dataset()
        
        if training_set is None:
            logger.warning("Not enough data for training")
            return {}
        
        # 执行优化
        activities = training_set['activities']
        search_space = self.feature_extractor.get_param_search_space()
        
        optimized_params = self.optimizer.optimize(
            activities=activities,
            search_space=search_space,
            max_iterations=self.config['optimizer']['max_iterations']
        )
        
        # 保存结果
        result = {
            'best_params': optimized_params,
            'training_stats': {
                'activities_used': len(activities),
                'feature_count': len(training_set['feature_names']),
            },
        }
        
        # 保存模型
        self.optimizer.save_model(
            filepath=self.config['output']['save_path'] + '/model.json'
        )
        
        logger.info(f"Training complete. Best params: {optimized_params}")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.deviation_db.get_statistics()
        stats['enabled'] = self.enabled
        stats['algorithm'] = self.config['optimizer']['algorithm']
        
        return stats
    
    def reset(self):
        """重置学习器"""
        self.deviation_db._deviations = []
        self.deviation_db._activities = {}
        self.deviation_db._training_set = None
        self.optimizer.best_params = {}
        self.optimizer.history = []
        
        logger.info("ParameterLearner reset")
    
    def save(self, filepath: Optional[str] = None):
        """
        保存学习器状态
        
        Args:
            filepath: 文件路径（可选）
        """
        if filepath is None:
            filepath = self.config['output']['save_path'] + '/learner_state.json'
        
        # 保存偏差数据库
        self.deviation_db.save_to_disk()
        
        # 保存优化器
        self.optimizer.save_model(filepath)
        
        logger.info(f"Saved learner state to {filepath}")
    
    def load(self, filepath: Optional[str] = None):
        """
        加载学习器状态
        
        Args:
            filepath: 文件路径（可选）
        """
        if filepath is None:
            filepath = self.config['output']['save_path'] + '/learner_state.json'
        
        # 从磁盘加载偏差数据库
        self.deviation_db.load_from_disk()
        
        # 加载优化器
        self.optimizer.load_model(filepath)
        
        logger.info(f"Loaded learner state from {filepath}")
