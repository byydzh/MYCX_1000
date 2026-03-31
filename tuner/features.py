"""
特征提取器 (Feature Extractor)

负责从活动数据和预测结果中提取特征，用于参数学习模型的训练。
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger('tuner.features')


class FeatureExtractor:
    """
    特征提取器类
    
    职责：
    1. 从活动数据中提取基础特征
    2. 从预测结果中提取偏差相关特征
    3. 对特征进行标准化处理
    """
    
    # 可学习的参数列表
    LEARNABLE_PARAMS = [
        'weekend_multiplier',
        'panic_ease_power',
        'panic_scaler',
        't_start_cmp',
        't_end_cap',
        'ratio_min',
        'ratio_max',
        'smooth_thresh1',
        'smooth_thresh2',
        'smooth_hard_cap',
        'refit_weight_scale',
    ]
    
    # 默认参数值（用于特征计算）
    DEFAULT_PARAMS = {
        'weekend_multiplier': 1.1,
        'panic_ease_power': 1.0,
        'panic_scaler': 1.1,
        't_start_cmp': 6.0,
        't_end_cap': 72.0,
        'ratio_min': 0.25,
        'ratio_max': 4.0,
        'smooth_thresh1': 0.5,
        'smooth_thresh2': 0.65,
        'smooth_hard_cap': 0.8,
        'refit_weight_scale': 10.0,
    }
    
    def __init__(self, default_params: Optional[Dict[str, float]] = None):
        """
        初始化特征提取器
        
        Args:
            default_params: 默认参数配置
        """
        self.default_params = default_params or self.DEFAULT_PARAMS.copy()
        
        # 特征名称列表
        self.feature_names = [
            'total_hours',              # 活动总时长
            'observed_points',          # 观测数据点数
            'observed_coverage',        # 观测覆盖率
            'initial_ratio',            # 初始比率系数
            'scale_factor',             # Scale 因子
            'panic_duration',           # 恐慌期时长
            'start_hour',               # 开始小时
            'is_weekend',               # 是否为周末
            'absolute_error',           # 绝对误差
            'relative_error',           # 相对误差
        ]
        
        logger.info(f"FeatureExtractor initialized with {len(self.feature_names)} features")
    
    def extract_activity_features(self, activity_meta: Dict[str, Any],
                                  prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取活动特征
        
        Args:
            activity_meta: 活动元数据
            prediction_result: 预测结果
            
        Returns:
            特征字典
        """
        # 基础特征
        features = {
            'total_hours': activity_meta.get('total_hours', 0),
            'start_hour': activity_meta.get('start_hour', 0),
            'is_weekend': 1.0 if activity_meta.get('is_weekend', False) else 0.0,
        }
        
        # 观测数据特征
        df = activity_meta.get('df', {})
        if isinstance(df, dict):
            points = len(df)
        elif hasattr(df, '__len__'):
            points = len(df)
        else:
            points = 0
        
        features.update({
            'observed_points': points,
            'observed_coverage': points / max(activity_meta.get('total_hours', 1), 1),
        })
        
        # 预测强度特征
        features.update({
            'initial_ratio': prediction_result.get('ratio', 1.0),
            'scale_factor': prediction_result.get('scale_factor', 1.0),
            'panic_duration': activity_meta.get('total_hours', 0) - 
                             activity_meta.get('panic_start_time', 0),
        })
        
        # 偏差特征
        features.update({
            'absolute_error': abs(
                prediction_result.get('predicted_final_score', 0) -
                prediction_result.get('actual_final_score', 0)
            ),
            'relative_error': (
                prediction_result.get('predicted_final_score', 0) -
                prediction_result.get('actual_final_score', 0)
            ) / max(prediction_result.get('actual_final_score', 1), 1),
        })
        
        return features
    
    def extract_param_features(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        提取参数特征（用于偏差预测）
        
        Args:
            params: 参数配置
            
        Returns:
            参数特征字典
        """
        param_features = {}
        
        for param_name in self.LEARNABLE_PARAMS:
            if param_name in params:
                # 将参数值映射到 [0, 1] 范围
                min_val = self.default_params.get(f'{param_name}_min', 0)
                max_val = self.default_params.get(f'{param_name}_max', 10)
                
                param_value = params.get(param_name, self.default_params[param_name])
                normalized_value = (param_value - min_val) / (max_val - min_val + 1e-9)
                
                param_features[f'{param_name}_normalized'] = normalized_value
        
        return param_features
    
    def standardize_features(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        对特征列表进行标准化
        
        Args:
            features_list: 特征列表
            
        Returns:
            标准化后的特征字典（包含均值和标准差）
        """
        if not features_list:
            return {}
        
        # 计算每个特征的均值和标准差
        stats = {}
        for feature_name in self.feature_names:
            values = [f.get(feature_name, 0) for f in features_list]
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            stats[feature_name] = {
                'mean': mean_val,
                'std': std_val,
            }
        
        # 应用标准化
        standardized = []
        for features in features_list:
            std_features = {}
            for feature_name in self.feature_names:
                if feature_name in stats:
                    mean_val = stats[feature_name]['mean']
                    std_val = stats[feature_name]['std']
                    
                    value = features.get(feature_name, 0)
                    std_features[feature_name] = (value - mean_val) / max(std_val, 1e-9)
                else:
                    std_features[feature_name] = features.get(feature_name, 0)
            
            standardized.append(std_features)
        
        return {
            'features': standardized,
            'stats': stats,
        }
    
    def get_feature_matrix(self, activities: List[Dict[str, Any]]) -> np.ndarray:
        """
        从活动列表构建特征矩阵
        
        Args:
            activities: 活动列表
            
        Returns:
            特征矩阵 (n_samples, n_features)
        """
        features_list = []
        
        for activity in activities:
            meta = activity.get('meta', {})
            prediction = activity.get('prediction_result', {})
            
            features = self.extract_activity_features(meta, prediction)
            features_list.append(features)
        
        if not features_list:
            return np.array([])
        
        # 转换为 numpy 数组
        feature_names = list(features_list[0].keys())
        X = np.array([[f.get(fn, 0) for fn in feature_names] for f in features_list])
        
        return X
    
    def get_target_vector(self, activities: List[Dict[str, Any]]) -> np.ndarray:
        """
        从活动列表构建目标向量
        
        Args:
            activities: 活动列表
            
        Returns:
            目标向量 (n_samples,)
        """
        targets = []
        
        for activity in activities:
            prediction = activity.get('prediction_result', {})
            
            # 使用绝对误差作为目标
            actual_score = prediction.get('actual_final_score', 0)
            predicted_score = prediction.get('predicted_final_score', 0)
            
            error = abs(predicted_score - actual_score)
            targets.append(error)
        
        return np.array(targets)
    
    def get_param_search_space(self) -> Dict[str, tuple]:
        """
        获取参数搜索空间
        
        Returns:
            参数搜索空间字典，值为 (min, max, step) 元组
        """
        search_space = {}
        
        for param_name in self.LEARNABLE_PARAMS:
            # 设置合理的搜索范围
            if param_name == 'weekend_multiplier':
                search_space[param_name] = (0.8, 1.5, 0.05)
            elif param_name == 'panic_ease_power':
                search_space[param_name] = (0.5, 2.0, 0.1)
            elif param_name == 'panic_scaler':
                search_space[param_name] = (0.8, 2.0, 0.05)
            elif param_name == 't_start_cmp':
                search_space[param_name] = (3.0, 12.0, 0.5)
            elif param_name == 't_end_cap':
                search_space[param_name] = (48.0, 96.0, 6.0)
            elif param_name == 'ratio_min':
                search_space[param_name] = (0.1, 0.5, 0.05)
            elif param_name == 'ratio_max':
                search_space[param_name] = (2.0, 8.0, 0.2)
            elif param_name == 'smooth_thresh1':
                search_space[param_name] = (0.3, 0.7, 0.05)
            elif param_name == 'smooth_thresh2':
                search_space[param_name] = (0.5, 0.9, 0.05)
            elif param_name == 'smooth_hard_cap':
                search_space[param_name] = (0.6, 0.95, 0.02)
            elif param_name == 'refit_weight_scale':
                search_space[param_name] = (1.0, 50.0, 5.0)
        
        return search_space
