"""
偏差数据库 (Deviation Database)

负责存储和管理历史活动的预测偏差数据，用于后续的参数学习。
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger('tuner.deviation_db')


class DeviationDatabase:
    """
    偏差数据库类
    
    职责：
    1. 存储历史活动的预测偏差数据
    2. 提供数据查询和过滤功能
    3. 管理训练数据集的构建
    """
    
    def __init__(self, data_dir: str = './tuner/data', 
                 max_activities: int = 100):
        """
        初始化偏差数据库
        
        Args:
            data_dir: 数据存储目录
            max_activities: 最大存储的活动数量（默认 100）
        """
        self.data_dir = data_dir
        self.max_activities = max_activities
        self._ensure_directory()
        
        # 活动数据缓存
        self._activities: Dict[int, Dict[str, Any]] = {}
        
        # 偏差记录
        self._deviations: List[Dict[str, Any]] = []
        
        # 训练数据集
        self._training_set: Optional[Dict[str, Any]] = None
        
        logger.info(f"DeviationDatabase initialized with max_activities={max_activities}")
    
    def _ensure_directory(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def record_prediction(self, activity_id: int, 
                         actual_final_score: float,
                         predicted_final_score: float,
                         prediction_params: Dict[str, float],
                         activity_features: Dict[str, Any]):
        """
        记录一次预测结果
        
        Args:
            activity_id: 活动 ID
            actual_final_score: 实际最终分数
            predicted_final_score: 预测的最终分数
            prediction_params: 使用的参数配置
            activity_features: 活动特征
        """
        deviation = {
            'activity_id': activity_id,
            'timestamp': datetime.now().isoformat(),
            'actual_final_score': actual_final_score,
            'predicted_final_score': predicted_final_score,
            'absolute_error': abs(predicted_final_score - actual_final_score),
            'relative_error': (predicted_final_score - actual_final_score) / actual_final_score if actual_final_score > 0 else 0,
            'prediction_params': prediction_params,
            'activity_features': activity_features,
        }
        
        self._deviations.append(deviation)
        
        # 存储活动数据（去重）
        if activity_id not in self._activities:
            if len(self._activities) >= self.max_activities:
                # 移除最旧的活动
                oldest_id = min(self._activities.keys(), key=lambda x: self._activities[x]['timestamp'])
                del self._activities[oldest_id]
            
            self._activities[activity_id] = {
                'actual_final_score': actual_final_score,
                'predicted_final_score': predicted_final_score,
                'prediction_params': prediction_params,
                'activity_features': activity_features,
                'timestamp': datetime.now().isoformat(),
            }
            
            # 限制活动数量
            if len(self._activities) >= self.max_activities:
                self._save_to_disk()
        
        logger.debug(f"Recorded deviation for activity {activity_id}: "
                     f"error={deviation['absolute_error']:.4f}")
    
    def get_recent_activities(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        获取最近的 N 个活动
        
        Args:
            count: 活动数量
            
        Returns:
            活动列表
        """
        # 按时间排序
        sorted_activities = sorted(
            self._activities.values(),
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        return sorted_activities[:count]
    
    def get_training_dataset(self) -> Optional[Dict[str, Any]]:
        """
        获取训练数据集
        
        Returns:
            训练数据集字典，或 None（如果数据不足）
        """
        if self._training_set is not None:
            return self._training_set
        
        activities = self.get_recent_activities(self.max_activities)
        
        if len(activities) < 10:  # 至少需要 10 个活动
            logger.warning(f"Not enough activities for training (have {len(activities)}, need at least 10)")
            return None
        
        # 构建特征矩阵和目标向量
        X = []
        y = []
        
        for activity in activities:
            features = activity['activity_features']
            
            # 提取数值特征
            feature_values = {
                'total_hours': features.get('total_hours', 0),
                'observed_points': features.get('observed_points', 0),
                'observed_coverage': features.get('observed_coverage', 0),
                'initial_ratio': features.get('initial_ratio', 1.0),
                'scale_factor': features.get('scale_factor', 1.0),
                'panic_duration': features.get('panic_duration', 24.0),
            }
            
            # 添加偏差特征
            feature_values['absolute_error'] = activity['absolute_error']
            feature_values['relative_error'] = activity['relative_error']
            
            X.append(feature_values)
            y.append(activity['absolute_error'])
        
        self._training_set = {
            'X': X,
            'y': y,
            'activities': activities,
            'feature_names': list(feature_values.keys()),
        }
        
        logger.info(f"Training dataset prepared with {len(activities)} activities")
        return self._training_set
    
    def save_to_disk(self):
        """保存到磁盘"""
        data = {
            'deviations': self._deviations,
            'activities': self._activities,
        }
        
        file_path = os.path.join(self.data_dir, 'deviations.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved deviations to {file_path}")
    
    def _save_to_disk(self):
        """内部方法：保存到磁盘"""
        self.save_to_disk()
    
    def load_from_disk(self):
        """从磁盘加载数据"""
        file_path = os.path.join(self.data_dir, 'deviations.json')
        
        if not os.path.exists(file_path):
            logger.info(f"No existing deviations data at {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._deviations = data.get('deviations', [])
            self._activities = data.get('activities', {})
            
            logger.info(f"Loaded {len(self._deviations)} deviations from disk")
        except Exception as e:
            logger.error(f"Failed to load deviations from disk: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        if not self._deviations:
            return {
                'total_activities': 0,
                'mean_absolute_error': 0,
                'max_absolute_error': 0,
                'min_absolute_error': 0,
            }
        
        errors = [d['absolute_error'] for d in self._deviations]
        
        return {
            'total_activities': len(self._deviations),
            'mean_absolute_error': sum(errors) / len(errors),
            'max_absolute_error': max(errors),
            'min_absolute_error': min(errors),
            'std_absolute_error': (sum((e - sum(errors)/len(errors))**2 for e in errors) / len(errors)) ** 0.5,
        }
