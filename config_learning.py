"""
学习配置 (Learning Configuration)

提供参数学习的配置选项，控制是否启用学习以及使用哪些数据。
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger('config.learning')


# 默认学习配置
DEFAULT_LEARNING_CONFIG: Dict[str, Any] = {
    # 是否启用参数学习
    'enabled': True,
    
    # 数据来源
    'data_source': {
        'history_count': 100,      # 使用过去 100 个活动的历史数据
        'min_activities': 50,      # 最少需要 50 个活动用于训练
        'data_dir': './tuner/data',
    },
    
    # 模型配置
    'model': {
        'type': 'random_forest',   # 或 'gradient_boosting'
        'max_depth': 10,
        'n_estimators': 100,
        'min_samples_split': 5,
    },
    
    # 优化配置
    'optimizer': {
        'algorithm': 'bayesian',   # 或 'grid', 'random'
        'max_iterations': 50,
        'early_stopping_patience': 10,
        'random_seed': 42,
    },
    
    # 输出配置
    'output': {
        'save_path': './tuner/models/',
        'log_level': 'INFO',
        'learned_params_file': 'learned_params.json',
    },
    
    # 参数搜索空间
    'search_space': {
        'weekend_multiplier': (0.8, 1.5, 0.05),
        'panic_ease_power': (0.5, 2.0, 0.1),
        'panic_scaler': (0.8, 2.0, 0.05),
        't_start_cmp': (3.0, 12.0, 0.5),
        't_end_cap': (48.0, 96.0, 6.0),
        'ratio_min': (0.1, 0.5, 0.05),
        'ratio_max': (2.0, 8.0, 0.2),
        'smooth_thresh1': (0.3, 0.7, 0.05),
        'smooth_thresh2': (0.5, 0.9, 0.05),
        'smooth_hard_cap': (0.6, 0.95, 0.02),
        'refit_weight_scale': (1.0, 50.0, 5.0),
    },
    
    # 评估指标
    'metrics': {
        'mae_threshold': 0.2,      # MAE 阈值
        'improvement_target': 0.05, # 目标改进率（5%）
    },
}


def get_learning_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    获取学习配置
    
    Args:
        config: 自定义配置字典
        
    Returns:
        合并后的配置字典
    """
    base_config = DEFAULT_LEARNING_CONFIG.copy()
    
    if config is not None:
        # 合并自定义配置
        for key, value in config.items():
            if key in base_config and isinstance(base_config[key], dict):
                base_config[key].update(value)
            else:
                base_config[key] = value
    
    logger.info(f"Learning config loaded: enabled={base_config['enabled']}")
    
    return base_config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        是否有效
    """
    required_keys = ['enabled', 'data_source', 'model', 'optimizer']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False
    
    # 验证数据源配置
    data_source = config.get('data_source', {})
    if data_source.get('history_count', 0) < data_source.get('min_activities', 10):
        logger.warning("history_count is less than min_activities")
    
    return True


def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置
    
    Returns:
        默认配置字典
    """
    return DEFAULT_LEARNING_CONFIG.copy()
