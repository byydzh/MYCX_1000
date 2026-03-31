"""
参数学习模块 (Tuner Module)

提供非侵入式的参数学习机制，使用过往活动的预测偏差值作为指标，
自动学习更优的参数配置。
"""

# 首先导入基础组件（不依赖其他模块）
from .deviation_db import DeviationDatabase
from .features import FeatureExtractor

# 然后导入优化器接口
from .optimizer import ParameterOptimizer

# 最后导入高级组件（依赖前面的模块）
try:
    from .bayesian_optimizer import BayesianOptimizer
except (ImportError, NameError):
    BayesianOptimizer = None

try:
    from .parameter_learner import ParameterLearner
except (ImportError, NameError):
    ParameterLearner = None

# 确保所有组件都被正确导出
__all__ = [
    'DeviationDatabase',
    'FeatureExtractor',
    'ParameterOptimizer',
    'BayesianOptimizer',
    'ParameterLearner',
]
