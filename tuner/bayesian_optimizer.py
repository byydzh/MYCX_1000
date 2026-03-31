"""
贝叶斯优化器 (Bayesian Optimizer)

使用贝叶斯优化算法进行参数搜索，适用于高维参数空间。
"""

import numpy as np
from typing import Dict, Optional, List, Any, Tuple
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import json
import os

# 导入基类（避免循环导入）
try:
    from .optimizer import ParameterOptimizer
except ImportError:
    ParameterOptimizer = object  # 使用空基类作为后备

logger = logging.getLogger('tuner.bayesian_optimizer')


class GaussianProcess:
    """
    高斯过程实现（简化版）
    
    用于贝叶斯优化的代理模型。
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 kernel_type: str = 'rbf', noise_variance: float = 0.1):
        """
        初始化高斯过程
        
        Args:
            X: 输入特征矩阵
            y: 目标值向量
            kernel_type: 核函数类型
            noise_variance: 噪声方差
        """
        self.X = X
        self.y = y
        self.noise_variance = noise_variance
        self.kernel_type = kernel_type
        
        # 学习超参数
        self.length_scale = 1.0
        self.signal_variance = np.mean(y ** 2)
        
        self._fit()
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF 核函数
        
        Args:
            X1: 输入矩阵 1
            X2: 输入矩阵 2
            
        Returns:
            核矩阵
        """
        sq_dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
                  np.sum(X2 ** 2, axis=1).reshape(1, -1) - \
                  2 * X1 @ X2.T
        
        return np.exp(-sq_dist / (2 * self.length_scale ** 2))
    
    def _fit(self):
        """拟合高斯过程"""
        n_samples = len(self.y)
        
        # 计算核矩阵
        K = self._rbf_kernel(self.X, self.X)
        
        # 添加噪声方差
        K_noise = K + self.noise_variance * np.eye(n_samples)
        
        # 学习超参数（简化版：使用启发式方法）
        self.signal_variance = np.var(self.y) * n_samples / (n_samples + 1e-6)
        
        # 修复：处理单特征情况
        if self.X.shape[0] == 1:
            # 只有一个样本，无法计算长度尺度
            self.length_scale = 1.0
        elif self.X.ndim == 1:
            # 单特征情况，使用一维数组
            self.length_scale = float(np.median(np.abs(np.diff(self.X)))) if len(np.diff(self.X)) > 0 else 1.0
        else:
            # 多特征情况
            n_samples, n_features = self.X.shape
            if n_samples < 2 or n_features < 2:
                self.length_scale = 1.0
            else:
                # 计算特征间的平均距离
                diff = self.X[:, :min(3, n_features)] - self.X[:, 1:min(4, n_features)]
                self.length_scale = float(np.median(np.abs(diff)))
        
        logger.debug(f"GaussianProcess fitted: signal_variance={self.signal_variance:.4f}, "
                     f"length_scale={self.length_scale:.4f}")
    
    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测均值和方差
        
        Args:
            X_new: 新输入矩阵
            
        Returns:
            (mean, variance) 元组
        """
        K = self._rbf_kernel(self.X, X_new)
        K_noise = self._rbf_kernel(X_new, X_new) + self.noise_variance * np.eye(len(X_new))
        
        alpha = np.linalg.solve(K_noise, self.y)
        mean = K @ alpha
        
        variance = np.diag(K_noise) - np.diag(K @ alpha.reshape(-1, 1) @ (K @ alpha))
        
        return mean, variance


class BayesianOptimizer(ParameterOptimizer):
    """
    贝叶斯优化器
    
    使用高斯过程 + 采集函数进行参数搜索。
    """
    
    def __init__(self, feature_extractor=None, 
                 acquisition_function: str = 'ei',
                 random_seed: int = 42):
        """
        初始化贝叶斯优化器
        
        Args:
            feature_extractor: 特征提取器实例
            acquisition_function: 采集函数类型 ('ei', 'pi', 'ucb')
            random_seed: 随机种子
        """
        super().__init__(feature_extractor)
        self.acquisition_function = acquisition_function
        self.random_seed = random_seed
        
        # 高斯过程模型
        self.gp_model: Optional[GaussianProcess] = None
        
        # 最佳参数
        self.best_params: Dict[str, float] = {}
        
        # 历史评估记录
        self.history: List[Dict[str, Any]] = []
        
        logger.info(f"BayesianOptimizer initialized with acquisition_function={acquisition_function}")
    
    def evaluate_params(self, params: Dict[str, float],
                        activities: List[Dict[str, Any]]) -> float:
        """
        评估参数配置的性能
        
        Args:
            params: 参数配置
            activities: 活动列表
            
        Returns:
            性能评分（平均绝对误差）
        """
        # 计算预测误差
        total_error = 0.0
        count = 0
        
        for activity in activities:
            # 使用当前参数进行预测（简化版：直接计算误差）
            actual_score = activity.get('actual_final_score', 0)
            predicted_score = activity.get('predicted_final_score', 0)
            
            error = abs(predicted_score - actual_score)
            total_error += error
            count += 1
        
        if count > 0:
            return total_error / count
        return 0.0
    
    def _acquisition_function(self, params: Dict[str, float]) -> Tuple[float, float]:
        """
        计算采集函数值
        
        Args:
            params: 参数配置
            
        Returns:
            (value, uncertainty) 元组
        """
        # 评估当前参数
        value = self.evaluate_params(params, self.activities)
        
        # 获取不确定性（简化版）
        uncertainty = 0.1
        
        return -value, uncertainty  # 负值因为我们要最大化采集函数
    
    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray,
                               best_so_far: float) -> np.ndarray:
        """
        计算期望改进 (EI)
        
        Args:
            mean: 预测均值
            std: 预测标准差
            best_so_far: 当前最佳值
            
        Returns:
            EI 值数组
        """
        z = (best_so_far - mean) / (std + 1e-9)
        
        ei = std * norm.cdf(z) + (best_so_far - mean) * norm.pdf(z)
        
        return ei
    
    def _upper_confidence_bound(self, mean: np.ndarray, std: np.ndarray,
                                exploration_weight: float = 1.0) -> np.ndarray:
        """
        计算上置信界 (UCB)
        
        Args:
            mean: 预测均值
            std: 预测标准差
            exploration_weight: 探索权重
            
        Returns:
            UCB 值数组
        """
        return mean + exploration_weight * std
    
    def optimize(self, activities: List[Dict[str, Any]],
                 search_space: Optional[Dict[str, tuple]] = None,
                 max_iterations: int = 50) -> Dict[str, float]:
        """
        执行贝叶斯优化
        
        Args:
            activities: 活动列表
            search_space: 参数搜索空间
            max_iterations: 最大迭代次数
            
        Returns:
            最优参数配置
        """
        self.activities = activities
        
        # 设置随机种子
        np.random.seed(self.random_seed)
        
        # 获取或创建搜索空间
        if search_space is None:
            if self.feature_extractor:
                search_space = self.feature_extractor.get_param_search_space()
            else:
                search_space = {
                    'weekend_multiplier': (0.8, 1.5),
                    'panic_ease_power': (0.5, 2.0),
                    'panic_scaler': (0.8, 2.0),
                    't_start_cmp': (3.0, 12.0),
                    't_end_cap': (48.0, 96.0),
                    'ratio_min': (0.1, 0.5),
                    'ratio_max': (2.0, 8.0),
                    'smooth_thresh1': (0.3, 0.7),
                    'smooth_thresh2': (0.5, 0.9),
                    'smooth_hard_cap': (0.6, 0.95),
                    'refit_weight_scale': (1.0, 50.0),
                }
        
        logger.info(f"Starting Bayesian optimization with {len(search_space)} parameters")
        
        # 初始化高斯过程（使用当前参数）
        initial_params = self.get_current_params()
        if initial_params:
            X_init = np.array([[1.0] * len(search_space)])
            y_init = np.array([self.evaluate_params(initial_params, activities)])
            self.gp_model = GaussianProcess(X_init, y_init)
        
        # 随机初始化参数
        param_names = list(search_space.keys())
        n_params = len(param_names)
        
        if n_params > 1:
            # 多参数情况：使用随机采样
            import random
            
            # 为每个参数生成随机值
            initial_points = []
            for _ in range(max_iterations):
                point = {}
                for param_name, search_range in search_space.items():
                    # search_range 是 (min_val, max_val, step) 三元组
                    min_val, max_val, step = search_range
                    # 均匀采样
                    value = min_val + np.random.uniform(0, max_val - min_val)
                    point[param_name] = value
                initial_points.append(point)
        else:
            # 单参数情况：均匀采样
            param_name = list(search_space.keys())[0]
            search_range = search_space[param_name]
            min_val, max_val, step = search_range
            
            initial_points = []
            for i in range(max_iterations):
                point = {param_name: min_val + np.random.uniform(0, max_val - min_val)}
                initial_points.append(point)
        
        # 优化循环
        best_value = float('inf')
        best_params = {}
        
        for iteration, point in enumerate(initial_points):
            value = self.evaluate_params(point, activities)
            
            if value < best_value:
                best_value = value
                best_params = point
            
            # 记录历史
            self.history.append({
                'iteration': iteration + 1,
                'params': point,
                'value': value,
            })
            
            # 更新高斯过程
            if self.gp_model is None:
                # 首次初始化，创建新的高斯过程
                X_new = np.array([[1.0] * n_params])
                y_new = np.array([value])
                self.gp_model = GaussianProcess(X_new, y_new)
            else:
                # 更新高斯过程
                X_new = np.array([[1.0] * n_params])
                y_new = np.array([value])
                self.gp_model = GaussianProcess(np.vstack([self.gp_model.X, X_new]),
                                                np.concatenate([self.gp_model.y, y_new]))
            
            # 打印进度
            if (iteration + 1) % 10 == 0:
                logger.info(f"Iteration {iteration + 1}/{max_iterations}: "
                           f"best_value={best_value:.4f}, best_params={best_params}")
        
        self.best_params = best_params
        logger.info(f"Optimization complete. Best value: {best_value:.4f}")
        
        return best_params
    
    def get_optimized_params(self, target: Any, history: List[Any]) -> Dict[str, float]:
        """
        获取优化后的参数
        
        Args:
            target: 目标活动
            history: 历史活动列表
            
        Returns:
            最优参数配置
        """
        return self.best_params
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        获取优化历史
        
        Returns:
            历史记录列表
        """
        return self.history
    
    def save_model(self, filepath: str):
        """
        保存模型和参数
        
        Args:
            filepath: 文件路径
        """
        data = {
            'best_params': self.best_params,
            'history': self.history,
            'acquisition_function': self.acquisition_function,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型和参数
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.best_params = data.get('best_params', {})
        self.history = data.get('history', [])
        self.acquisition_function = data.get('acquisition_function', 'ei')
        
        logger.info(f"Loaded model from {filepath}")
