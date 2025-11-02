# 创建者/修改者: chenliang
# 修改时间：2025年7月27日 22:32
# 主要修改内容：
# 1. 创建配置管理模块
# 2. 2025-01-27: 添加get_tool_pose_config()方法，支持从configs/default.yaml读取工具关键点配置
# 3. 2025-01-27: 修改get_tool_pose_config()从default.yaml读取，统一配置管理

"""
配置管理模块

负责加载、验证和管理项目配置
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为 configs/default.yaml
        """
        if config_path is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent.parent
            
            # 检查是否在Docker环境中，优先使用docker.yaml
            docker_config = project_root / "configs" / "docker.yaml"
            default_config = project_root / "configs" / "default.yaml"
            
            if docker_config.exists() and os.getenv('DOCKER_ENV', '').lower() == 'true':
                config_path = docker_config
                logger.info("检测到Docker环境，使用docker.yaml配置")
            else:
                config_path = default_config
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 处理环境变量覆盖
            config = self._apply_env_overrides(config)
            
            logger.info(f"成功加载配置文件: {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        # 检查常用的环境变量覆盖
        env_mappings = {
            'YOLO_DETECTION_MODEL': ['models', 'detection', 'path'],
            'YOLO_SEGMENTATION_MODEL': ['models', 'segmentation', 'path'],
            'YOLO_INPUT_FOLDER': ['data', 'input_folder'],
            'YOLO_OUTPUT_FOLDER': ['data', 'output_folder'],
            'YOLO_CONFIDENCE': ['detection', 'confidence_threshold'],
            'YOLO_DEVICE': ['system', 'device']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_value(config, config_path, env_value)
                logger.info(f"环境变量覆盖: {env_var} -> {'.'.join(config_path)}")
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any):
        """设置嵌套字典的值"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 尝试转换数据类型
        if isinstance(current.get(path[-1]), (int, float)):
            try:
                value = type(current[path[-1]])(value)
            except (ValueError, TypeError):
                pass
        elif isinstance(current.get(path[-1]), bool):
            value = value.lower() in ('true', '1', 'yes', 'on')
        
        current[path[-1]] = value
    
    def _validate_config(self):
        """验证配置的有效性"""
        # 处理空配置
        if self._config is None:
            self._config = {}

        required_sections = ['models', 'data', 'detection', 'ui']

        for section in required_sections:
            if section not in self._config:
                logger.warning(f"配置文件缺少节: {section}，将使用默认值")
                self._config[section] = {}
        
        # 验证模型路径
        for model_type in ['detection', 'segmentation']:
            if model_type in self._config['models']:
                model_path = self._config['models'][model_type]['path']
                if not os.path.exists(model_path):
                    logger.warning(f"{model_type}模型文件不存在: {model_path}")
        
        # 验证数据路径
        if 'input_folder' in self._config['data']:
            input_folder = self._config['data']['input_folder']
            if not os.path.exists(input_folder):
                logger.warning(f"输入文件夹不存在: {input_folder}")

        # 确保输出文件夹存在
        if 'output_folder' in self._config['data']:
            output_folder = self._config['data']['output_folder']
            os.makedirs(output_folder, exist_ok=True)
        
        logger.info("配置验证完成")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的嵌套键
        
        Args:
            key: 配置键，支持 'models.detection.path' 格式
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        current = self._config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，支持 'models.detection.path' 格式
            value: 配置值
        """
        keys = key.split('.')
        self._set_nested_value(self._config, keys, value)
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取指定模型的配置"""
        return self.get(f'models.{model_type}', {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """获取检测参数配置"""
        return self.get('detection', {})
    
    def get_segmentation_config(self) -> Dict[str, Any]:
        """获取分割参数配置"""
        return self.get('segmentation', {})
    
    def get_classification_config(self) -> Dict[str, Any]:
        """获取分类参数配置"""
        return self.get('classification', {})
    
    def get_pose_config(self) -> Dict[str, Any]:
        """获取姿态检测参数配置"""
        return self.get('pose', {})
    
    def get_tool_pose_config(self) -> Dict[str, Any]:
        """
        从default.yaml的pose.keypoints读取工具关键点配置
        
        Returns:
            包含工具关键点配置的字典，包括：
            - kpt_shape: 关键点形状 [数量, 维度]
            - num_classes: 类别数量
            - class_names: 类别名称映射
            - kpt_names: 每个类别的关键点名称字典
            - skeleton: 骨架连接列表
            - flip_idx: 水平翻转索引列表
            - pose_weights: 关键点权重列表
        """
        try:
            # 从default.yaml读取配置
            pose_config = self.get_pose_config()
            keypoints_config = pose_config.get('keypoints', {})
            
            if not keypoints_config:
                raise ValueError("配置中未找到pose.keypoints配置")
            
            # 解析配置并返回结构化数据
            kpt_shape = keypoints_config.get('kpt_shape', [7, 3])
            num_classes = keypoints_config.get('num_classes', 2)
            class_names = keypoints_config.get('class_names', {})
            kpt_names = keypoints_config.get('kpt_names', {})
            skeleton = keypoints_config.get('skeleton', [])
            flip_idx = keypoints_config.get('flip_idx', [])
            pose_weights = keypoints_config.get('pose_weights', [1.0] * kpt_shape[0])
            
            # 验证基本配置
            if not isinstance(kpt_shape, list) or len(kpt_shape) != 2:
                raise ValueError(f"kpt_shape必须是[数量, 维度]格式，当前: {kpt_shape}")
            
            num_kpts = kpt_shape[0]
            
            # 验证flip_idx
            if flip_idx:
                if len(flip_idx) != num_kpts:
                    logger.warning(f"flip_idx长度({len(flip_idx)})与关键点数量({num_kpts})不匹配，将使用默认值")
                    flip_idx = []
                else:
                    # 验证自反性（可选，仅警告）
                    for i in range(len(flip_idx)):
                        if flip_idx[flip_idx[i]] != i:
                            logger.warning(f"flip_idx在索引{i}处自反性验证失败")
            
            # 验证pose_weights
            if len(pose_weights) != num_kpts:
                logger.warning(f"pose_weights长度({len(pose_weights)})与关键点数量({num_kpts})不匹配，将使用默认值")
                pose_weights = [1.0] * num_kpts
            
            # 验证关键点名称
            for class_id, names in kpt_names.items():
                if len(names) != num_kpts:
                    logger.warning(f"类别{class_id}的关键点数量({len(names)})与kpt_shape({num_kpts})不匹配")
            
            result = {
                'kpt_shape': kpt_shape,
                'num_classes': num_classes,
                'class_names': class_names,
                'kpt_names': kpt_names,
                'skeleton': skeleton,
                'flip_idx': flip_idx if flip_idx else list(range(num_kpts)),  # 如果没有配置，使用默认（不翻转）
                'pose_weights': pose_weights
            }
            
            logger.info(f"成功加载工具关键点配置: {num_classes}个类别, {num_kpts}个关键点")
            return result
            
        except Exception as e:
            logger.error(f"读取工具关键点配置失败: {e}")
            # 返回默认配置
            logger.warning("使用默认工具关键点配置")
            return {
                'kpt_shape': [7, 3],
                'num_classes': 2,
                'class_names': {0: 'tool1', 1: 'tool2'},
                'kpt_names': {
                    0: ['t1_1', 't1_2', 't1_3', 't1_4', 't1_5', 't1_6', 't1_7'],
                    1: ['t2_1', 't2_2', 't2_3', 't2_4', 't2_5', 't2_6', 't2_7']
                },
                'skeleton': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0]],
                'flip_idx': [0, 2, 1, 4, 3, 5, 6],
                'pose_weights': [1.0] * 7
            }
    
    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI配置"""
        return self.get('ui', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据路径配置"""
        return self.get('data', {})
    
    def get_batch_config(self) -> Dict[str, Any]:
        """获取批量处理配置"""
        return self.get('batch_processing', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.get('logging', {})
    
    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return self.get('system', {})
    
    def save(self, path: Optional[str] = None):
        """保存配置到文件"""
        save_path = Path(path) if path else self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            logger.info(f"配置已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def reload(self):
        """重新加载配置"""
        self._config = self._load_config()
        self._validate_config()
        logger.info("配置已重新加载")
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return f"Config(path={self.config_path})"
    
    def __repr__(self) -> str:
        return self.__str__()


# 全局配置实例
_global_config = None


def get_config(config_path: Optional[str] = None) -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None or config_path is not None:
        _global_config = Config(config_path)
    return _global_config


def reload_config():
    """重新加载全局配置"""
    global _global_config
    if _global_config:
        _global_config.reload()
