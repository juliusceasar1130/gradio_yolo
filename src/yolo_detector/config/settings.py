# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建配置管理模块

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
            config_path = project_root / "configs" / "default.yaml"
        
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
