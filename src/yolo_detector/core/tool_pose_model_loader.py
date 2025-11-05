# 创建者/修改者: chenliang
# 修改时间：2025年11月02日 10:00
# 主要修改内容：
# 1. 从 tool_pose/demo/core/model_loader.py 移植
# 2. 模型加载模块（单例模式）
# 3. 模型缓存（避免重复加载）
# 4. 设备管理

"""
模型加载模块（单例模式）

用于工具关键点检测的YOLO模型加载和管理
"""

from ultralytics import YOLO
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ToolPoseModelLoader:
    """
    模型加载器（单例模式）
    
    用于工具关键点检测的YOLO模型加载
    """
    _instance: Optional['ToolPoseModelLoader'] = None
    _model: Optional[YOLO] = None
    _model_path: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls, model_path: str, device: str = "") -> YOLO:
        """
        获取模型实例（支持缓存）
        
        Args:
            model_path: 模型文件路径
            device: 设备（"cuda" 或 "cpu"，空字符串表示自动选择）
            
        Returns:
            YOLO模型实例
        """
        # 如果模型已加载且路径相同，直接返回缓存的模型
        if cls._model is not None and cls._model_path == str(model_path):
            logger.debug(f"使用缓存的模型: {model_path}")
            return cls._model
        
        # 加载新模型
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        logger.info(f"正在加载模型: {model_path}")
        cls._model = YOLO(str(model_path))
        cls._model_path = str(model_path)
        logger.info(f"模型加载完成: {model_path}")
        
        return cls._model
    
    @classmethod
    def clear_cache(cls):
        """清除模型缓存"""
        cls._model = None
        cls._model_path = None
        logger.debug("模型缓存已清除")
    
    @classmethod
    def reload_model(cls, model_path: str, device: str = "") -> YOLO:
        """
        重新加载模型（清除缓存后加载）
        
        Args:
            model_path: 模型文件路径
            device: 设备
            
        Returns:
            YOLO模型实例
        """
        cls.clear_cache()
        return cls.get_model(model_path, device)

