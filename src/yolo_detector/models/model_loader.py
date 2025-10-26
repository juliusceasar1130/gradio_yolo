# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建YOLO模型加载器模块

"""
YOLO模型加载器

负责YOLO模型的加载、管理和初始化
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics未安装，模型加载功能将不可用")

logger = logging.getLogger(__name__)


class ModelLoader:
    """YOLO模型加载器类"""
    
    def __init__(self, config=None):
        """
        初始化模型加载器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.models = {}  # 缓存已加载的模型
        self._validate_ultralytics()
    
    def _validate_ultralytics(self):
        """验证ultralytics是否可用"""
        if YOLO is None:
            raise ImportError(
                "ultralytics库未安装。请运行: pip install ultralytics"
            )
    
    def load_model(self, model_type: str, model_path: Optional[str] = None, 
                   force_reload: bool = False) -> Optional[YOLO]:
        """
        加载指定类型的模型
        
        Args:
            model_type: 模型类型 ('detection', 'segmentation', 'classification')
            model_path: 模型文件路径，如果为None则从配置中获取
            force_reload: 是否强制重新加载
            
        Returns:
            YOLO模型对象，加载失败返回None
        """
        try:
            # 检查是否已缓存且不强制重新加载
            if not force_reload and model_type in self.models:
                logger.debug(f"使用缓存的{model_type}模型")
                return self.models[model_type]
            
            # 获取模型路径
            if model_path is None:
                if self.config is None:
                    raise ValueError("未提供配置对象且未指定模型路径")
                
                model_config = self.config.get_model_config(model_type)
                if not model_config:
                    raise ValueError(f"配置中未找到{model_type}模型配置")
                
                model_path = model_config.get('path')
                if not model_path:
                    raise ValueError(f"配置中未指定{model_type}模型路径")
            
            # 验证模型文件
            if not self._validate_model_path(model_path):
                return None
            
            # 加载模型
            logger.info(f"开始加载{model_type}模型: {model_path}")
            model = YOLO(model_path)
            
            # 验证模型类型
            if not self._validate_model_type(model, model_type):
                logger.error(f"模型类型验证失败: {model_type}")
                return None
            
            # 缓存模型
            self.models[model_type] = model
            
            logger.info(f"成功加载{model_type}模型: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"加载{model_type}模型失败: {e}")
            return None
    
    def _validate_model_path(self, model_path: str) -> bool:
        """验证模型文件路径"""
        try:
            path = Path(model_path)
            
            if not path.exists():
                logger.error(f"模型文件不存在: {model_path}")
                return False
            
            if not path.is_file():
                logger.error(f"模型路径不是文件: {model_path}")
                return False
            
            if path.suffix.lower() not in ['.pt', '.onnx', '.engine']:
                logger.warning(f"模型文件格式可能不支持: {path.suffix}")
            
            # 检查文件大小
            file_size = path.stat().st_size
            if file_size == 0:
                logger.error(f"模型文件为空: {model_path}")
                return False
            
            logger.debug(f"模型文件验证通过: {model_path} ({file_size / 1024 / 1024:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"模型文件验证失败: {model_path}, 错误: {e}")
            return False
    
    def _validate_model_type(self, model: YOLO, expected_type: str) -> bool:
        """验证模型类型是否匹配"""
        try:
            # 获取模型任务类型
            model_task = getattr(model, 'task', None)
            
            # 定义任务类型映射
            task_mapping = {
                'detection': ['detect', 'detection'],
                'segmentation': ['segment', 'segmentation'],
                'classification': ['classify', 'classification', 'cls'],
                'pose': ['pose']  # 新增
            }
            
            expected_tasks = task_mapping.get(expected_type, [])
            
            if model_task in expected_tasks:
                logger.debug(f"模型类型验证通过: {model_task} -> {expected_type}")
                return True
            else:
                logger.warning(f"模型类型不匹配: 期望 {expected_type}, 实际 {model_task}")
                # 不严格验证，允许继续使用
                return True
                
        except Exception as e:
            logger.warning(f"模型类型验证出错: {e}")
            # 验证出错时允许继续使用
            return True
    
    def get_model(self, model_type: str) -> Optional[YOLO]:
        """
        获取已加载的模型
        
        Args:
            model_type: 模型类型
            
        Returns:
            YOLO模型对象，未加载返回None
        """
        return self.models.get(model_type)
    
    def is_model_loaded(self, model_type: str) -> bool:
        """检查模型是否已加载"""
        return model_type in self.models
    
    def unload_model(self, model_type: str) -> bool:
        """
        卸载指定模型
        
        Args:
            model_type: 模型类型
            
        Returns:
            是否成功卸载
        """
        try:
            if model_type in self.models:
                del self.models[model_type]
                logger.info(f"成功卸载{model_type}模型")
                return True
            else:
                logger.warning(f"{model_type}模型未加载，无需卸载")
                return True
                
        except Exception as e:
            logger.error(f"卸载{model_type}模型失败: {e}")
            return False
    
    def unload_all_models(self):
        """卸载所有模型"""
        try:
            model_types = list(self.models.keys())
            for model_type in model_types:
                self.unload_model(model_type)
            logger.info("所有模型已卸载")
        except Exception as e:
            logger.error(f"卸载所有模型失败: {e}")
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_type: 模型类型
            
        Returns:
            模型信息字典
        """
        info = {
            'type': model_type,
            'loaded': self.is_model_loaded(model_type),
            'path': None,
            'task': None,
            'names': None
        }
        
        try:
            # 从配置获取路径
            if self.config:
                model_config = self.config.get_model_config(model_type)
                if model_config:
                    info['path'] = model_config.get('path')
            
            # 从已加载的模型获取信息
            model = self.get_model(model_type)
            if model:
                info['task'] = getattr(model, 'task', None)
                info['names'] = getattr(model, 'names', None)
                
                # 获取模型文件信息
                if hasattr(model, 'ckpt_path'):
                    model_path = model.ckpt_path
                    if os.path.exists(model_path):
                        stat = os.stat(model_path)
                        info['file_size'] = stat.st_size
                        info['file_size_mb'] = stat.st_size / 1024 / 1024
            
            return info
            
        except Exception as e:
            logger.error(f"获取{model_type}模型信息失败: {e}")
            return info
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """列出所有可用的模型"""
        available_models = {}
        
        if self.config:
            models_config = self.config.get('models', {})
            for model_type, model_config in models_config.items():
                available_models[model_type] = {
                    'path': model_config.get('path'),
                    'description': model_config.get('description', ''),
                    'loaded': self.is_model_loaded(model_type),
                    'exists': os.path.exists(model_config.get('path', ''))
                }
        
        return available_models
    
    def preload_models(self, model_types: list = None) -> Dict[str, bool]:
        """
        预加载模型
        
        Args:
            model_types: 要预加载的模型类型列表，None表示加载所有配置的模型
            
        Returns:
            加载结果字典
        """
        results = {}
        
        if model_types is None:
            if self.config:
                model_types = list(self.config.get('models', {}).keys())
            else:
                logger.warning("未提供模型类型列表且无配置对象")
                return results
        
        for model_type in model_types:
            try:
                model = self.load_model(model_type)
                results[model_type] = model is not None
                logger.info(f"预加载{model_type}模型: {'成功' if results[model_type] else '失败'}")
            except Exception as e:
                results[model_type] = False
                logger.error(f"预加载{model_type}模型失败: {e}")
        
        return results
    
    def __del__(self):
        """析构函数，清理资源"""
        try:
            self.unload_all_models()
        except:
            pass
