# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:40；主要修改内容：创建统一的图像处理流程

"""
图像处理器模块

提供标准化的图像输入、预处理和输出流程
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Optional, Tuple, Dict, Any, List
import logging
from pathlib import Path

from ..utils.image_utils import (
    load_image, load_image_pil, convert_color_space, 
    resize_image, get_image_info, validate_image_format
)

logger = logging.getLogger(__name__)


class ImageProcessor:
    """图像处理器类"""
    
    def __init__(self, config=None):
        """
        初始化图像处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        # 从配置获取支持的格式
        if config:
            ui_config = config.get_ui_config()
            if 'allowed_extensions' in ui_config:
                self.supported_formats = ui_config['allowed_extensions']
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str], 
                        target_size: Optional[Tuple[int, int]] = None,
                        normalize: bool = False,
                        keep_aspect_ratio: bool = True) -> Optional[np.ndarray]:
        """
        预处理图像
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            normalize: 是否归一化到[0,1]
            keep_aspect_ratio: 是否保持宽高比
            
        Returns:
            预处理后的图像数组
        """
        try:
            # 加载图像
            processed_image = self._load_image_unified(image)
            if processed_image is None:
                return None
            
            # 调整尺寸
            if target_size is not None:
                processed_image = resize_image(processed_image, target_size, keep_aspect_ratio)
            
            # 归一化
            if normalize:
                processed_image = self._normalize_image(processed_image)
            
            logger.debug(f"图像预处理完成: {processed_image.shape}")
            return processed_image
            
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return None
    
    def _load_image_unified(self, image: Union[np.ndarray, Image.Image, str]) -> Optional[np.ndarray]:
        """统一的图像加载方法"""
        if isinstance(image, np.ndarray):
            # 已经是numpy数组
            return image.copy()
        elif isinstance(image, Image.Image):
            # PIL图像转numpy数组
            return np.array(image)
        elif isinstance(image, str):
            # 文件路径
            return load_image(image)
        else:
            logger.error(f"不支持的图像类型: {type(image)}")
            return None
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """归一化图像到[0,1]范围"""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.float32 or image.dtype == np.float64:
            # 假设已经在合理范围内
            return image.astype(np.float32)
        else:
            logger.warning(f"未知的图像数据类型: {image.dtype}")
            return image.astype(np.float32)
    
    def postprocess_result(self, result_image: np.ndarray, 
                          output_format: str = 'RGB',
                          quality: int = 95) -> Optional[np.ndarray]:
        """
        后处理结果图像
        
        Args:
            result_image: 结果图像
            output_format: 输出格式 ('RGB', 'BGR', 'GRAY')
            quality: 图像质量 (1-100)
            
        Returns:
            后处理的图像
        """
        try:
            if result_image is None:
                return None
            
            # 确保数据类型正确
            if result_image.dtype != np.uint8:
                if result_image.max() <= 1.0:
                    # 归一化的图像
                    result_image = (result_image * 255).astype(np.uint8)
                else:
                    result_image = result_image.astype(np.uint8)
            
            # 转换颜色格式
            if len(result_image.shape) == 3:
                current_format = 'BGR' if result_image.shape[2] == 3 else 'RGB'
                if current_format != output_format:
                    result_image = convert_color_space(result_image, current_format, output_format)
            
            logger.debug(f"图像后处理完成: {result_image.shape}, 格式: {output_format}")
            return result_image
            
        except Exception as e:
            logger.error(f"图像后处理失败: {e}")
            return result_image
    
    def validate_image_input(self, image: Union[np.ndarray, Image.Image, str]) -> Dict[str, Any]:
        """
        验证图像输入
        
        Args:
            image: 输入图像
            
        Returns:
            验证结果字典
        """
        result = {
            'valid': False,
            'error': None,
            'info': {},
            'warnings': []
        }
        
        try:
            if image is None:
                result['error'] = "输入图像为None"
                return result
            
            if isinstance(image, str):
                # 文件路径验证
                if not Path(image).exists():
                    result['error'] = f"图像文件不存在: {image}"
                    return result
                
                if not validate_image_format(image, self.supported_formats):
                    result['error'] = f"不支持的图像格式: {Path(image).suffix}"
                    return result
                
                # 获取文件信息
                result['info'] = get_image_info(image)
                
            elif isinstance(image, Image.Image):
                # PIL图像验证
                if image.size == (0, 0):
                    result['error'] = "PIL图像尺寸为0"
                    return result
                
                result['info'] = get_image_info(image)
                
            elif isinstance(image, np.ndarray):
                # numpy数组验证
                if image.size == 0:
                    result['error'] = "numpy图像数组为空"
                    return result
                
                if len(image.shape) not in [2, 3]:
                    result['error'] = f"不支持的图像维度: {image.shape}"
                    return result
                
                result['info'] = get_image_info(image)
                
            else:
                result['error'] = f"不支持的图像类型: {type(image)}"
                return result
            
            # 检查图像尺寸
            info = result['info']
            if 'width' in info and 'height' in info:
                width, height = info['width'], info['height']
                
                if width <= 0 or height <= 0:
                    result['error'] = f"无效的图像尺寸: {width}x{height}"
                    return result
                
                # 尺寸警告
                if width > 4096 or height > 4096:
                    result['warnings'].append(f"图像尺寸较大: {width}x{height}，可能影响处理速度")
                
                if width < 32 or height < 32:
                    result['warnings'].append(f"图像尺寸较小: {width}x{height}，可能影响检测效果")
            
            # 检查文件大小
            if 'size_mb' in info:
                size_mb = info['size_mb']
                if size_mb > 50:
                    result['warnings'].append(f"图像文件较大: {size_mb:.1f}MB，可能影响处理速度")
            
            result['valid'] = True
            logger.debug(f"图像验证通过: {result['info']}")
            return result
            
        except Exception as e:
            result['error'] = f"图像验证异常: {e}"
            logger.error(result['error'])
            return result
    
    def create_image_batch(self, images: List[Union[np.ndarray, Image.Image, str]], 
                          batch_size: int = 8,
                          target_size: Optional[Tuple[int, int]] = None) -> List[List[np.ndarray]]:
        """
        创建图像批次
        
        Args:
            images: 图像列表
            batch_size: 批次大小
            target_size: 目标尺寸
            
        Returns:
            图像批次列表
        """
        batches = []
        current_batch = []
        
        try:
            for i, image in enumerate(images):
                # 预处理图像
                processed_image = self.preprocess_image(image, target_size)
                
                if processed_image is not None:
                    current_batch.append(processed_image)
                    
                    # 检查是否达到批次大小
                    if len(current_batch) >= batch_size:
                        batches.append(current_batch)
                        current_batch = []
                else:
                    logger.warning(f"跳过无效图像: {i}")
            
            # 添加最后一个批次
            if current_batch:
                batches.append(current_batch)
            
            logger.info(f"创建了 {len(batches)} 个批次，总计 {len(images)} 张图像")
            return batches
            
        except Exception as e:
            logger.error(f"创建图像批次失败: {e}")
            return []
    
    def save_image(self, image: np.ndarray, output_path: str, 
                   quality: int = 95, format: str = None) -> bool:
        """
        保存图像
        
        Args:
            image: 图像数组
            output_path: 输出路径
            quality: 图像质量
            format: 图像格式
            
        Returns:
            是否保存成功
        """
        try:
            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为PIL图像
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 确保RGB格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image, 'RGB')
            elif len(image.shape) == 2:
                pil_image = Image.fromarray(image, 'L')
            else:
                logger.error(f"不支持的图像格式: {image.shape}")
                return False
            
            # 保存图像
            save_kwargs = {}
            if format is None:
                format = output_path.suffix.lower().lstrip('.')
            
            if format in ['jpg', 'jpeg']:
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
                pil_image.save(output_path, format='JPEG', **save_kwargs)
            elif format == 'png':
                save_kwargs['optimize'] = True
                pil_image.save(output_path, format='PNG', **save_kwargs)
            else:
                pil_image.save(output_path, **save_kwargs)
            
            logger.info(f"图像保存成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存图像失败: {output_path}, 错误: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            'supported_formats': self.supported_formats,
            'config_loaded': self.config is not None
        }
