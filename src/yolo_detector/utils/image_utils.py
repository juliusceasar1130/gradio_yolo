# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建图像处理工具函数模块

"""
图像处理工具函数

提供图像加载、处理、转换等常用功能
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    加载图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        图像数组，加载失败返回None
    """
    try:
        # 使用OpenCV加载图像
        image = cv2.imread(image_path)
        if image is not None:
            # 转换为RGB格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.debug(f"成功加载图像: {image_path}, 形状: {image.shape}")
            return image
        else:
            logger.warning(f"OpenCV无法加载图像: {image_path}")
            return None
    except Exception as e:
        logger.error(f"加载图像失败: {image_path}, 错误: {e}")
        return None


def load_image_pil(image_path: str) -> Optional[Image.Image]:
    """
    使用PIL加载图像文件
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        PIL图像对象，加载失败返回None
    """
    try:
        image = Image.open(image_path)
        logger.debug(f"成功加载PIL图像: {image_path}, 大小: {image.size}, 模式: {image.mode}")
        return image
    except Exception as e:
        logger.error(f"PIL加载图像失败: {image_path}, 错误: {e}")
        return None


def convert_color_space(image: np.ndarray, from_space: str, to_space: str) -> np.ndarray:
    """
    转换颜色空间
    
    Args:
        image: 输入图像
        from_space: 源颜色空间 ('BGR', 'RGB', 'GRAY', 'HSV')
        to_space: 目标颜色空间 ('BGR', 'RGB', 'GRAY', 'HSV')
        
    Returns:
        转换后的图像
    """
    if from_space == to_space:
        return image.copy()
    
    # 定义转换映射
    conversion_map = {
        ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
        ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
        ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
        ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
        ('BGR', 'HSV'): cv2.COLOR_BGR2HSV,
        ('RGB', 'HSV'): cv2.COLOR_RGB2HSV,
        ('HSV', 'BGR'): cv2.COLOR_HSV2BGR,
        ('HSV', 'RGB'): cv2.COLOR_HSV2RGB,
        ('GRAY', 'BGR'): cv2.COLOR_GRAY2BGR,
        ('GRAY', 'RGB'): cv2.COLOR_GRAY2RGB,
    }
    
    conversion_key = (from_space.upper(), to_space.upper())
    
    if conversion_key in conversion_map:
        try:
            converted = cv2.cvtColor(image, conversion_map[conversion_key])
            logger.debug(f"颜色空间转换成功: {from_space} -> {to_space}")
            return converted
        except Exception as e:
            logger.error(f"颜色空间转换失败: {e}")
            return image
    else:
        logger.warning(f"不支持的颜色空间转换: {from_space} -> {to_space}")
        return image


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """
    调整图像尺寸
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
        keep_aspect_ratio: 是否保持宽高比
        
    Returns:
        调整后的图像
    """
    try:
        if keep_aspect_ratio:
            # 计算缩放比例
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 调整尺寸
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 创建目标尺寸的画布并居中放置
            canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            # 直接调整到目标尺寸
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            
    except Exception as e:
        logger.error(f"图像尺寸调整失败: {e}")
        return image


def get_image_info(image: Union[np.ndarray, Image.Image, str]) -> Dict[str, Any]:
    """
    获取图像信息
    
    Args:
        image: 图像数组、PIL图像对象或图像路径
        
    Returns:
        包含图像信息的字典
    """
    info = {}
    
    try:
        if isinstance(image, str):
            # 图像路径
            pil_img = load_image_pil(image)
            if pil_img:
                info['width'], info['height'] = pil_img.size
                info['mode'] = pil_img.mode
                info['format'] = pil_img.format
                info['size_bytes'] = len(pil_img.tobytes())
                info['size_kb'] = info['size_bytes'] / 1024
                info['path'] = image
        elif isinstance(image, Image.Image):
            # PIL图像对象
            info['width'], info['height'] = image.size
            info['mode'] = image.mode
            info['format'] = image.format
            info['size_bytes'] = len(image.tobytes())
            info['size_kb'] = info['size_bytes'] / 1024
        elif isinstance(image, np.ndarray):
            # numpy数组
            if len(image.shape) == 3:
                info['height'], info['width'], info['channels'] = image.shape
            else:
                info['height'], info['width'] = image.shape
                info['channels'] = 1
            info['dtype'] = str(image.dtype)
            info['size_bytes'] = image.nbytes
            info['size_kb'] = info['size_bytes'] / 1024
        
        logger.debug(f"获取图像信息成功: {info}")
        return info
        
    except Exception as e:
        logger.error(f"获取图像信息失败: {e}")
        return {}


def format_image_info(image: Union[np.ndarray, Image.Image, str], 
                     filename: Optional[str] = None) -> str:
    """
    格式化图像信息为显示文本
    
    Args:
        image: 图像对象
        filename: 文件名
        
    Returns:
        格式化的图像信息文本
    """
    info = get_image_info(image)
    
    if not info:
        return "无法获取图像信息"
    
    text_parts = ["**图片信息**"]
    
    if filename:
        text_parts.append(f"- 文件名: {filename}")
    
    if 'width' in info and 'height' in info:
        text_parts.append(f"- 尺寸: {info['width']}×{info['height']} 像素")
    
    if 'channels' in info:
        text_parts.append(f"- 通道数: {info['channels']}")
    
    if 'mode' in info:
        text_parts.append(f"- 颜色模式: {info['mode']}")
    
    if 'size_kb' in info:
        if info['size_kb'] > 1024:
            size_mb = info['size_kb'] / 1024
            text_parts.append(f"- 大小: {size_mb:.2f} MB")
        else:
            text_parts.append(f"- 大小: {info['size_kb']:.2f} KB")
    
    return "\n".join(text_parts)


def validate_image_format(image_path: str, allowed_extensions: list = None) -> bool:
    """
    验证图像格式
    
    Args:
        image_path: 图像文件路径
        allowed_extensions: 允许的文件扩展名列表
        
    Returns:
        是否为有效的图像格式
    """
    if allowed_extensions is None:
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    try:
        # 检查文件扩展名
        import os
        _, ext = os.path.splitext(image_path.lower())
        
        if ext not in allowed_extensions:
            logger.warning(f"不支持的图像格式: {ext}")
            return False
        
        # 尝试加载图像验证
        pil_img = load_image_pil(image_path)
        if pil_img is None:
            return False
        
        logger.debug(f"图像格式验证通过: {image_path}")
        return True
        
    except Exception as e:
        logger.error(f"图像格式验证失败: {image_path}, 错误: {e}")
        return False


def create_thumbnail(image: Union[np.ndarray, Image.Image], 
                    size: Tuple[int, int] = (128, 128)) -> Optional[Image.Image]:
    """
    创建图像缩略图
    
    Args:
        image: 输入图像
        size: 缩略图尺寸
        
    Returns:
        缩略图PIL对象
    """
    try:
        if isinstance(image, np.ndarray):
            # 转换numpy数组为PIL图像
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_img = Image.fromarray(image)
        else:
            pil_img = image.copy()
        
        # 创建缩略图
        pil_img.thumbnail(size, Image.Resampling.LANCZOS)
        logger.debug(f"创建缩略图成功: {pil_img.size}")
        return pil_img
        
    except Exception as e:
        logger.error(f"创建缩略图失败: {e}")
        return None
