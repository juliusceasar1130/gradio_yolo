#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像工具模块

主要修改内容：
1. 图像格式转换
2. 图像显示
3. 图像保存
修改时间：2025年1月31日
最后更新：2025年11月2日
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    将BGR格式图像转换为RGB格式（YOLO返回的是BGR格式）
    
    Args:
        image: 输入图像（BGR格式）
        
    Returns:
        RGB格式图像
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return image


def display_image(
    image: np.ndarray,
    title: str = "图像显示",
    figsize: tuple[int, int] = (12, 8),
    show: bool = True
) -> None:
    """
    使用matplotlib显示图像
    
    Args:
        image: 图像数组（BGR或RGB格式）
        title: 图像标题
        figsize: 图像显示尺寸
        show: 是否立即显示
    """
    # 转换为RGB格式
    image_rgb = convert_bgr_to_rgb(image)
    
    # 显示图像
    plt.figure(figsize=figsize)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if show:
        plt.show()


def save_image(
    image: np.ndarray,
    output_path: Path,
    is_bgr: bool = True
) -> bool:
    """
    保存图像到文件
    
    Args:
        image: 图像数组
        output_path: 输出文件路径
        is_bgr: 是否为BGR格式（cv2.imwrite需要BGR格式）
        
    Returns:
        是否保存成功
    """
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # cv2.imwrite需要BGR格式
        if not is_bgr and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(output_path), image)
        return True
    except Exception as e:
        print(f"❌ 保存图像失败: {e}")
        return False

