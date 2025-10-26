# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建yolo_detector包初始化文件
"""
YOLO检测工具 - 主包

提供基于YOLOv11的目标检测和图像分割功能
"""

__version__ = "1.0.0"
__author__ = "chenliang"

# 导出主要类和函数
from .config.settings import Config
from .models.model_loader import ModelLoader
from .core import (
    ObjectDetector, SegmentationDetector, PoseDetector, DetectionResult,
    ImageProcessor, ResultProcessor, BatchProcessor
)
from .ui import create_gradio_interface, GradioApp

__all__ = [
    "Config",
    "ModelLoader",
    "ObjectDetector",
    "SegmentationDetector",
    "PoseDetector",
    "DetectionResult",
    "ImageProcessor",
    "ResultProcessor",
    "BatchProcessor",
    "create_gradio_interface",
    "GradioApp"
]
