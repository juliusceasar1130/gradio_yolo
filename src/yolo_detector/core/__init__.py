# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建core包初始化文件
"""
核心功能模块

包含检测器、图像处理器、结果处理器等核心组件
"""

from .detector import BaseDetector, ObjectDetector, SegmentationDetector, PoseDetector, DetectionResult
from .image_processor import ImageProcessor
from .result_processor import ResultProcessor
from .batch_processor import BatchProcessor

__all__ = [
    "BaseDetector",
    "ObjectDetector",
    "SegmentationDetector",
    "PoseDetector",
    "DetectionResult",
    "ImageProcessor",
    "ResultProcessor",
    "BatchProcessor"
]
