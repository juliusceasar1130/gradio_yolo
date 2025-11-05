# 创建者/修改者: chenliang
# 修改时间：2025年11月02日 10:00
# 主要修改内容：
# 1. 从 tool_pose/demo/core/predictor.py 移植
# 2. 封装YOLO预测逻辑
# 3. 结果标准化
# 4. 错误处理

"""
预测执行模块

用于工具关键点检测的YOLO预测封装
"""

from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ToolPosePredictor:
    """
    预测执行器
    
    用于工具关键点检测的YOLO预测
    """
    
    @staticmethod
    def predict(
        model: YOLO,
        source: Union[str, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        half: bool = False,
        max_det: int = 1000,
        verbose: bool = False
    ) -> List:
        """
        执行预测
        
        Args:
            model: YOLO模型实例
            source: 输入源（图片路径或numpy数组）
            conf: 置信度阈值
            iou: IoU阈值
            imgsz: 图像尺寸
            device: 设备
            half: 半精度推理
            max_det: 最大检测数量
            verbose: 是否输出详细信息
            
        Returns:
            预测结果列表
        """
        # 如果是文件路径，检查是否存在
        if isinstance(source, str):
            image_path_obj = Path(source)
            if not image_path_obj.exists():
                raise FileNotFoundError(f"图片文件不存在: {source}")
        
        # 执行预测
        try:
            results = model.predict(
                source=source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                half=half,
                max_det=max_det,
                verbose=verbose
            )
            
            if not results or len(results) == 0:
                logger.warning("未检测到任何对象")
                return []
            
            return results
            
        except Exception as e:
            logger.error(f"预测执行失败: {e}", exc_info=True)
            raise
    
    @staticmethod
    def validate_results(results: List) -> bool:
        """
        验证预测结果
        
        Args:
            results: 预测结果列表
            
        Returns:
            是否有效
        """
        if not results or len(results) == 0:
            return False
        
        result = results[0]
        
        # 检查是否有检测框
        if result.boxes is None or len(result.boxes) == 0:
            return False
        
        # 检查是否有关键点
        if result.keypoints is None or len(result.keypoints) == 0:
            return False
        
        return True

