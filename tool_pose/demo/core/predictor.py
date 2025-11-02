#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测执行模块

主要修改内容：
1. 封装YOLO预测逻辑
2. 结果标准化
3. 错误处理
修改时间：2025年1月31日
最后更新：2025年11月2日
"""

from ultralytics import YOLO
from pathlib import Path
from typing import List

# 导入配置类（使用相对导入）
try:
    from ..config.settings import Config
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import Config


class Predictor:
    """
    预测执行器
    """
    
    @staticmethod
    def predict(
        model: YOLO,
        image_path: str,
        config: Config
    ) -> List:
        """
        执行预测
        
        Args:
            model: YOLO模型实例
            image_path: 输入图片路径
            config: 配置对象
            
        Returns:
            预测结果列表
        """
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 执行预测
        results = model.predict(
            source=str(image_path),
            conf=config.conf,
            iou=config.iou,
            imgsz=config.imgsz,
            device=config.device,
            half=config.half,
            max_det=config.max_det,
            verbose=False
        )
        
        if not results or len(results) == 0:
            raise ValueError("未检测到任何对象")
        
        return results
    
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

