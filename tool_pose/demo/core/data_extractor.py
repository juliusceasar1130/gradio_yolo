#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果数据提取模块

主要修改内容：
1. 从YOLO结果中提取结构化数据
2. 数据格式转换
修改时间：2025年1月31日
最后更新：2025年11月2日
"""

import numpy as np
from typing import Dict, List, Any, Optional
from ultralytics.engine.results import Results


class DataExtractor:
    """
    数据提取器
    """
    
    @staticmethod
    def extract(results: List[Results]) -> Dict[str, Any]:
        """
        从预测结果中提取结构化数据
        
        Args:
            results: YOLO预测结果列表
            
        Returns:
            提取的结构化数据字典
        """
        if not results or len(results) == 0:
            return {}
        
        result = results[0]
        
        extracted_data = {
            'image_shape': result.orig_shape,
            'boxes': [],
            'keypoints': [],
            'classes': []
        }
        
        if result.boxes is not None and len(result.boxes) > 0:
            # 提取边界框数据
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
            boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(result.boxes)):
                extracted_data['boxes'].append({
                    'index': i,
                    'class_id': int(boxes_cls[i]),
                    'confidence': float(boxes_conf[i]),
                    'bbox': {
                        'x1': float(boxes_xyxy[i][0]),
                        'y1': float(boxes_xyxy[i][1]),
                        'x2': float(boxes_xyxy[i][2]),
                        'y2': float(boxes_xyxy[i][3])
                    }
                })
                extracted_data['classes'].append(int(boxes_cls[i]))
        
        if result.keypoints is not None and len(result.keypoints) > 0:
            # 提取关键点数据
            keypoints_data = result.keypoints.data.cpu().numpy()  # [N, num_keypoints, 3]
            
            for i in range(len(result.keypoints)):
                kpt_data = keypoints_data[i]  # [num_keypoints, 3]
                kpt_xy = result.keypoints.xy[i].cpu().numpy()  # [num_keypoints, 2]
                
                extracted_data['keypoints'].append({
                    'index': i,
                    'data': kpt_data.tolist(),  # [x, y, visibility]
                    'xy': kpt_xy.tolist()  # [x, y]
                })
        
        return extracted_data
    
    @staticmethod
    def get_keypoints_for_object(
        result: Results,
        obj_idx: int
    ) -> Optional[np.ndarray]:
        """
        获取指定对象的关键点数据
        
        Args:
            result: YOLO预测结果
            obj_idx: 对象索引
            
        Returns:
            关键点数据 [num_keypoints, 3]，如果不存在返回None
        """
        if result.keypoints is None or obj_idx >= len(result.keypoints):
            return None
        
        keypoints_data = result.keypoints.data.cpu().numpy()
        if obj_idx < len(keypoints_data):
            return keypoints_data[obj_idx]
        
        return None
    
    @staticmethod
    def get_class_for_object(result: Results, obj_idx: int) -> Optional[int]:
        """
        获取指定对象的类别ID
        
        Args:
            result: YOLO预测结果
            obj_idx: 对象索引
            
        Returns:
            类别ID，如果不存在返回None
        """
        if result.boxes is None or obj_idx >= len(result.boxes):
            return None
        
        boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
        if obj_idx < len(boxes_cls):
            return int(boxes_cls[obj_idx])
        
        return None

