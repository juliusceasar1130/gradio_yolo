# 创建者/修改者: chenliang
# 修改时间：2025年11月02日 10:00
# 主要修改内容：
# 1. 从 tool_pose/demo/utils/json_utils.py 移植
# 2. JSON序列化工具函数
# 3. 数据类型转换
# 4. 结果保存功能

"""
JSON工具模块

用于工具关键点检测的JSON序列化和结果保存
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


def convert_to_json_serializable(obj: Any) -> Any:
    """
    将对象转换为JSON可序列化的格式
    处理numpy类型、bool类型等
    
    Args:
        obj: 待转换的对象
        
    Returns:
        JSON可序列化的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def save_results_to_json(
    result,
    angles_results: List[Dict[str, Any]],
    image_path: Path,
    output_dir: Path,
    class_names: Dict[int, str],
    kpt_names_dict: Dict[int, List[str]]
) -> Path:
    """
    保存检测结果和角度计算结果到JSON文件
    
    Args:
        result: YOLO预测结果对象
        angles_results: 角度计算结果列表
        image_path: 图片路径
        output_dir: 输出目录
        class_names: 类别ID到类别名称的映射
        kpt_names_dict: 类别ID到关键点名称列表的映射
        
    Returns:
        保存的JSON文件路径
    """
    logger.info("保存角度结果...")
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = output_dir / f"angle_results_{timestamp}.json"
    
    # 构建完整的结果数据
    output_data = {
        'image_path': str(image_path),
        'image_shape': result.orig_shape,
        'timestamp': timestamp,
        'predictions': []
    }
    
    # 添加每个对象的检测和角度信息
    if result.boxes is not None:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()
        boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
        keypoints_data = result.keypoints.data.cpu().numpy()
        
        for obj_idx in range(len(result.boxes)):
            cls_id = boxes_cls[obj_idx]
            class_name = class_names.get(cls_id, f"class_{cls_id}")
            
            # 找到对应的角度结果
            angles_result = {}
            for obj_angles_info in angles_results:
                if obj_angles_info['object_id'] == obj_idx:
                    angles_result = obj_angles_info['angles']
                    break
            
            # 构建对象信息
            obj_info = {
                'object_id': obj_idx,
                'class_id': int(cls_id),
                'class_name': class_name,
                'confidence': float(boxes_conf[obj_idx]),
                'bbox': {
                    'x1': float(boxes_xyxy[obj_idx][0]),
                    'y1': float(boxes_xyxy[obj_idx][1]),
                    'x2': float(boxes_xyxy[obj_idx][2]),
                    'y2': float(boxes_xyxy[obj_idx][3])
                },
                'angles': angles_result
            }
            
            # 添加关键点信息
            if obj_idx < len(keypoints_data):
                kpt_data = keypoints_data[obj_idx]
                kpt_names = kpt_names_dict.get(cls_id, [])
                keypoints = []
                
                for kpt_idx, (x, y, vis) in enumerate(kpt_data):
                    kpt_name = kpt_names[kpt_idx] if kpt_idx < len(kpt_names) else f"kpt_{kpt_idx}"
                    keypoints.append({
                        'index': int(kpt_idx),
                        'name': kpt_name,
                        'x': float(x),
                        'y': float(y),
                        'visibility': float(vis),
                        'visible': bool(vis > 0.5)
                    })
                
                obj_info['keypoints'] = keypoints
            
            output_data['predictions'].append(obj_info)
    
    # 转换为JSON可序列化格式
    output_data_serializable = convert_to_json_serializable(output_data)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data_serializable, f, indent=2, ensure_ascii=False)
    
    logger.info(f"角度结果已保存: {output_json_path}")
    return output_json_path

