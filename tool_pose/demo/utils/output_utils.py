#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
输出工具模块

主要修改内容：
1. 从 pose_predict.py 迁移打印函数到 utils 模块
2. 提供检测结果和角度结果的格式化输出功能
修改时间：2025年11月2日
最后更新：2025年11月2日
"""

from typing import Dict, List, Any


def print_detection_results(result, class_names: Dict[int, str]):
    """
    打印检测结果
    
    Args:
        result: YOLO预测结果对象
        class_names: 类别ID到类别名称的映射字典
    """
    print("\n📊 检测结果:")
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"   检测到 {len(result.boxes)} 个对象")
        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names.get(cls, f"class_{cls}")
            print(f"   对象 {i+1}: 类别={class_name} (ID:{cls}), 置信度={conf:.3f}")
    
    if result.keypoints is not None and len(result.keypoints) > 0:
        print(f"   检测到 {len(result.keypoints)} 组关键点")
        for i, kpts in enumerate(result.keypoints):
            kpt_data = kpts.data.cpu().numpy()
            if len(kpt_data.shape) == 3:
                kpt_data = kpt_data[0]
            visible_count = sum(1 for _, _, conf in kpt_data if conf > 0.5)
            print(f"   对象 {i+1}: {visible_count}/{len(kpt_data)} 个关键点可见")


def print_angle_results(angles_results: List[Dict[str, Any]]):
    """
    打印角度计算结果
    
    Args:
        angles_results: 角度计算结果列表，每个元素包含：
            - object_id: 对象索引
            - class_name: 类别名称
            - angles: 角度计算结果字典
    """
    print("\n📐 计算角度...")
    
    for obj_info in angles_results:
        obj_idx = obj_info['object_id']
        class_name = obj_info['class_name']
        angles_result = obj_info['angles']
        
        print(f"\n   📏 对象 {obj_idx+1} ({class_name}) 的角度:")
        valid_count = 0
        for angle_name, angle_info in angles_result.items():
            if angle_info.get('valid', False):
                angle_value = angle_info.get('value')
                if angle_value is not None:
                    print(f"      {angle_name}: {angle_value:.2f}° ✓")
                    valid_count += 1
            else:
                reason = angle_info.get('reason', '未知原因')
                print(f"      {angle_name}: 无法计算 ✗ ({reason})")
        
        if valid_count == 0:
            print(f"      ⚠️  所有角度都无法计算")
        else:
            print(f"      ✅ 成功计算 {valid_count}/{len(angles_result)} 个角度")
    
    print("\n✅ 角度计算完成")

