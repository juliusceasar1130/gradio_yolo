#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原生 YOLO Pose 预测脚本

主要修改内容：
1. 创建原生 YOLO pose 预测脚本，硬编码图片路径
2. CLI 运行后显示检测结果图片
3. 集成关键点角度计算功能，支持角度标注和结果输出
4. 弧度绘制功能已禁用，待日后开发（仅显示角度文本和关键点连线）
修改时间：2025年1月28日
最后更新：2025年1月31日（禁用弧度绘制功能）
时间更新：2025年11月2日
"""

from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
from datetime import datetime
import sys


def convert_to_json_serializable(obj):
    """
    将对象转换为JSON可序列化的格式
    处理numpy类型、bool类型等
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

# 添加工具目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'tools'))
from angle_calculator import (
    load_angle_config,
    load_tool_pose_config,
    calculate_angles_for_object,
    annotate_angles_on_image
)

# ========== 硬编码配置 ==========
# 模型路径（绝对地址）
MODEL_PATH = r"D:\00deeplearn\yolo11\gradio_web_yolo\tool_pose\tools\train2\weights\best.pt"

# 输入图片路径（绝对地址）
IMAGE_PATH = r"C:\Users\julius\Desktop\张春亮\zcl\GG5087612_01_20251028152549437.jpg"

# 配置文件路径（相对于脚本目录）
SCRIPT_DIR = Path(__file__).parent
ANGLE_CONFIG_PATH = SCRIPT_DIR / "tools" / "angle_config.yaml"
TOOL_POSE_CONFIG_PATH = SCRIPT_DIR / "tools" / "tool_pose.yaml"

# 输出目录
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ========== 预测参数 ==========
# 置信度阈值
CONF = 0.25
# IoU阈值
IOU = 0.45
# 图像尺寸
IMGSZ = 640
# 设备（"cuda" 或 "cpu"，空字符串表示自动选择）
DEVICE = ""
# 半精度推理（True 加速，False 精度更高）
HALF = False
# 最大检测数量
MAX_DET = 1000


def main():
    """主函数：执行 YOLO Pose 预测并显示结果"""
    
    # 使用硬编码的绝对路径，转换为 Path 对象
    model_path = Path(MODEL_PATH)
    image_path = Path(IMAGE_PATH)
    
    # 检查文件是否存在
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not image_path.exists():
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    print("=" * 60)
    print("YOLO Pose 预测 + 角度计算")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"图片路径: {image_path}")
    print(f"预测参数: conf={CONF}, iou={IOU}, imgsz={IMGSZ}")
    print("=" * 60)
    
    # 加载角度配置和工具姿态配置
    print("\n📋 加载配置文件...")
    try:
        angle_config = load_angle_config(ANGLE_CONFIG_PATH)
        tool_pose_config = load_tool_pose_config(TOOL_POSE_CONFIG_PATH)
        print("✅ 配置文件加载完成")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return
    
    # 获取类别名称映射和关键点名称映射
    class_names = tool_pose_config.get('names', {})
    kpt_names_dict = tool_pose_config.get('kpt_names', {})
    
    # 加载模型
    print("\n📦 加载模型...")
    model = YOLO(str(model_path))
    print("✅ 模型加载完成")
    
    # 执行预测
    print("\n🔍 执行预测...")
    results = model.predict(
        source=str(image_path),
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        half=HALF,
        max_det=MAX_DET,
        verbose=False
    )
    
    if not results or len(results) == 0:
        print("❌ 未检测到任何对象")
        return
    
    result = results[0]
    print("✅ 预测完成")
    
    # 打印检测结果
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
    
    # ========== 角度计算 ==========
    print("\n📐 计算角度...")
    all_angles_results = []
    
    if result.boxes is not None and result.keypoints is not None:
        # 获取关键点数据（原图坐标）
        keypoints_data = result.keypoints.data.cpu().numpy()  # [N, num_keypoints, 3]
        boxes_cls = result.boxes.cls.cpu().numpy().astype(int)  # [N]
        
        for obj_idx in range(len(result.boxes)):
            cls_id = boxes_cls[obj_idx]
            class_name = class_names.get(cls_id, f"class_{cls_id}")
            
            # 获取该对象的关键点数据
            if obj_idx < len(keypoints_data):
                kpt_data = keypoints_data[obj_idx]  # [num_keypoints, 3]
                
                # 获取该类别的关键点名称列表
                kpt_names = kpt_names_dict.get(cls_id, [])
                
                if not kpt_names:
                    print(f"   ⚠️  对象 {obj_idx+1} ({class_name}): 未找到关键点名称配置")
                    continue
                
                # 计算角度
                angles_result = calculate_angles_for_object(
                    class_name=class_name,
                    kpt_data=kpt_data,
                    kpt_names=kpt_names,
                    angle_config=angle_config
                )
                
                # 保存结果
                obj_angles_info = {
                    'object_id': obj_idx,
                    'class_id': int(cls_id),
                    'class_name': class_name,
                    'angles': angles_result
                }
                all_angles_results.append(obj_angles_info)
                
                # 打印角度结果
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
    
    # 获取标注后的图片
    print("\n🎨 生成可视化结果...")
    annotated_image = result.plot()
    
    if annotated_image is None:
        print("❌ 可视化失败")
        return
    
    # ========== 在图像上标注角度 ==========
    if all_angles_results and annotated_image is not None:
        print("\n📝 在图像上标注角度...")
        
        keypoints_data = result.keypoints.data.cpu().numpy()
        boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
        
        for obj_idx, obj_angles_info in enumerate(all_angles_results):
            if obj_idx >= len(keypoints_data):
                continue
            
            cls_id = obj_angles_info['class_id']
            class_name = obj_angles_info['class_name']
            angles_result = obj_angles_info['angles']
            kpt_data = keypoints_data[obj_idx]
            kpt_names = kpt_names_dict.get(cls_id, [])
            
            # 创建关键点名称到索引的映射（用于标注函数）
            kpt_indices_map = {name: idx for idx, name in enumerate(kpt_names)}
            
            # 获取关键点坐标（xy格式，只有坐标）
            kpt_xy = result.keypoints.xy[obj_idx].cpu().numpy()  # [num_keypoints, 2]
            
            # 标注角度
            annotated_image = annotate_angles_on_image(
                image=annotated_image,
                angles_result=angles_result,
                keypoints_xy=kpt_xy,
                keypoint_indices_map=kpt_indices_map
            )
        
        print("✅ 角度标注完成")
    
    print("✅ 可视化完成")
    
    # ========== 保存角度结果到文件 ==========
    if all_angles_results:
        print("\n💾 保存角度结果...")
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json_path = OUTPUT_DIR / f"angle_results_{timestamp}.json"
        
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
                for obj_angles_info in all_angles_results:
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
                            'visible': bool(vis > 0.5)  # 转换为Python原生bool类型
                        })
                    
                    obj_info['keypoints'] = keypoints
                
                output_data['predictions'].append(obj_info)
        
        # 转换为JSON可序列化格式
        output_data_serializable = convert_to_json_serializable(output_data)
        
        # 保存JSON文件
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 角度结果已保存: {output_json_path}")
    
    # 显示图片
    print("\n🖼️  显示检测结果...")
    
    # 将 BGR 转换为 RGB（YOLO 返回的是 BGR 格式）
    if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    else:
        annotated_image_rgb = annotated_image
    
    # 使用 matplotlib 显示
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image_rgb)
    plt.axis('off')
    plt.title('YOLO Pose 检测结果（含角度标注）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("✅ 图片显示完成")
    print("\n" + "=" * 60)
    print("预测完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

