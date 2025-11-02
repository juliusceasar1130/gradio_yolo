#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
原生 YOLO Pose 预测脚本（模块化重构版）

主要修改内容：
1. 模块化重构，职责分离
2. 配置管理独立
3. 核心功能解耦
4. 便于扩展和维护
5. 简化结构：移除冗余的processing封装，直接使用utils中的角度计算函数
6. 将打印和保存函数迁移至 utils 模块，提高代码复用性
修改时间：2025年1月28日
重构时间：2025年1月31日
简化时间：2025年1月31日
utils迁移时间：2025年11月2日
最后更新：2025年11月2日
"""

import sys
from pathlib import Path

# 添加当前目录到 Python 路径，确保可以导入本地模块
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# 导入配置模块
from config import Config, get_default_config, get_config_dict

# 导入核心模块
from core import ModelLoader, Predictor, DataExtractor

# 导入工具模块（包含角度计算函数）
from utils import (
    display_image,
    validate_paths,
    load_angle_config,
    calculate_angles_for_object,
    annotate_angles_on_image,
    print_detection_results,
    print_angle_results,
    save_results_to_json
)


def main():
    """主函数：执行 YOLO Pose 预测并显示结果（模块化版本）"""
    
    # ========== 1. 加载配置 ==========
    config = get_default_config()
    
    # 验证配置
    is_valid, error_msg = config.validate()
    if not is_valid:
        print(error_msg)
        return
    
    # 验证路径
    path_valid, path_error = validate_paths(
        (config.model_path, "模型文件"),
        (config.image_path, "图片文件")
    )
    if not path_valid:
        print(path_error)
        return
    
    # 打印配置信息
    print("=" * 60)
    print("YOLO Pose 预测 + 角度计算（模块化版本）")
    print("=" * 60)
    print(f"模型路径: {config.model_path}")
    print(f"图片路径: {config.image_path}")
    print(f"预测参数: conf={config.conf}, iou={config.iou}, imgsz={config.imgsz}")
    print("=" * 60)
    
    # ========== 2. 加载配置文件 ==========
    print("\n📋 加载配置文件...")
    try:
        angle_config = load_angle_config(config.config_path)
        class_names = angle_config.get('names', {})
        kpt_names_dict = angle_config.get('kpt_names', {})
        
        # 提取角度定义（从 angles 键下获取）
        angles_config = angle_config.get('angles', {})
        if not angles_config:
            # 旧格式兼容：直接从根级别获取 tool1, tool2
            angles_config = {
                k: v for k, v in angle_config.items() 
                if k not in ['names', 'kpt_names', 'paths', 'predict', 'output']
            }
        print("✅ 配置文件加载完成")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return
    
    # ========== 3. 加载模型 ==========
    print("\n📦 加载模型...")
    try:
        model = ModelLoader.get_model(config.model_path, config.device)
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # ========== 4. 执行预测 ==========
    print("\n🔍 执行预测...")
    try:
        results = Predictor.predict(model, config.image_path, config)
        result = results[0]
        print("✅ 预测完成")
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return
    
    # ========== 5. 打印检测结果 ==========
    print_detection_results(result, class_names)
    
    # ========== 6. 计算角度 ==========
    try:
        angles_results = []
        
        if result.boxes is not None and result.keypoints is not None:
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
                    
                    if kpt_names:
                        # 计算角度
                        angles_result = calculate_angles_for_object(
                            class_name=class_name,
                            kpt_data=kpt_data,
                            kpt_names=kpt_names,
                            angle_config=angles_config
                        )
                        
                        # 保存结果
                        obj_angles_info = {
                            'object_id': obj_idx,
                            'class_id': int(cls_id),
                            'class_name': class_name,
                            'angles': angles_result
                        }
                        angles_results.append(obj_angles_info)
        
        print_angle_results(angles_results)
    except Exception as e:
        print(f"❌ 角度计算失败: {e}")
        return
    
    # ========== 7. 标注图像 ==========
    print("\n🎨 生成可视化结果...")
    try:
        # 获取YOLO默认标注后的图像
        annotated_image = result.plot()
        
        if annotated_image is None:
            raise ValueError("无法生成标注图像")
        
        # 在图像上标注角度
        if angles_results:
            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
            
            for obj_idx, obj_angles_info in enumerate(angles_results):
                if obj_idx >= len(keypoints_data):
                    continue
                
                cls_id = obj_angles_info['class_id']
                angles_result = obj_angles_info['angles']
                kpt_names = kpt_names_dict.get(cls_id, [])
                
                if kpt_names:
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
        
        print("✅ 可视化完成")
    except Exception as e:
        print(f"❌ 图像标注失败: {e}")
        return
    
    # ========== 8. 保存结果 ==========
    if angles_results and config.save_json:
        save_results_to_json(
            result,
            angles_results,
            Path(config.image_path),
            config.output_dir,
            class_names,
            kpt_names_dict
        )
    
    # ========== 9. 显示结果 ==========
    if config.show_image:
        print("\n🖼️  显示检测结果...")
        display_image(
            annotated_image,
            title='YOLO Pose 检测结果（含角度标注）',
            show=True
        )
        print("✅ 图片显示完成")
    
    print("\n" + "=" * 60)
    print("预测完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
