#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试关键点坐标偏移问题

用于诊断关节点位置偏移的原因
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
    from PIL import Image
    import numpy as np
    import cv2
    
    def test_keypoint_coordinates(model_path, image_path):
        """
        测试关键点坐标是否正确
        
        Args:
            model_path: 模型路径
            image_path: 测试图像路径
        """
        print("=" * 60)
        print("关键点坐标偏移测试")
        print("=" * 60)
        
        # 加载模型
        print(f"\n1. 加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 加载图像
        print(f"2. 加载图像: {image_path}")
        image = Image.open(image_path)
        image_cv = cv2.imread(image_path)
        
        original_size = image.size  # (width, height)
        original_shape = image_cv.shape[:2]  # (height, width)
        
        print(f"   原始图像尺寸 (PIL): {original_size}")
        print(f"   原始图像尺寸 (CV2): {original_shape}")
        
        # 测试不同的imgsz
        test_imgsz_values = [320, 640, 1280]
        
        for imgsz in test_imgsz_values:
            print(f"\n{'='*60}")
            print(f"测试 imgsz = {imgsz}")
            print(f"{'='*60}")
            
            # 执行推理
            print(f"3. 执行推理 (imgsz={imgsz})...")
            results = model.predict(image_path, imgsz=imgsz, conf=0.25, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                # 获取关键点
                if result.keypoints is not None and len(result.keypoints) > 0:
                    print(f"   检测到 {len(result.keypoints)} 个对象")
                    
                    for obj_idx, kpts in enumerate(result.keypoints):
                        print(f"\n   对象 {obj_idx + 1} 的关键点:")
                        kpt_data = kpts.data.cpu().numpy()
                        
                        if len(kpt_data.shape) == 3:
                            kpt_data = kpt_data[0]  # 去除batch维度
                        
                        # 检查坐标是否在图像范围内
                        out_of_range_count = 0
                        visible_count = 0
                        
                        for kpt_idx, (x, y, conf) in enumerate(kpt_data):
                            is_visible = conf > 0.5
                            
                            if is_visible:
                                visible_count += 1
                                # 检查坐标是否在图像范围内
                                in_range = (0 <= x <= original_size[0] and 
                                           0 <= y <= original_size[1])
                                
                                if not in_range:
                                    out_of_range_count += 1
                                    print(f"     ⚠️ 关键点 {kpt_idx+1}: ({x:.2f}, {y:.2f}) "
                                          f"超出图像范围 [0-{original_size[0]}, 0-{original_size[1]}]")
                                
                                print(f"     关键点 {kpt_idx+1}: ({x:.2f}, {y:.2f}), "
                                      f"置信度: {conf:.3f}, "
                                      f"在范围内: {in_range}")
                        
                        print(f"\n   可见关键点: {visible_count}/{len(kpt_data)}")
                        if out_of_range_count > 0:
                            print(f"   ⚠️ 超出范围的关键点: {out_of_range_count}")
                else:
                    print("   未检测到关键点")
                
                # 测试plot()可视化
                print(f"\n4. 测试plot()可视化...")
                try:
                    annotated = result.plot()
                    if annotated is not None:
                        print(f"   ✅ plot()成功，输出尺寸: {annotated.shape}")
                    else:
                        print("   ❌ plot()返回None")
                except Exception as e:
                    print(f"   ❌ plot()失败: {e}")
            else:
                print("   未检测到任何对象")
        
        print(f"\n{'='*60}")
        print("测试完成")
        print("=" * 60)
        
        # 建议
        print("\n📝 建议:")
        print("1. 检查训练时的imgsz配置")
        print("2. 确保推理时的imgsz与训练时一致")
        print("3. 如果关键点坐标超出图像范围，说明坐标映射有问题")
        print("4. 尝试使用与训练时相同的imgsz重新推理")
    
    if __name__ == '__main__':
        import argparse
        
        parser = argparse.ArgumentParser(description='测试关键点坐标偏移')
        parser.add_argument('--model', type=str, required=True,
                          help='模型路径')
        parser.add_argument('--image', type=str, required=True,
                          help='测试图像路径')
        
        args = parser.parse_args()
        
        test_keypoint_coordinates(args.model, args.image)
        
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装 ultralytics: pip install ultralytics")
    sys.exit(1)
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

