# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：多类别关键点检测推理示例

#!/usr/bin/env python3
"""
多类别关键点检测推理示例
演示如何在代码运行时区分不同类别
"""

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

class MultiClassPoseDetector:
    """多类别关键点检测器"""
    
    def __init__(self, model_path, class_names=None):
        """
        初始化检测器
        
        Args:
            model_path: 训练好的模型路径
            class_names: 类别名称列表，如 ['triangle', 'quadrilateral']
        """
        self.model = YOLO(model_path)
        self.class_names = class_names or ['triangle', 'quadrilateral']
        
        # 定义每个类别的关键点名称
        self.keypoint_names = {
            0: ['top', 'bottom_left', 'bottom_right'],  # 三角形
            1: ['top_left', 'top_right', 'bottom_right', 'bottom_left']  # 四边形
        }
        
        # 定义骨架连接
        self.skeleton_connections = {
            0: [[0, 1], [1, 2], [2, 0]],  # 三角形骨架
            1: [[0, 1], [1, 2], [2, 3], [3, 0]]  # 四边形骨架
        }
    
    def detect_and_classify(self, image_path, conf_threshold=0.25):
        """
        检测并分类图像中的对象
        
        Args:
            image_path: 图像路径
            conf_threshold: 置信度阈值
            
        Returns:
            dict: 包含检测结果的字典
        """
        # 运行检测
        results = self.model.predict(image_path, conf=conf_threshold)
        
        # 解析结果
        detections = {
            'triangles': [],
            'quadrilaterals': [],
            'image': None
        }
        
        for result in results:
            image = result.orig_img
            detections['image'] = image.copy()
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                keypoints = result.keypoints
                
                # 获取类别ID和置信度
                class_ids = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                box_coords = boxes.xyxy.cpu().numpy()
                
                # 处理每个检测结果
                for i, class_id in enumerate(class_ids):
                    class_id = int(class_id)
                    confidence = confidences[i]
                    box = box_coords[i]
                    
                    # 获取关键点
                    if keypoints is not None and len(keypoints.data) > i:
                        kpts = keypoints.data[i].cpu().numpy()
                        
                        # 根据类别ID区分处理
                        if class_id == 0:  # 三角形
                            triangle_data = self._process_triangle(
                                box, kpts, confidence, image.shape
                            )
                            detections['triangles'].append(triangle_data)
                            
                        elif class_id == 1:  # 四边形
                            quad_data = self._process_quadrilateral(
                                box, kpts, confidence, image.shape
                            )
                            detections['quadrilaterals'].append(quad_data)
        
        return detections
    
    def _process_triangle(self, box, keypoints, confidence, image_shape):
        """处理三角形检测结果"""
        height, width = image_shape[:2]
        
        # 提取3个关键点
        kpts = keypoints[:3]  # 只取前3个关键点
        
        # 转换坐标到图像尺寸
        kpts_pixels = []
        for kpt in kpts:
            x, y, v = kpt
            if v > 0:  # 关键点可见
                pixel_x = int(x * width)
                pixel_y = int(y * height)
                kpts_pixels.append((pixel_x, pixel_y))
            else:
                kpts_pixels.append(None)
        
        return {
            'class_name': 'triangle',
            'class_id': 0,
            'confidence': confidence,
            'bbox': box,
            'keypoints': kpts_pixels,
            'keypoint_names': self.keypoint_names[0],
            'skeleton': self.skeleton_connections[0]
        }
    
    def _process_quadrilateral(self, box, keypoints, confidence, image_shape):
        """处理四边形检测结果"""
        height, width = image_shape[:2]
        
        # 提取4个关键点
        kpts = keypoints[:4]  # 只取前4个关键点
        
        # 转换坐标到图像尺寸
        kpts_pixels = []
        for kpt in kpts:
            x, y, v = kpt
            if v > 0:  # 关键点可见
                pixel_x = int(x * width)
                pixel_y = int(y * height)
                kpts_pixels.append((pixel_x, pixel_y))
            else:
                kpts_pixels.append(None)
        
        return {
            'class_name': 'quadrilateral',
            'class_id': 1,
            'confidence': confidence,
            'bbox': box,
            'keypoints': kpts_pixels,
            'keypoint_names': self.keypoint_names[1],
            'skeleton': self.skeleton_connections[1]
        }
    
    def visualize_results(self, detections, save_path=None):
        """
        可视化检测结果
        
        Args:
            detections: 检测结果字典
            save_path: 保存路径（可选）
        """
        image = detections['image'].copy()
        
        # 定义颜色
        colors = {
            'triangle': (0, 255, 0),      # 绿色
            'quadrilateral': (255, 0, 0)  # 红色
        }
        
        # 绘制三角形
        for triangle in detections['triangles']:
            self._draw_object(image, triangle, colors['triangle'])
        
        # 绘制四边形
        for quad in detections['quadrilaterals']:
            self._draw_object(image, quad, colors['quadrilateral'])
        
        # 显示结果
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"检测结果: {len(detections['triangles'])}个三角形, {len(detections['quadrilaterals'])}个四边形")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        
        plt.show()
    
    def _draw_object(self, image, obj_data, color):
        """绘制单个对象"""
        # 绘制边界框
        x1, y1, x2, y2 = map(int, obj_data['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制类别标签
        label = f"{obj_data['class_name']}: {obj_data['confidence']:.2f}"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 绘制关键点
        for i, kpt in enumerate(obj_data['keypoints']):
            if kpt is not None:
                x, y = kpt
                cv2.circle(image, (x, y), 5, color, -1)
                cv2.putText(image, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制骨架连接
        for connection in obj_data['skeleton']:
            start_idx, end_idx = connection
            if (start_idx < len(obj_data['keypoints']) and 
                end_idx < len(obj_data['keypoints']) and
                obj_data['keypoints'][start_idx] is not None and
                obj_data['keypoints'][end_idx] is not None):
                
                start_point = obj_data['keypoints'][start_idx]
                end_point = obj_data['keypoints'][end_idx]
                cv2.line(image, start_point, end_point, color, 2)

def main():
    """主函数 - 演示多类别检测"""
    
    # 初始化检测器
    model_path = "path/to/your/trained/model.pt"  # 替换为您的模型路径
    detector = MultiClassPoseDetector(model_path)
    
    # 测试图像路径
    test_image = "path/to/test/image.jpg"  # 替换为您的测试图像
    
    print("=== 多类别关键点检测演示 ===")
    print(f"模型路径: {model_path}")
    print(f"测试图像: {test_image}")
    print()
    
    # 运行检测
    print("正在检测...")
    detections = detector.detect_and_classify(test_image, conf_threshold=0.25)
    
    # 打印结果
    print("检测结果:")
    print(f"  三角形数量: {len(detections['triangles'])}")
    print(f"  四边形数量: {len(detections['quadrilaterals'])}")
    print()
    
    # 详细结果
    for i, triangle in enumerate(detections['triangles']):
        print(f"三角形 {i+1}:")
        print(f"  置信度: {triangle['confidence']:.3f}")
        print(f"  关键点: {triangle['keypoint_names']}")
        print(f"  坐标: {triangle['keypoints']}")
        print()
    
    for i, quad in enumerate(detections['quadrilaterals']):
        print(f"四边形 {i+1}:")
        print(f"  置信度: {quad['confidence']:.3f}")
        print(f"  关键点: {quad['keypoint_names']}")
        print(f"  坐标: {quad['keypoints']}")
        print()
    
    # 可视化结果
    detector.visualize_results(detections, save_path="multi_class_result.jpg")

if __name__ == "__main__":
    main()
