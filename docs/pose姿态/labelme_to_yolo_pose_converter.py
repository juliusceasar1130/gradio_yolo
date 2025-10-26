# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：Labelme格式转YOLO Pose格式转换脚本

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def labelme_to_yolo_pose(labelme_json_path, output_dir, class_id=0, dim=3, keypoint_names=None):
    """
    将Labelme格式转换为YOLO Pose格式
    
    Args:
        labelme_json_path: Labelme JSON文件路径
        output_dir: 输出目录
        class_id: 类别ID
        dim: 关键点维度 (2 或 3)
        keypoint_names: 关键点名称列表
    """
    
    # 默认关键点名称（四边形）
    if keypoint_names is None:
        keypoint_names = [
            "top_left",      # 左上角
            "top_right",     # 右上角
            "bottom_right",  # 右下角
            "bottom_left"    # 左下角
        ]
    
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个标注对象
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            # 处理边界框
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 转换为YOLO格式 (中心点, 宽, 高) - 归一化
            bbox_center_x = (x1 + x2) / 2 / image_width
            bbox_center_y = (y1 + y2) / 2 / image_height
            bbox_width = abs(x2 - x1) / image_width
            bbox_height = abs(y2 - y1) / image_height
            
            # 查找对应的关键点
            keypoints = []
            for i, keypoint_name in enumerate(keypoint_names):
                keypoint_found = False
                for kp_shape in data['shapes']:
                    if (kp_shape['shape_type'] == 'point' and 
                        kp_shape['label'] == keypoint_name):
                        kp_x, kp_y = kp_shape['points'][0]
                        # 转换为相对坐标
                        kp_x_norm = kp_x / image_width
                        kp_y_norm = kp_y / image_height
                        
                        if dim == 2:
                            # Dim=2格式：只有x,y坐标
                            keypoints.extend([kp_x_norm, kp_y_norm])
                        elif dim == 3:
                            # Dim=3格式：x,y坐标 + 可见性
                            keypoints.extend([kp_x_norm, kp_y_norm, 2])  # 2表示完全可见
                        
                        keypoint_found = True
                        break
                
                if not keypoint_found:
                    if dim == 2:
                        keypoints.extend([0, 0])  # 不可见的关键点
                    elif dim == 3:
                        keypoints.extend([0, 0, 0])  # 不可见的关键点
            
            # 生成YOLO格式标签
            yolo_line = [str(class_id)]
            yolo_line.extend([str(bbox_center_x), str(bbox_center_y), 
                            str(bbox_width), str(bbox_height)])
            yolo_line.extend([str(kp) for kp in keypoints])
            
            # 保存标签文件
            image_name = Path(data['imagePath']).stem
            label_file = os.path.join(output_dir, f"{image_name}.txt")
            
            with open(label_file, 'w') as f:
                f.write(' '.join(yolo_line) + '\n')

def labelme_to_yolo_pose_with_visibility(labelme_json_path, output_dir, class_id=0, base_keypoint_names=None):
    """
    将Labelme格式转换为YOLO Pose格式，支持visibility标注
    
    Args:
        labelme_json_path: Labelme JSON文件路径
        output_dir: 输出目录
        class_id: 类别ID
        base_keypoint_names: 基础关键点名称列表
    """
    
    # 默认基础关键点名称
    if base_keypoint_names is None:
        base_keypoint_names = [
            "top_left",
            "top_right", 
            "bottom_right",
            "bottom_left"
        ]
    
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个标注对象
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            # 处理边界框
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 转换为YOLO格式 (中心点, 宽, 高) - 归一化
            bbox_center_x = (x1 + x2) / 2 / image_width
            bbox_center_y = (y1 + y2) / 2 / image_height
            bbox_width = abs(x2 - x1) / image_width
            bbox_height = abs(y2 - y1) / image_height
            
            # 查找对应的关键点
            keypoints = []
            for i, base_name in enumerate(base_keypoint_names):
                keypoint_found = False
                
                # 查找匹配的关键点标注
                for kp_shape in data['shapes']:
                    if kp_shape['shape_type'] == 'point':
                        label = kp_shape['label']
                        
                        # 检查是否匹配基础名称
                        if label.startswith(base_name):
                            kp_x, kp_y = kp_shape['points'][0]
                            # 转换为相对坐标
                            kp_x_norm = kp_x / image_width
                            kp_y_norm = kp_y / image_height
                            
                            # 根据标签名称确定visibility
                            visibility = determine_visibility(label)
                            
                            # Dim=3格式：x,y坐标 + 可见性
                            keypoints.extend([kp_x_norm, kp_y_norm, visibility])
                            keypoint_found = True
                            break
                
                if not keypoint_found:
                    # 未找到关键点，设置为不可见
                    keypoints.extend([0, 0, 0])
            
            # 生成YOLO格式标签
            yolo_line = [str(class_id)]
            yolo_line.extend([str(bbox_center_x), str(bbox_center_y), 
                            str(bbox_width), str(bbox_height)])
            yolo_line.extend([str(kp) for kp in keypoints])
            
            # 保存标签文件
            image_name = Path(data['imagePath']).stem
            label_file = os.path.join(output_dir, f"{image_name}.txt")
            
            with open(label_file, 'w') as f:
                f.write(' '.join(yolo_line) + '\n')

def determine_visibility(label):
    """
    根据标签名称确定visibility值
    """
    if 'visible' in label.lower():
        return 2  # 完全可见
    elif 'occluded' in label.lower():
        return 1  # 部分可见/被遮挡
    elif 'invisible' in label.lower():
        return 0  # 不可见
    else:
        return 2  # 默认完全可见

def batch_convert(input_dir, output_dir, dim=3, use_visibility=False, keypoint_names=None):
    """
    批量转换Labelme标注文件
    
    Args:
        input_dir: 输入目录（包含JSON文件）
        output_dir: 输出目录
        dim: 关键点维度 (2 或 3)
        use_visibility: 是否使用可见性标注
        keypoint_names: 关键点名称列表
    """
    json_files = list(Path(input_dir).glob('*.json'))
    
    if not json_files:
        print(f"在 {input_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for json_file in json_files:
        print(f"转换文件: {json_file}")
        try:
            if use_visibility:
                labelme_to_yolo_pose_with_visibility(str(json_file), output_dir, keypoint_names=keypoint_names)
            else:
                labelme_to_yolo_pose(str(json_file), output_dir, dim=dim, keypoint_names=keypoint_names)
        except Exception as e:
            print(f"转换文件 {json_file} 时出错: {e}")
    
    print("批量转换完成！")

def create_dataset_yaml(output_path, dataset_name, keypoint_names, kpt_shape, skeleton=None):
    """
    创建数据集YAML配置文件
    
    Args:
        output_path: 输出路径
        dataset_name: 数据集名称
        keypoint_names: 关键点名称列表
        kpt_shape: 关键点形状 [num_keypoints, dim]
        skeleton: 骨架连接列表
    """
    
    if skeleton is None:
        # 默认四边形骨架连接
        skeleton = [
            [0, 1],  # 上边
            [1, 2],  # 右边
            [2, 3],  # 下边
            [3, 0]   # 左边
        ]
    
    yaml_content = f"""# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：{dataset_name}数据集配置文件

# 数据集路径
path: ./{dataset_name}
train: images/train
val: images/val

# 关键点配置
kpt_shape: {kpt_shape}  # {kpt_shape[0]}个关键点，{kpt_shape[1]}个维度

# 类别配置
nc: 1
names:
  0: {dataset_name}

# 关键点名称
kpt_names:
  0:
"""
    
    for i, name in enumerate(keypoint_names):
        yaml_content += f"    - {name}\n"
    
    yaml_content += "\n# 骨架连接\nskeleton:\n"
    for connection in skeleton:
        yaml_content += f"  - {connection}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"数据集配置文件已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Labelme到YOLO Pose格式转换工具')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录（包含JSON文件）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--dim', type=int, choices=[2, 3], default=3, help='关键点维度 (2或3)')
    parser.add_argument('--use_visibility', action='store_true', help='使用可见性标注')
    parser.add_argument('--keypoints', type=str, nargs='+', help='关键点名称列表')
    parser.add_argument('--create_yaml', action='store_true', help='创建数据集YAML文件')
    parser.add_argument('--dataset_name', type=str, default='custom_dataset', help='数据集名称')
    
    args = parser.parse_args()
    
    # 设置默认关键点名称
    if args.keypoints is None:
        if args.use_visibility:
            args.keypoints = ["top_left", "top_right", "bottom_right", "bottom_left"]
        else:
            args.keypoints = ["top_left", "top_right", "bottom_right", "bottom_left"]
    
    # 执行转换
    batch_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dim=args.dim,
        use_visibility=args.use_visibility,
        keypoint_names=args.keypoints
    )
    
    # 创建YAML配置文件
    if args.create_yaml:
        kpt_shape = [len(args.keypoints), args.dim]
        yaml_path = os.path.join(args.output_dir, f"{args.dataset_name}.yaml")
        create_dataset_yaml(yaml_path, args.dataset_name, args.keypoints, kpt_shape)

if __name__ == "__main__":
    main()
