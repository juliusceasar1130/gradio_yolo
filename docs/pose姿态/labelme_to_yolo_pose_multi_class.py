# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：支持多类别关键点检测的Labelme转换脚本

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def labelme_to_yolo_pose_multi_class(labelme_json_path, output_dir, class_mapping, dim=3):
    """
    将Labelme格式转换为YOLO Pose格式（支持多类别）
    
    Args:
        labelme_json_path: Labelme JSON文件路径
        output_dir: 输出目录
        class_mapping: 类别映射字典 {'label_name': {'class_id': int, 'keypoints': list}}
        dim: 关键点维度 (2 或 3)
    
    Example:
        class_mapping = {
            'triangle': {
                'class_id': 0,
                'keypoints': ['top', 'bottom_left', 'bottom_right']
            },
            'quadrilateral': {
                'class_id': 1, 
                'keypoints': ['top_left', 'top_right', 'bottom_right', 'bottom_left']
            }
        }
    """
    
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储所有检测结果
    detections = []
    
    # 处理每个标注对象
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            # 处理边界框
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 转换为YOLO格式 (中心点, 宽, 高) - 归一化
            center_x = (x1 + x2) / 2.0 / image_width
            center_y = (y1 + y2) / 2.0 / image_height
            width = abs(x2 - x1) / image_width
            height = abs(y2 - y1) / image_height
            
            # 获取类别信息
            label = shape.get('label', '')
            if label not in class_mapping:
                print(f"警告: 未找到类别 '{label}' 的映射，跳过此对象")
                continue
            
            class_info = class_mapping[label]
            class_id = class_info['class_id']
            expected_keypoints = class_info['keypoints']
            
            # 查找对应的关键点
            keypoints = []
            for kpt_name in expected_keypoints:
                # 查找匹配的关键点
                found_kpt = None
                for kpt_shape in data['shapes']:
                    if (kpt_shape['shape_type'] == 'point' and 
                        kpt_shape.get('label', '') == kpt_name):
                        found_kpt = kpt_shape
                        break
                
                if found_kpt:
                    kpt_x, kpt_y = found_kpt['points'][0]
                    # 归一化坐标
                    norm_x = kpt_x / image_width
                    norm_y = kpt_y / image_height
                    
                    if dim == 3:
                        # 检查可见性（基于关键点是否在边界框内）
                        visibility = 2 if (x1 <= kpt_x <= x2 and y1 <= kpt_y <= y2) else 1
                        keypoints.extend([norm_x, norm_y, visibility])
                    else:
                        keypoints.extend([norm_x, norm_y])
                else:
                    # 关键点未找到，使用默认值
                    if dim == 3:
                        keypoints.extend([0.0, 0.0, 0])  # 不可见
                    else:
                        keypoints.extend([0.0, 0.0])
            
            # 创建检测结果
            detection = [class_id, center_x, center_y, width, height] + keypoints
            detections.append(detection)
    
    # 保存结果
    if detections:
        output_file = os.path.join(output_dir, Path(labelme_json_path).stem + '.txt')
        with open(output_file, 'w') as f:
            for detection in detections:
                line = ' '.join(map(str, detection)) + '\n'
                f.write(line)
        
        print(f"转换完成: {labelme_json_path} -> {output_file}")
        print(f"检测到 {len(detections)} 个对象")
    else:
        print(f"未检测到任何对象: {labelme_json_path}")

def batch_convert_multi_class(input_dir, output_dir, class_mapping, dim=3):
    """
    批量转换多类别Labelme格式到YOLO Pose格式
    
    Args:
        input_dir: 输入目录（包含JSON文件）
        output_dir: 输出目录
        class_mapping: 类别映射字典
        dim: 关键点维度 (2 或 3)
    """
    json_files = list(Path(input_dir).glob('*.json'))
    
    if not json_files:
        print(f"在 {input_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    print(f"类别映射: {class_mapping}")
    
    for json_file in json_files:
        print(f"转换文件: {json_file}")
        try:
            labelme_to_yolo_pose_multi_class(str(json_file), output_dir, class_mapping, dim=dim)
        except Exception as e:
            print(f"转换文件 {json_file} 时出错: {e}")
    
    print("批量转换完成！")

def create_multi_class_dataset_yaml(output_path, dataset_name, class_mapping, dim=3):
    """
    创建多类别数据集YAML配置文件
    
    Args:
        output_path: 输出路径
        dataset_name: 数据集名称
        class_mapping: 类别映射字典
        dim: 关键点维度
    """
    
    # 计算最大关键点数量
    max_keypoints = max(len(info['keypoints']) for info in class_mapping.values())
    
    yaml_content = f"""# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：{dataset_name}多类别数据集配置文件

# 数据集路径
path: ./{dataset_name}
train: images/train
val: images/val

# 关键点配置
kpt_shape: [{max_keypoints}, {dim}]  # 最大{max_keypoints}个关键点，{dim}个维度

# 类别配置
nc: {len(class_mapping)}
names:
"""
    
    # 添加类别名称
    for label, info in class_mapping.items():
        yaml_content += f"  {info['class_id']}: {label}\n"
    
    yaml_content += "\n# 关键点名称\nkpt_names:\n"
    
    # 为每个类别添加关键点名称
    for label, info in class_mapping.items():
        yaml_content += f"  {info['class_id']}:\n"
        for kpt_name in info['keypoints']:
            yaml_content += f"    - {kpt_name}\n"
    
    # 添加骨架连接（这里使用默认的，实际使用时需要根据具体形状调整）
    yaml_content += "\n# 骨架连接（需要根据具体形状调整）\nskeleton:\n"
    
    # 为每个类别添加骨架连接
    for label, info in class_mapping.items():
        yaml_content += f"  # {label}的骨架连接\n"
        if label == 'triangle':
            # 三角形骨架
            skeleton = [[0, 1], [1, 2], [2, 0]]
        elif label == 'quadrilateral':
            # 四边形骨架
            skeleton = [[0, 1], [1, 2], [2, 3], [3, 0]]
        else:
            # 默认骨架（根据关键点数量）
            skeleton = [[i, (i+1) % len(info['keypoints'])] for i in range(len(info['keypoints']))]
        
        for connection in skeleton:
            yaml_content += f"  - {connection}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"多类别数据集配置文件已保存到: {output_path}")

def create_sample_labels_file(output_path, class_mapping):
    """
    创建多类别示例标签文件
    
    Args:
        output_path: 输出路径
        class_mapping: 类别映射字典
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 多类别关键点标签文件\n")
        f.write("# 格式: 类别名_关键点名\n\n")
        
        for label, info in class_mapping.items():
            f.write(f"# {label} 的关键点\n")
            for kpt_name in info['keypoints']:
                f.write(f"{label}_{kpt_name}\n")
            f.write("\n")
    
    print(f"多类别标签文件已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='多类别Labelme到YOLO Pose格式转换工具')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录（包含JSON文件）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--dim', type=int, choices=[2, 3], default=3, help='关键点维度 (2或3)')
    parser.add_argument('--create_yaml', action='store_true', help='创建数据集YAML文件')
    parser.add_argument('--dataset_name', type=str, default='multi_class_dataset', help='数据集名称')
    parser.add_argument('--create_labels', action='store_true', help='创建示例标签文件')
    parser.add_argument('--labels_file', type=str, help='标签文件输出路径')
    
    args = parser.parse_args()
    
    # 定义三角形和四边形的类别映射
    class_mapping = {
        'triangle': {
            'class_id': 0,
            'keypoints': ['top', 'bottom_left', 'bottom_right']
        },
        'quadrilateral': {
            'class_id': 1,
            'keypoints': ['top_left', 'top_right', 'bottom_right', 'bottom_left']
        }
    }
    
    print("多类别关键点检测转换工具")
    print(f"支持的类别: {list(class_mapping.keys())}")
    print(f"关键点维度: {args.dim}")
    
    # 创建标签文件
    if args.create_labels:
        labels_file = args.labels_file or os.path.join(args.output_dir, 'multi_class_labels.txt')
        create_sample_labels_file(labels_file, class_mapping)
    
    # 批量转换
    batch_convert_multi_class(args.input_dir, args.output_dir, class_mapping, dim=args.dim)
    
    # 创建YAML配置文件
    if args.create_yaml:
        yaml_file = os.path.join(args.output_dir, f"{args.dataset_name}.yaml")
        create_multi_class_dataset_yaml(yaml_file, args.dataset_name, class_mapping, dim=args.dim)

if __name__ == "__main__":
    main()
