# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 最后更新：2025年11月2日
# 主要修改内容：支持多类别关键点检测的Labelme转换脚本（配置文件版本）
# 
# 格式符合Ultralytics YOLO官方格式要求：
# - 支持Dim=2格式: <class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
# - 支持Dim=3格式: <class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> ... <pxn> <pyn> <pn-visibility>
# 参考文档: https://docs.ultralytics.com/zh/datasets/pose/#ultralytics-yolo-format

import os
import json
import cv2
import numpy as np
from pathlib import Path
import yaml

def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def labelme_to_yolo_pose_multi_class(labelme_json_path, output_dir, class_mapping, dim=3):
    """
    将Labelme格式转换为YOLO Pose格式（支持多类别）
    
    Args:
        labelme_json_path: Labelme JSON文件路径
        output_dir: 输出目录
        class_mapping: 类别映射字典 {'label_name': {'class_id': int, 'keypoints': list}}
        dim: 关键点维度 (2 或 3)
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
            
            # 规范化边界框坐标（确保左上角在前，右下角在后）
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            
            # 转换为YOLO格式 (中心点, 宽, 高) - 归一化
            # 格式: <class-index> <x_center> <y_center> <width> <height> <px1> <py1> ...
            # 参考: https://docs.ultralytics.com/zh/datasets/pose/#ultralytics-yolo-format
            center_x = (x_min + x_max) / 2.0 / image_width
            center_y = (y_min + y_max) / 2.0 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            
            # 获取类别信息
            label = shape.get('label', '')
            if label not in class_mapping:
                print(f"警告: 未找到类别 '{label}' 的映射，跳过此对象")
                continue
            
            class_info = class_mapping[label]
            class_id = class_info['class_id']
            expected_keypoints = class_info['keypoints']
            
            # ========== 查找对应的关键点 ==========
            # 原理说明：
            # 1. 遍历配置文件中定义的关键点名称列表
            # 2. 根据配置的关键点名称格式，构建匹配标签：
            #    - 如果关键点名称包含下划线（如 "t1_1"）→ 直接使用该名称匹配
            #    - 如果关键点名称不包含下划线（如 "1"）→ 拼接类别名（如 "tool1_1"）
            # 3. 在JSON的shapes数组中查找：
            #    - shape_type == 'point'（必须是点类型）
            #    - label == 完整标签名（必须完全匹配）
            # 4. 找到则提取坐标，未找到则使用默认值 (0, 0, visibility=0)
            #
            # 支持的两种配置模式：
            # 模式1（拼接模式）：keypoints: ['1', '2', '3'] → 查找 "tool1_1", "tool1_2", "tool1_3"
            # 模式2（直接模式）：keypoints: ['t1_1', 't1_2', 't1_3'] → 直接查找 "t1_1", "t1_2", "t1_3"
            #
            # 注意事项：
            # - 匹配是精确匹配，区分大小写
            # - Labelme标注的关键点label必须与构建的完整标签名完全一致
            keypoints = []
            for kpt_name in expected_keypoints:
                # 构建关键点标签名称（支持两种模式）
                # 模式1：如果关键点名称包含下划线（如 t1_1），直接使用该名称
                # 模式2：如果关键点名称不包含下划线（如 1），拼接类别名（如 tool1_1）
                if '_' in kpt_name:
                    # 直接使用配置中的完整标签名
                    full_kpt_name = kpt_name
                else:
                    # 拼接模式：类别名_关键点名
                    full_kpt_name = f"{label}_{kpt_name}"
                
                # 查找匹配的关键点
                # 遍历所有shapes，查找同时满足以下条件的点：
                # 1. shape_type == 'point' （必须是Labelme的点类型）
                # 2. label == full_kpt_name （标签名必须完全匹配）
                found_kpt = None
                for kpt_shape in data['shapes']:
                    if (kpt_shape['shape_type'] == 'point' and 
                        kpt_shape.get('label', '') == full_kpt_name):
                        found_kpt = kpt_shape
                        break  # 找到第一个匹配的点就停止搜索
                
                if found_kpt:
                    kpt_x, kpt_y = found_kpt['points'][0]
                    # 归一化坐标（关键点坐标归一化到0-1之间）
                    norm_x = kpt_x / image_width
                    norm_y = kpt_y / image_height
                    
                    if dim == 3:
                        # 可见性标志判断（符合Ultralytics YOLO格式）
                        # 0 = 不可见/未标注, 1 = 部分遮挡, 2 = 完全可见
                        # 关键点在边界框内认为是完全可见（2）
                        # 关键点不在边界框内认为是不可见（0）
                        if x_min <= kpt_x <= x_max and y_min <= kpt_y <= y_max:
                            visibility = 2  # 完全可见
                        else:
                            visibility = 0  # 不可见（不在边界框内）
                        keypoints.extend([norm_x, norm_y, visibility])
                    else:
                        # Dim=2格式：只有坐标，无可见性
                        keypoints.extend([norm_x, norm_y])
                else:
                    # 关键点未找到，使用默认值填充（符合官方格式）
                    if dim == 3:
                        keypoints.extend([0.0, 0.0, 0])  # 坐标(0,0) + 可见性=0（未标注）
                    else:
                        keypoints.extend([0.0, 0.0])  # 坐标(0,0)
            
            # 创建检测结果
            # 格式符合Ultralytics YOLO Pose格式:
            # Dim=2: <class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
            # Dim=3: <class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> ... <pxn> <pyn> <pn-visibility>
            # 参考: https://docs.ultralytics.com/zh/datasets/pose/#ultralytics-yolo-format
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
# 最后更新：2025年11月2日
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
    """
    主函数：从配置文件读取参数并执行转换
    """
    import sys
    
    # 获取配置文件路径（默认使用当前目录下的配置文件）
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # 默认配置文件路径
        script_dir = Path(__file__).parent
        config_path = script_dir / 'labelme_to_yolo_pose_multi_class_config.yaml'
        if not config_path.exists():
            print(f"错误: 未找到配置文件 {config_path}")
            print("请提供配置文件路径作为参数，或确保配置文件存在于脚本目录下")
            print("使用方法: python labelme_to_yolo_pose_multi_class_config.py [配置文件路径]")
            sys.exit(1)
    
    # 加载配置文件
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 读取配置参数
    input_dir = config.get('input_dir')
    output_dir = config.get('output_dir')
    dim = config.get('dim', 3)
    create_yaml = config.get('create_yaml', False)
    dataset_name = config.get('dataset_name', 'multi_class_dataset')
    create_labels = config.get('create_labels', False)
    labels_file = config.get('labels_file')
    
    # 读取类别映射
    class_mapping_config = config.get('class_mapping', {})
    
    # 转换为内部格式
    class_mapping = {}
    for label, info in class_mapping_config.items():
        class_mapping[label] = {
            'class_id': info['class_id'],
            'keypoints': info['keypoints']
        }
    
    # 验证必填参数
    if not input_dir or not output_dir:
        print("错误: 配置文件中必须包含 input_dir 和 output_dir")
        sys.exit(1)
    
    print("=" * 60)
    print("多类别关键点检测转换工具（配置文件版本）")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"关键点维度: {dim}")
    print(f"支持的类别: {list(class_mapping.keys())}")
    print(f"创建YAML配置: {create_yaml}")
    print(f"创建标签文件: {create_labels}")
    print("=" * 60)
    
    # 创建标签文件
    if create_labels:
        if not labels_file:
            labels_file = os.path.join(output_dir, 'multi_class_labels.txt')
        create_sample_labels_file(labels_file, class_mapping)
    
    # 批量转换
    batch_convert_multi_class(input_dir, output_dir, class_mapping, dim=dim)
    
    # 创建YAML配置文件
    if create_yaml:
        yaml_file = os.path.join(output_dir, f"{dataset_name}.yaml")
        create_multi_class_dataset_yaml(yaml_file, dataset_name, class_mapping, dim=dim)
    
    print("=" * 60)
    print("所有操作完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

