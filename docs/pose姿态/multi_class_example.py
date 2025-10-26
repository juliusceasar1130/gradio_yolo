# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：多类别关键点检测使用示例脚本

#!/usr/bin/env python3
"""
多类别关键点检测使用示例脚本

本脚本演示如何在同一个数据集中标注和训练多种形状的关键点检测模型：
1. 创建多类别标签文件
2. 设置数据集目录结构
3. 转换Labelme格式到YOLO格式
4. 创建多类别数据集配置
5. 训练和验证模型

使用方法:
    python multi_class_example.py --help
"""

import os
import argparse
from pathlib import Path
import subprocess
import sys

def create_multi_class_labels_file(output_path):
    """
    创建多类别标签文件
    """
    labels_content = """# 多类别关键点标签文件
# 格式: 类别名_关键点名

# triangle 的关键点
triangle_top
triangle_bottom_left
triangle_bottom_right

# quadrilateral 的关键点
quadrilateral_top_left
quadrilateral_top_right
quadrilateral_bottom_right
quadrilateral_bottom_left
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(labels_content)
    
    print(f"多类别标签文件已创建: {output_path}")

def create_dataset_structure(base_dir):
    """
    创建多类别数据集目录结构
    """
    directories = [
        f"{base_dir}/images/train",
        f"{base_dir}/images/val", 
        f"{base_dir}/labels/train",
        f"{base_dir}/labels/val",
        f"{base_dir}/annotations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    print(f"数据集目录结构已创建: {base_dir}")

def create_sample_yaml_config(output_path, dataset_name):
    """
    创建示例多类别YAML配置文件
    """
    yaml_content = f"""# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：{dataset_name}多类别数据集配置文件

# 数据集路径
path: ./{dataset_name}
train: images/train
val: images/val

# 关键点配置
kpt_shape: [4, 3]  # 最大4个关键点，3个维度

# 类别配置
nc: 2
names:
  0: triangle
  1: quadrilateral

# 关键点名称
kpt_names:
  0:  # triangle (3个关键点)
    - top
    - bottom_left
    - bottom_right
  1:  # quadrilateral (4个关键点)
    - top_left
    - top_right
    - bottom_right
    - bottom_left

# 骨架连接
skeleton:
  # triangle的骨架连接
  - [0, 1]  # top -> bottom_left
  - [1, 2]  # bottom_left -> bottom_right
  - [2, 0]  # bottom_right -> top
  # quadrilateral的骨架连接
  - [0, 1]  # top_left -> top_right
  - [1, 2]  # top_right -> bottom_right
  - [2, 3]  # bottom_right -> bottom_left
  - [3, 0]  # bottom_left -> top_left
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"多类别YAML配置文件已创建: {output_path}")

def run_conversion(input_dir, output_dir, dim=3):
    """
    运行多类别格式转换
    """
    cmd = [
        sys.executable, "labelme_to_yolo_pose_multi_class.py",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--dim", str(dim),
        "--create_yaml",
        "--create_labels",
        "--dataset_name", "triangle_quad_dataset"
    ]
    
    print(f"运行转换命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("多类别转换成功!")
        print(result.stdout)
    else:
        print("转换失败!")
        print(result.stderr)
        return False
    
    return True

def run_training(data_yaml, epochs=50, batch=8, model_size='s'):
    """
    运行多类别模型训练
    """
    cmd = [
        sys.executable, "pose_training_tool.py", "train",
        "--data", data_yaml,
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--model_size", model_size,
        "--project", "multi_class_training",
        "--name", "triangle_quad_model"
    ]
    
    print(f"运行训练命令: {' '.join(cmd)}")
    print("注意: 多类别训练可能需要更长时间，请耐心等待...")
    
    # 这里只是演示，实际训练需要真实数据
    print("训练命令已准备就绪，请确保有足够的训练数据后再运行")

def run_validation(model_path, test_image=None, test_dir=None):
    """
    运行多类别模型验证
    """
    cmd = [
        sys.executable, "pose_training_tool.py", "validate",
        "--model", model_path,
        "--conf", "0.25"
    ]
    
    if test_image:
        cmd.extend(["--image", test_image])
    elif test_dir:
        cmd.extend(["--dir", test_dir])
    else:
        print("请指定测试图像或测试目录")
        return False
    
    print(f"运行验证命令: {' '.join(cmd)}")
    print("验证命令已准备就绪")

def print_labeling_instructions():
    """
    打印标注说明
    """
    instructions = """
=== 多类别关键点标注说明 ===

1. 启动Labelme:
   labelme

2. 加载标签文件:
   - 点击 "Tools" → "Load Labels"
   - 选择 multi_class_labels.txt

3. 标注三角形:
   - 使用 "Create Rectangle" 框选三角形
   - Label字段输入: triangle
   - 标注3个关键点:
     * triangle_top (顶点)
     * triangle_bottom_left (左下角)
     * triangle_bottom_right (右下角)

4. 标注四边形:
   - 使用 "Create Rectangle" 框选四边形
   - Label字段输入: quadrilateral
   - 标注4个关键点:
     * quadrilateral_top_left (左上角)
     * quadrilateral_top_right (右上角)
     * quadrilateral_bottom_right (右下角)
     * quadrilateral_bottom_left (左下角)

5. 保存标注:
   - 每个图像保存为对应的JSON文件
   - 将JSON文件放入 annotations/ 目录

注意事项:
- 确保关键点标注顺序一致
- 边界框要完全包含目标对象
- 关键点要精确标注在正确位置
"""
    print(instructions)

def main():
    parser = argparse.ArgumentParser(description='多类别关键点检测使用示例')
    parser.add_argument('--mode', type=str, 
                       choices=['setup', 'instructions', 'convert', 'train', 'validate'], 
                       default='setup', help='运行模式')
    parser.add_argument('--dataset_dir', type=str, default='triangle_quad_dataset', 
                       help='数据集目录')
    parser.add_argument('--dim', type=int, choices=[2, 3], default=3, 
                       help='关键点维度')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, 
                       help='批次大小')
    parser.add_argument('--model_size', type=str, choices=['n', 's', 'm', 'l', 'x'], 
                       default='s', help='模型大小')
    parser.add_argument('--model_path', type=str, 
                       help='模型路径（用于验证）')
    parser.add_argument('--test_image', type=str, 
                       help='测试图像路径')
    parser.add_argument('--test_dir', type=str, 
                       help='测试图像目录')
    
    args = parser.parse_args()
    
    print("=== 多类别关键点检测使用示例 ===")
    print(f"运行模式: {args.mode}")
    print(f"数据集目录: {args.dataset_dir}")
    print(f"关键点维度: {args.dim}")
    print()
    
    if args.mode == 'setup':
        print("1. 创建多类别数据集结构...")
        create_dataset_structure(args.dataset_dir)
        
        print("\n2. 创建多类别标签文件...")
        labels_file = os.path.join(args.dataset_dir, "multi_class_labels.txt")
        create_sample_labels_file(labels_file)
        
        print("\n3. 创建示例YAML配置文件...")
        yaml_file = os.path.join(args.dataset_dir, "triangle_quad_dataset.yaml")
        create_sample_yaml_config(yaml_file, args.dataset_dir)
        
        print("\n设置完成!")
        print(f"请将您的图像文件放入: {args.dataset_dir}/images/train/")
        print(f"请将Labelme JSON文件放入: {args.dataset_dir}/annotations/")
        print("然后运行: python multi_class_example.py --mode convert")
    
    elif args.mode == 'instructions':
        print_labeling_instructions()
    
    elif args.mode == 'convert':
        print("运行多类别格式转换...")
        input_dir = os.path.join(args.dataset_dir, "annotations")
        output_dir = os.path.join(args.dataset_dir, "labels")
        
        if not os.path.exists(input_dir):
            print(f"错误: 输入目录不存在 {input_dir}")
            print("请先运行 --mode setup 创建目录结构")
            return
        
        success = run_conversion(
            input_dir=input_dir,
            output_dir=output_dir,
            dim=args.dim
        )
        
        if success:
            print("\n转换完成!")
            print("现在可以运行: python multi_class_example.py --mode train")
    
    elif args.mode == 'train':
        print("准备多类别模型训练...")
        data_yaml = os.path.join(args.dataset_dir, "triangle_quad_dataset.yaml")
        
        if not os.path.exists(data_yaml):
            print(f"错误: 数据集配置文件不存在 {data_yaml}")
            print("请先运行 --mode convert 创建配置文件")
            return
        
        run_training(
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch=args.batch,
            model_size=args.model_size
        )
    
    elif args.mode == 'validate':
        if not args.model_path:
            print("错误: 请指定模型路径 --model_path")
            return
        
        if not os.path.exists(args.model_path):
            print(f"错误: 模型文件不存在 {args.model_path}")
            return
        
        run_validation(
            model_path=args.model_path,
            test_image=args.test_image,
            test_dir=args.test_dir
        )
    
    print("\n=== 示例完成 ===")

if __name__ == "__main__":
    main()
