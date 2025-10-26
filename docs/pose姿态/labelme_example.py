# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：Labelme数据标注使用示例脚本

#!/usr/bin/env python3
"""
Labelme数据标注使用示例脚本

本脚本演示如何使用Labelme进行数据标注的完整流程：
1. 创建示例标签文件
2. 转换Labelme格式到YOLO格式
3. 创建数据集配置
4. 训练模型
5. 验证模型

使用方法:
    python labelme_example.py --help
"""

import os
import argparse
from pathlib import Path
import subprocess
import sys

def create_sample_labels_file(output_path, keypoint_names):
    """
    创建示例标签文件
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for name in keypoint_names:
            f.write(f"{name}\n")
    
    print(f"示例标签文件已创建: {output_path}")

def create_sample_dataset_structure(base_dir):
    """
    创建示例数据集目录结构
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

def run_conversion(input_dir, output_dir, keypoint_names, dim=3, use_visibility=False):
    """
    运行格式转换
    """
    cmd = [
        sys.executable, "labelme_to_yolo_pose_converter.py",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--dim", str(dim),
        "--create_yaml",
        "--dataset_name", "sample_dataset"
    ]
    
    if use_visibility:
        cmd.append("--use_visibility")
    
    if keypoint_names:
        cmd.extend(["--keypoints"] + keypoint_names)
    
    print(f"运行转换命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("转换成功!")
        print(result.stdout)
    else:
        print("转换失败!")
        print(result.stderr)
        return False
    
    return True

def run_training(data_yaml, epochs=50, batch=8, model_size='s'):
    """
    运行模型训练
    """
    cmd = [
        sys.executable, "pose_training_tool.py", "train",
        "--data", data_yaml,
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--model_size", model_size,
        "--project", "example_training",
        "--name", "sample_model"
    ]
    
    print(f"运行训练命令: {' '.join(cmd)}")
    print("注意: 训练可能需要较长时间，请耐心等待...")
    
    # 这里只是演示，实际训练需要真实数据
    print("训练命令已准备就绪，请确保有足够的训练数据后再运行")

def run_validation(model_path, test_image=None, test_dir=None):
    """
    运行模型验证
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

def main():
    parser = argparse.ArgumentParser(description='Labelme数据标注使用示例')
    parser.add_argument('--mode', type=str, choices=['setup', 'convert', 'train', 'validate'], 
                       default='setup', help='运行模式')
    parser.add_argument('--dataset_dir', type=str, default='sample_dataset', 
                       help='数据集目录')
    parser.add_argument('--keypoints', type=str, nargs='+', 
                       default=['top_left', 'top_right', 'bottom_right', 'bottom_left'],
                       help='关键点名称列表')
    parser.add_argument('--dim', type=int, choices=[2, 3], default=3, 
                       help='关键点维度')
    parser.add_argument('--use_visibility', action='store_true', 
                       help='使用可见性标注')
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
    
    print("=== Labelme数据标注使用示例 ===")
    print(f"运行模式: {args.mode}")
    print(f"数据集目录: {args.dataset_dir}")
    print(f"关键点: {args.keypoints}")
    print(f"维度: {args.dim}")
    print()
    
    if args.mode == 'setup':
        print("1. 创建示例数据集结构...")
        create_sample_dataset_structure(args.dataset_dir)
        
        print("\n2. 创建示例标签文件...")
        labels_file = os.path.join(args.dataset_dir, "labels.txt")
        create_sample_labels_file(labels_file, args.keypoints)
        
        print("\n3. 创建带可见性的标签文件...")
        if args.use_visibility:
            visibility_labels = []
            for name in args.keypoints:
                visibility_labels.extend([
                    f"{name}_visible",
                    f"{name}_occluded", 
                    f"{name}_invisible"
                ])
            visibility_file = os.path.join(args.dataset_dir, "labels_visibility.txt")
            create_sample_labels_file(visibility_file, visibility_labels)
        
        print("\n设置完成!")
        print(f"请将您的图像文件放入: {args.dataset_dir}/images/train/")
        print(f"请将Labelme JSON文件放入: {args.dataset_dir}/annotations/")
        print("然后运行: python labelme_example.py --mode convert")
    
    elif args.mode == 'convert':
        print("运行格式转换...")
        input_dir = os.path.join(args.dataset_dir, "annotations")
        output_dir = os.path.join(args.dataset_dir, "labels")
        
        if not os.path.exists(input_dir):
            print(f"错误: 输入目录不存在 {input_dir}")
            print("请先运行 --mode setup 创建目录结构")
            return
        
        success = run_conversion(
            input_dir=input_dir,
            output_dir=output_dir,
            keypoint_names=args.keypoints,
            dim=args.dim,
            use_visibility=args.use_visibility
        )
        
        if success:
            print("\n转换完成!")
            print("现在可以运行: python labelme_example.py --mode train")
    
    elif args.mode == 'train':
        print("准备模型训练...")
        data_yaml = os.path.join(args.dataset_dir, "sample_dataset.yaml")
        
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
