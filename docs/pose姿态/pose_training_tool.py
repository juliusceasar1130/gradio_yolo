# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：自定义Pose模型训练和验证脚本

from ultralytics import YOLO
import torch
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

def train_custom_pose_model(data_yaml, epochs=100, imgsz=640, batch=16, device=None, 
                          project='pose_training', name='custom_model', model_size='s'):
    """
    训练自定义Pose模型
    
    Args:
        data_yaml: 数据集YAML配置文件路径
        epochs: 训练轮数
        imgsz: 图像尺寸
        batch: 批次大小
        device: 设备 ('cuda', 'cpu', 或 None自动选择)
        project: 项目目录
        name: 实验名称
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
    """
    
    # 检查GPU可用性
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    model_path = f'yolo11{model_size}-pose.pt'
    print(f"加载预训练模型: {model_path}")
    model = YOLO(model_path)
    
    # 训练参数
    print(f"开始训练，数据集配置: {data_yaml}")
    results = model.train(
        data=data_yaml,                # 数据集配置文件
        epochs=epochs,                 # 训练轮数
        imgsz=imgsz,                   # 图像尺寸
        batch=batch,                   # 批次大小
        device=device,                 # 设备
        workers=4,                     # 数据加载线程数
        project=project,               # 项目目录
        name=name,                     # 实验名称
        save=True,                     # 保存检查点
        save_period=10,                # 每10个epoch保存一次
        val=True,                      # 验证
        plots=True,                    # 生成训练图表
        verbose=True                   # 详细输出
    )
    
    print("自定义Pose模型训练完成！")
    return results

def validate_custom_pose_model(model_path, test_image_path, conf=0.25, imgsz=640, save_results=True):
    """
    验证训练好的自定义Pose模型
    
    Args:
        model_path: 训练好的模型路径
        test_image_path: 测试图像路径
        conf: 置信度阈值
        imgsz: 图像尺寸
        save_results: 是否保存结果
    """
    
    # 加载训练好的模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 预测
    print(f"对图像进行预测: {test_image_path}")
    results = model.predict(
        source=test_image_path,
        conf=conf,
        imgsz=imgsz,
        save=save_results,
        save_txt=save_results
    )
    
    # 显示结果
    for i, result in enumerate(results):
        print(f"\n=== 图像 {i+1} 预测结果 ===")
        
        if result.keypoints is not None and len(result.keypoints) > 0:
            print(f"检测到 {len(result.keypoints)} 个对象")
            
            for obj_idx, kpts in enumerate(result.keypoints):
                print(f"\n对象 {obj_idx+1} 的关键点:")
                visible_points = 0
                
                for kpt_idx, kpt in enumerate(kpts):
                    if kpt[2] > 0:  # 如果关键点可见
                        print(f"  关键点 {kpt_idx+1}: ({kpt[0]:.2f}, {kpt[1]:.2f}) - 可见性: {kpt[2]}")
                        visible_points += 1
                    else:
                        print(f"  关键点 {kpt_idx+1}: 不可见")
                
                print(f"  可见关键点数量: {visible_points}/{len(kpts)}")
        else:
            print("未检测到任何对象")
        
        # 保存结果图像
        if save_results:
            annotated_image = result.plot(
                show_boxes=True,
                show_labels=True,
                show_conf=True,
                line_width=2
            )
            
            output_path = f'pose_result_{i+1}.jpg'
            cv2.imwrite(output_path, annotated_image)
            print(f"结果已保存为: {output_path}")

def batch_validate(model_path, test_dir, conf=0.25, imgsz=640, save_results=True):
    """
    批量验证模型
    
    Args:
        model_path: 训练好的模型路径
        test_dir: 测试图像目录
        conf: 置信度阈值
        imgsz: 图像尺寸
        save_results: 是否保存结果
    """
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(test_dir).glob(f'*{ext}'))
        image_files.extend(Path(test_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"在 {test_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 批量预测
    results = model.predict(
        source=str(test_dir),
        conf=conf,
        imgsz=imgsz,
        save=save_results,
        save_txt=save_results
    )
    
    print(f"批量验证完成，处理了 {len(results)} 个图像")

def create_sample_dataset_yaml(output_path, dataset_name, keypoint_names, kpt_shape, skeleton=None):
    """
    创建示例数据集YAML配置文件
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
    
    for name in keypoint_names:
        yaml_content += f"    - {name}\n"
    
    yaml_content += "\n# 骨架连接\nskeleton:\n"
    for connection in skeleton:
        yaml_content += f"  - {connection}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"示例数据集配置文件已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Pose模型训练和验证工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--data', type=str, required=True, help='数据集YAML配置文件路径')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    train_parser.add_argument('--batch', type=int, default=16, help='批次大小')
    train_parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='设备')
    train_parser.add_argument('--project', type=str, default='pose_training', help='项目目录')
    train_parser.add_argument('--name', type=str, default='custom_model', help='实验名称')
    train_parser.add_argument('--model_size', type=str, choices=['n', 's', 'm', 'l', 'x'], default='s', help='模型大小')
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='验证模型')
    validate_parser.add_argument('--model', type=str, required=True, help='模型路径')
    validate_parser.add_argument('--image', type=str, help='测试图像路径')
    validate_parser.add_argument('--dir', type=str, help='测试图像目录')
    validate_parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    validate_parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    validate_parser.add_argument('--no_save', action='store_true', help='不保存结果')
    
    # 创建示例配置命令
    config_parser = subparsers.add_parser('create_config', help='创建示例数据集配置')
    config_parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    config_parser.add_argument('--dataset_name', type=str, default='custom_dataset', help='数据集名称')
    config_parser.add_argument('--keypoints', type=str, nargs='+', help='关键点名称列表')
    config_parser.add_argument('--kpt_shape', type=int, nargs=2, default=[4, 3], help='关键点形状 [num_keypoints, dim]')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_custom_pose_model(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            model_size=args.model_size
        )
    
    elif args.command == 'validate':
        if args.image:
            validate_custom_pose_model(
                model_path=args.model,
                test_image_path=args.image,
                conf=args.conf,
                imgsz=args.imgsz,
                save_results=not args.no_save
            )
        elif args.dir:
            batch_validate(
                model_path=args.model,
                test_dir=args.dir,
                conf=args.conf,
                imgsz=args.imgsz,
                save_results=not args.no_save
            )
        else:
            print("请指定 --image 或 --dir 参数")
    
    elif args.command == 'create_config':
        if args.keypoints is None:
            args.keypoints = ["top_left", "top_right", "bottom_right", "bottom_left"]
        
        create_sample_dataset_yaml(
            output_path=args.output,
            dataset_name=args.dataset_name,
            keypoint_names=args.keypoints,
            kpt_shape=args.kpt_shape
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
