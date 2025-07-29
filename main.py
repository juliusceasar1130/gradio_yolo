# 创建者/修改者: chenliang；修改时间：2025年7月27日 23:20；主要修改内容：创建项目主入口文件

"""
YOLO检测工具主入口文件

提供命令行和Web界面两种使用方式
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from yolo_detector import (
    Config, ModelLoader, ObjectDetector, SegmentationDetector,
    BatchProcessor, create_gradio_interface
)
from yolo_detector.utils import setup_logging, setup_error_handling, get_logger


def setup_environment():
    """设置环境"""
    try:
        # 加载配置
        config = Config()
        
        # 设置日志和错误处理
        setup_logging(config)
        setup_error_handling()
        
        logger = get_logger(__name__)
        logger.info("环境设置完成")
        
        return config
        
    except Exception as e:
        print(f"环境设置失败: {e}")
        sys.exit(1)


def run_web_interface(config: Config, share: bool = False, debug: bool = False):
    """运行Web界面"""
    try:
        logger = get_logger(__name__)
        logger.info("启动Web界面...")
        
        # 创建Gradio界面
        demo = create_gradio_interface(config)
        
        # 启动界面
        demo.launch(
            share=share,
            debug=debug,
            server_name="127.0.0.1",
            server_port=7861,  # 使用不同的端口
            show_error=True
        )
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"启动Web界面失败: {e}")
        print(f"启动Web界面失败: {e}")
        sys.exit(1)


def run_batch_processing(config: Config, input_folder: str, output_folder: str = None,
                        detector_type: str = "detection", confidence: float = 0.25):
    """运行批量处理"""
    try:
        logger = get_logger(__name__)
        logger.info(f"开始批量处理: {input_folder}")
        
        # 初始化组件
        model_loader = ModelLoader(config)
        
        if detector_type == "detection":
            detector = ObjectDetector(model_loader, config)
        else:
            detector = SegmentationDetector(model_loader, config)
        
        batch_processor = BatchProcessor(detector, config)
        
        # 定义进度回调
        def progress_callback(current, total, message):
            if total > 0:
                progress = (current / total) * 100
                print(f"进度: {progress:.1f}% - {message}")
            else:
                print(f"状态: {message}")
        
        # 执行批量处理
        result = batch_processor.process_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            progress_callback=progress_callback,
            conf=confidence
        )
        
        if result['success']:
            print(f"\n✅ 批量处理完成!")
            print(f"总图像数: {result['total_images']}")
            print(f"成功检测: {result['successful_detections']}")
            print(f"检测对象总数: {result['total_objects']}")
            print(f"处理时间: {result['processing_time']:.2f}秒")
            print(f"输出目录: {result['output_folder']}")
            
            # 显示类别统计
            if result['class_summary']:
                print("\n类别统计:")
                for class_name, info in result['class_summary'].items():
                    print(f"- {class_name}: {info['count']}个 (出现在{info['images']}张图像中)")
        else:
            print(f"❌ 批量处理失败: {result.get('error', '未知错误')}")
            sys.exit(1)
            
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"批量处理失败: {e}")
        print(f"批量处理失败: {e}")
        sys.exit(1)


def run_single_detection(config: Config, image_path: str, output_path: str = None,
                        detector_type: str = "detection", confidence: float = 0.25):
    """运行单张图像检测"""
    try:
        logger = get_logger(__name__)
        logger.info(f"检测单张图像: {image_path}")
        
        # 验证输入文件
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            sys.exit(1)
        
        # 初始化组件
        model_loader = ModelLoader(config)
        
        if detector_type == "detection":
            detector = ObjectDetector(model_loader, config)
        else:
            detector = SegmentationDetector(model_loader, config)
        
        # 执行检测
        result = detector.detect(image_path, conf=confidence)
        
        if result is None:
            print("❌ 检测失败")
            sys.exit(1)
        
        # 显示结果
        stats = result.get_statistics()
        print(f"✅ 检测完成!")
        print(f"检测对象数: {stats['total_detections']}")
        
        if stats['classes']:
            print("类别统计:")
            for class_name, class_info in stats['classes'].items():
                avg_conf = sum(class_info['confidences']) / len(class_info['confidences'])
                print(f"- {class_name}: {class_info['count']}个 (平均置信度: {avg_conf:.2f})")
        
        # 保存结果图像
        if output_path:
            vis_image = result.get_visualization()
            if vis_image is not None:
                from PIL import Image
                Image.fromarray(vis_image).save(output_path)
                print(f"结果已保存到: {output_path}")
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"单张图像检测失败: {e}")
        print(f"单张图像检测失败: {e}")
        sys.exit(1)


def show_system_info(config: Config):
    """显示系统信息"""
    try:
        print("=== YOLO检测工具系统信息 ===")
        
        # 项目信息
        print(f"项目根目录: {project_root}")
        print(f"配置文件: {config.config_path}")
        
        # 模型信息
        model_loader = ModelLoader(config)
        available_models = model_loader.list_available_models()
        
        print("\n模型状态:")
        for model_type, model_info in available_models.items():
            status = "✅ 可用" if model_info['exists'] else "❌ 不可用"
            print(f"- {model_type.title()}模型: {status}")
            print(f"  路径: {model_info['path']}")
        
        # 数据路径
        data_config = config.get_data_config()
        print(f"\n数据路径:")
        print(f"- 输入文件夹: {data_config.get('input_folder', 'N/A')}")
        print(f"- 输出文件夹: {data_config.get('output_folder', 'N/A')}")
        
        # 检测参数
        detection_config = config.get_detection_config()
        print(f"\n检测参数:")
        print(f"- 置信度阈值: {detection_config.get('confidence_threshold', 'N/A')}")
        print(f"- 最大检测数: {detection_config.get('max_detections', 'N/A')}")
        
    except Exception as e:
        print(f"获取系统信息失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="YOLO检测工具 - 支持目标检测和图像分割",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 启动Web界面
  python main.py web
  
  # 批量处理文件夹
  python main.py batch --input /path/to/images --output /path/to/results
  
  # 检测单张图像
  python main.py detect --image /path/to/image.jpg --output result.jpg
  
  # 显示系统信息
  python main.py info
        """
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # Web界面命令
    web_parser = subparsers.add_parser('web', help='启动Web界面')
    web_parser.add_argument('--share', action='store_true', help='创建公共链接')
    web_parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    # 批量处理命令
    batch_parser = subparsers.add_parser('batch', help='批量处理图像')
    batch_parser.add_argument('--input', '-i', required=True, help='输入文件夹路径')
    batch_parser.add_argument('--output', '-o', help='输出文件夹路径')
    batch_parser.add_argument('--type', choices=['detection', 'segmentation'], 
                             default='detection', help='检测类型')
    batch_parser.add_argument('--confidence', '-c', type=float, default=0.25,
                             help='置信度阈值 (0.1-0.9)')
    
    # 单张检测命令
    detect_parser = subparsers.add_parser('detect', help='检测单张图像')
    detect_parser.add_argument('--image', '-i', required=True, help='图像文件路径')
    detect_parser.add_argument('--output', '-o', help='输出图像路径')
    detect_parser.add_argument('--type', choices=['detection', 'segmentation'],
                              default='detection', help='检测类型')
    detect_parser.add_argument('--confidence', '-c', type=float, default=0.25,
                              help='置信度阈值 (0.1-0.9)')
    
    # 系统信息命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 设置环境
    config = setup_environment()
    
    # 执行命令
    if args.command == 'web':
        run_web_interface(config, args.share, args.debug)
    elif args.command == 'batch':
        run_batch_processing(config, args.input, args.output, args.type, args.confidence)
    elif args.command == 'detect':
        run_single_detection(config, args.image, args.output, args.type, args.confidence)
    elif args.command == 'info':
        show_system_info(config)


if __name__ == "__main__":
    main()
