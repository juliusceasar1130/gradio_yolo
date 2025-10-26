# 创建者/修改者: chenliang；修改时间：2025年7月27日 23:20；主要修改内容：创建项目主入口文件

"""
YOLO检测工具主入口文件

提供命令行和Web界面两种使用方式
"""

import argparse
import sys
import os
# 该语句的含义和作用：
# 导入pathlib库中的Path类，用于进行跨平台的文件路径操作。
# Path类可以方便地处理文件和文件夹路径，支持多种操作系统，提升代码的可移植性和可读性。
from pathlib import Path

# 添加src目录到Python路径
# 该语句的含义和作用：
# 获取当前文件（main.py）所在目录的路径，并赋值给变量project_root。
# 这样可以方便后续基于项目根目录进行路径拼接和资源定位，保证路径的可移植性和正确性。
# __file__ 代表当前脚本文件（即 main.py）的文件路径
project_root = Path(__file__).parent
# 下面这行代码的作用是将项目根目录下的 "src" 文件夹路径添加到 Python 的模块搜索路径（sys.path）的最前面。
# 这样做的目的是：当你在后续代码中通过 import 语句导入自定义模块（如 yolo_detector）时，Python 解释器会优先在 "src" 目录下查找对应的模块文件。
# 详细参数说明：
# - sys.path：这是一个列表，包含了所有 Python 解释器查找模块时会遍历的目录路径。
# - insert(0, ...)：表示在 sys.path 的第 0 个位置（即最前面）插入一个新路径，确保 "src" 目录的优先级最高。
# - str(project_root / "src")：通过 pathlib 库将项目根目录（project_root）与 "src" 拼接，得到 "src" 目录的绝对路径，并转换为字符串格式。
# 这样设置后，可以保证即使系统中存在同名的第三方包，也会优先导入项目自定义的模块，避免命名冲突。
sys.path.insert(0, str(project_root / "src"))

# yolo_detector模块不需要指定具体路径的原因如下：
# 1. 已在前面通过sys.path.insert(0, str(project_root / "src"))将src目录添加到Python模块搜索路径的最前面。
# 2. 这样Python解释器在import yolo_detector时，会优先在src目录下查找yolo_detector包，无需写明具体路径。
# 3. 这种做法保证了项目结构的灵活性和可移植性，避免硬编码路径带来的维护问题。
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
            print("\n✅ 批量处理完成!")
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
        print("✅ 检测完成!")
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
        print("\n数据路径:")
        print(f"- 输入文件夹: {data_config.get('input_folder', 'N/A')}")
        print(f"- 输出文件夹: {data_config.get('output_folder', 'N/A')}")
        
        # 检测参数
        detection_config = config.get_detection_config()
        print("\n检测参数:")
        print(f"- 置信度阈值: {detection_config.get('confidence_threshold', 'N/A')}")
        print(f"- 最大检测数: {detection_config.get('max_detections', 'N/A')}")
        
    except Exception as e:
        print(f"获取系统信息失败: {e}")


def main():
    """主函数"""
    # 下面这行代码的作用是创建一个命令行参数解析器对象parser，使用的是argparse库中的ArgumentParser类。
    # 详细解释如下：
    # - argparse.ArgumentParser(...) 用于定义一个命令行工具的参数解析规则和帮助信息。
    # - 通过parser对象，可以方便地添加子命令（如web、batch、detect、info）和各种参数选项，实现灵活的命令行交互。
    # - 这样做的好处是：用户可以通过命令行传递不同的参数，控制程序的运行方式（如启动Web界面、批量处理、单张检测、显示信息等）。
    # - 该解析器还支持自动生成帮助文档（-h/--help），提升用户体验和易用性。
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
    # 这句话的含义和详细作用如下：
    # - parser.add_subparsers(...) 用于为命令行工具添加“子命令”支持，使程序可以根据不同子命令执行不同的功能模块。
    # - dest='command' 表示解析命令行参数时，会将用户输入的子命令名称保存到args.command属性，便于后续分支判断和处理。
    # - help='可用命令' 用于自动生成命令行帮助文档时，显示所有可用子命令的说明，提升用户体验。
    # - 返回值subparsers是一个子命令解析器对象，可以基于它继续添加具体的子命令（如web、batch、detect、info等）。
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
