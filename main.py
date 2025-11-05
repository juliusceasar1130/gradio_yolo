# 创建者/修改者: chenliang
# 修改时间：2025年11月02日 11:30
# 主要修改内容：
# 1. 简化为主程序，只保留Web界面启动功能
# 2. 删除批量处理、单张检测等功能
# 3. 简化命令行参数
# 4. 适配工具关键点检测界面

"""
工具关键点检测系统 - 主入口文件

提供Web界面启动功能
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# 添加src目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from yolo_detector.ui import create_gradio_interface
from yolo_detector.utils import setup_logging, get_logger


def setup_environment():
    """设置环境"""
    try:
        # 设置日志（使用默认配置）
        setup_logging()
        
        logger = get_logger(__name__)
        logger.info("环境设置完成")
        
        return True
        
    except Exception as e:
        print(f"环境设置失败: {e}")
        sys.exit(1)


def run_web_interface(
    config_path: Optional[str] = None,
    output_dir: str = "outputs/tool_pose",
    port: int = 7861,
    share: bool = False,
    debug: bool = False
):
    """
    运行Web界面
    
    Args:
        config_path: 工具关键点配置文件路径，如果为None则使用默认路径
                    (src/yolo_detector/config/tool_pose_config.yaml)
        output_dir: 输出目录路径
        port: 服务器端口
        share: 是否创建公共链接
        debug: 是否启用调试模式
    """
    logger = get_logger(__name__)
    
    try:
        logger.info("启动工具关键点检测Web界面...")
        
        # 创建Gradio界面
        demo = create_gradio_interface(
            config_path=config_path,
            output_dir=output_dir
        )
        
        # 启动界面
        # 如果端口被占用，尝试自动寻找可用端口
        import socket
        def is_port_available(port_num):
            """检查端口是否可用"""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port_num))
                    return True
                except OSError:
                    return False
        
        actual_port = port
        if not is_port_available(port):
            logger.warning(f"端口 {port} 被占用，尝试寻找可用端口...")
            for p in range(port, port + 10):
                if is_port_available(p):
                    actual_port = p
                    logger.info(f"找到可用端口: {actual_port}")
                    break
            else:
                logger.error(f"无法找到可用端口（尝试范围: {port}-{port+9}）")
        
        demo.launch(
            share=share,
            debug=debug,
            server_name="127.0.0.1",
            server_port=actual_port,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"启动Web界面失败: {e}", exc_info=True)
        print(f"启动Web界面失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="工具关键点检测系统 - 实时检测工具关键点并计算角度",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 启动Web界面（默认配置）
  python main.py web
  
  # 启动Web界面（自定义端口）
  python main.py web --port 7862
  
  # 启动Web界面（创建公共链接）
  python main.py web --share
  
  # 启动Web界面（启用调试模式）
  python main.py web --debug
  
  # 启动Web界面（指定配置文件）
  python main.py web --config src/yolo_detector/config/tool_pose_config.yaml
        """
    )
    
    # Web界面命令
    parser.add_argument(
        'command',
        choices=['web'],
        help='启动Web界面'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='工具关键点配置文件路径（默认: src/yolo_detector/config/tool_pose_config.yaml）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/tool_pose',
        help='输出目录路径（默认: outputs/tool_pose）'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7861,
        help='服务器端口（默认: 7861）'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='创建公共链接'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置环境
    setup_environment()
    
    # 执行命令
    if args.command == 'web':
        run_web_interface(
            config_path=args.config,
            output_dir=args.output_dir,
            port=args.port,
            share=args.share,
            debug=args.debug
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
