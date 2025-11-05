# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建utils包初始化文件
"""
工具函数模块

包含图像处理、文件操作、日志等工具函数
"""

from .image_utils import (
    load_image, load_image_pil, convert_color_space, resize_image,
    get_image_info, format_image_info, validate_image_format, create_thumbnail
)
from .file_utils import (
    get_image_files, ensure_dir, get_file_info, get_example_images,
    safe_filename, copy_file, move_file, delete_file, get_unique_filename, clean_directory
)
from .logger import setup_logging, get_logger, log_performance, LoggerManager
from .exceptions import (
    YOLODetectorError, ConfigurationError, ModelLoadError, ImageProcessingError,
    DetectionError, BatchProcessingError, ValidationError, FileOperationError,
    ErrorHandler, handle_exceptions, safe_execute, get_error_handler, setup_error_handling
)
from .tool_pose_utils import ToolPoseDetector
from .tool_pose_angle_calculator import (
    load_angle_config,
    calculate_angle,
    calculate_angles_for_object,
    annotate_angles_on_image,
    get_keypoint_index_by_name,
    draw_text_with_pil,
    get_system_font_path
)
from .tool_pose_json_utils import (
    convert_to_json_serializable,
    save_results_to_json
)

__all__ = [
    # 图像工具函数
    "load_image",
    "load_image_pil",
    "convert_color_space",
    "resize_image",
    "get_image_info",
    "format_image_info",
    "validate_image_format",
    "create_thumbnail",
    # 文件工具函数
    "get_image_files",
    "ensure_dir",
    "get_file_info",
    "get_example_images",
    "safe_filename",
    "copy_file",
    "move_file",
    "delete_file",
    "get_unique_filename",
    "clean_directory",
    # 日志系统
    "setup_logging",
    "get_logger",
    "log_performance",
    "LoggerManager",
    # 异常处理
    "YOLODetectorError",
    "ConfigurationError",
    "ModelLoadError",
    "ImageProcessingError",
    "DetectionError",
    "BatchProcessingError",
    "ValidationError",
    "FileOperationError",
    "ErrorHandler",
    "handle_exceptions",
    "safe_execute",
    "get_error_handler",
    "setup_error_handling",
    # 工具关键点检测
    "ToolPoseDetector",
    # 工具关键点角度计算
    "load_angle_config",
    "calculate_angle",
    "calculate_angles_for_object",
    "annotate_angles_on_image",
    "get_keypoint_index_by_name",
    "draw_text_with_pil",
    "get_system_font_path",
    # 工具关键点JSON工具
    "convert_to_json_serializable",
    "save_results_to_json"
]
