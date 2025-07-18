"""
Loguru日志配置模块
"""

import os
import sys
import time
from loguru import logger

# 获取当前日期，格式为年月日
def get_date_folder():
    return time.strftime("%Y%m%d")

# 配置日志系统
def setup_logger():
    # 创建日志目录
    log_dir = os.path.join("API", "log", get_date_folder())
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名，使用时间戳
    log_file = os.path.join(log_dir, f"{time.strftime('%H%M%S')}.log")
    
    # 移除默认的处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
    
    # 添加文件输出
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",  # 日志文件大小达到10MB时轮转
        retention="1 week",  # 保留1周的日志
        encoding="utf-8"
    )
    
    logger.info(f"日志系统已初始化，日志文件: {log_file}")
    return logger 