# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:45；主要修改内容：创建统一的日志系统

"""
日志系统模块

提供统一的日志记录功能，支持多种输出格式和级别
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 添加颜色
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_data'):
            log_entry['extra'] = record.extra_data
        
        return json.dumps(log_entry, ensure_ascii=False)


class LoggerManager:
    """日志管理器"""
    
    def __init__(self, config=None):
        """
        初始化日志管理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.loggers = {}
        self._setup_default_config()
        self._setup_root_logger()
    
    def _setup_default_config(self):
        """设置默认配置"""
        self.default_config = {
            'level': 'INFO',
            'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}',
            'file': 'logs/yolo_detector.log',
            'rotation': '10 MB',
            'retention': '7 days',
            'console_output': True,
            'file_output': True,
            'json_format': False,
            'colored_output': True
        }
        
        # 从配置文件更新
        if self.config:
            logging_config = self.config.get_logging_config()
            self.default_config.update(logging_config)
    
    def _setup_root_logger(self):
        """设置根日志记录器"""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.default_config['level'].upper()))
        
        # 清除现有处理器
        root_logger.handlers.clear()
        
        # 添加控制台处理器
        if self.default_config['console_output']:
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        if self.default_config['file_output']:
            file_handler = self._create_file_handler()
            if file_handler:
                root_logger.addHandler(file_handler)
    
    def _create_console_handler(self) -> logging.Handler:
        """创建控制台处理器"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, self.default_config['level'].upper()))
        
        if self.default_config['json_format']:
            formatter = JSONFormatter()
        elif self.default_config['colored_output']:
            formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_file_handler(self) -> Optional[logging.Handler]:
        """创建文件处理器"""
        try:
            log_file = Path(self.default_config['file'])
            
            # 确保日志目录存在
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建轮转文件处理器
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self._parse_size(self.default_config['rotation']),
                backupCount=5,
                encoding='utf-8'
            )
            
            handler.setLevel(getattr(logging, self.default_config['level'].upper()))
            
            if self.default_config['json_format']:
                formatter = JSONFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            
            handler.setFormatter(formatter)
            return handler
            
        except Exception as e:
            print(f"创建文件日志处理器失败: {e}")
            return None
    
    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串"""
        size_str = size_str.upper().strip()
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            # 默认为字节
            return int(size_str)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取指定名称的日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            日志记录器对象
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def set_level(self, level: Union[str, int]):
        """设置日志级别"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        for handler in root_logger.handlers:
            handler.setLevel(level)
    
    def add_file_handler(self, log_file: str, level: str = None) -> bool:
        """
        添加额外的文件处理器
        
        Args:
            log_file: 日志文件路径
            level: 日志级别
            
        Returns:
            是否添加成功
        """
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            
            if level:
                handler.setLevel(getattr(logging, level.upper()))
            
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            
            return True
            
        except Exception as e:
            print(f"添加文件处理器失败: {e}")
            return False
    
    def log_performance(self, logger_name: str, operation: str, 
                       duration: float, extra_data: Dict[str, Any] = None):
        """
        记录性能日志
        
        Args:
            logger_name: 日志记录器名称
            operation: 操作名称
            duration: 持续时间（秒）
            extra_data: 额外数据
        """
        logger = self.get_logger(logger_name)
        
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'duration_ms': duration * 1000
        }
        
        if extra_data:
            perf_data.update(extra_data)
        
        # 创建日志记录并添加额外数据
        record = logger.makeRecord(
            logger.name, logging.INFO, '', 0,
            f"Performance: {operation} completed in {duration:.3f}s",
            (), None
        )
        record.extra_data = perf_data
        
        logger.handle(record)
    
    def log_detection_result(self, logger_name: str, image_path: str, 
                           result_stats: Dict[str, Any]):
        """
        记录检测结果日志
        
        Args:
            logger_name: 日志记录器名称
            image_path: 图像路径
            result_stats: 检测结果统计
        """
        logger = self.get_logger(logger_name)
        
        detection_data = {
            'image_path': image_path,
            'total_detections': result_stats.get('total_detections', 0),
            'classes': result_stats.get('classes', {}),
            'has_detections': result_stats.get('has_detections', False)
        }
        
        message = f"Detection completed: {os.path.basename(image_path)} - {detection_data['total_detections']} objects"
        
        record = logger.makeRecord(
            logger.name, logging.INFO, '', 0, message, (), None
        )
        record.extra_data = detection_data
        
        logger.handle(record)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        stats = {
            'active_loggers': len(self.loggers),
            'log_level': self.default_config['level'],
            'console_output': self.default_config['console_output'],
            'file_output': self.default_config['file_output'],
            'log_file': self.default_config['file']
        }
        
        # 检查日志文件大小
        if self.default_config['file_output']:
            log_file = Path(self.default_config['file'])
            if log_file.exists():
                stats['log_file_size'] = log_file.stat().st_size
                stats['log_file_size_mb'] = stats['log_file_size'] / (1024 * 1024)
        
        return stats


# 全局日志管理器实例
_logger_manager = None


def setup_logging(config=None) -> LoggerManager:
    """
    设置日志系统
    
    Args:
        config: 配置对象
        
    Returns:
        日志管理器实例
    """
    global _logger_manager
    _logger_manager = LoggerManager(config)
    return _logger_manager


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称，默认使用调用模块名
        
    Returns:
        日志记录器对象
    """
    global _logger_manager
    
    if _logger_manager is None:
        _logger_manager = LoggerManager()
    
    if name is None:
        # 自动获取调用模块名
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return _logger_manager.get_logger(name)


def log_performance(operation: str, duration: float, **kwargs):
    """
    记录性能日志的便捷函数
    
    Args:
        operation: 操作名称
        duration: 持续时间
        **kwargs: 额外数据
    """
    global _logger_manager
    
    if _logger_manager is None:
        _logger_manager = LoggerManager()
    
    # 获取调用模块名
    import inspect
    frame = inspect.currentframe().f_back
    logger_name = frame.f_globals.get('__name__', 'performance')
    
    _logger_manager.log_performance(logger_name, operation, duration, kwargs)
