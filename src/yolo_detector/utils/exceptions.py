# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:45；主要修改内容：创建自定义异常类和错误处理机制

"""
异常处理模块

定义项目专用的异常类和错误处理机制
"""

import traceback
import functools
from typing import Optional, Dict, Any, Callable, Union
import logging


class YOLODetectorError(Exception):
    """YOLO检测器基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        """
        初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 错误详情
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'UNKNOWN_ERROR'
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Code: {self.error_code}, Details: {self.details})"
        return f"{self.message} (Code: {self.error_code})"


class ConfigurationError(YOLODetectorError):
    """配置相关错误"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(message, error_code='CONFIG_ERROR', **kwargs)
        if config_key:
            self.details['config_key'] = config_key


class ModelLoadError(YOLODetectorError):
    """模型加载错误"""
    
    def __init__(self, message: str, model_path: str = None, model_type: str = None, **kwargs):
        super().__init__(message, error_code='MODEL_LOAD_ERROR', **kwargs)
        if model_path:
            self.details['model_path'] = model_path
        if model_type:
            self.details['model_type'] = model_type


class ImageProcessingError(YOLODetectorError):
    """图像处理错误"""
    
    def __init__(self, message: str, image_path: str = None, operation: str = None, **kwargs):
        super().__init__(message, error_code='IMAGE_PROCESSING_ERROR', **kwargs)
        if image_path:
            self.details['image_path'] = image_path
        if operation:
            self.details['operation'] = operation


class DetectionError(YOLODetectorError):
    """检测过程错误"""
    
    def __init__(self, message: str, detector_type: str = None, image_path: str = None, **kwargs):
        super().__init__(message, error_code='DETECTION_ERROR', **kwargs)
        if detector_type:
            self.details['detector_type'] = detector_type
        if image_path:
            self.details['image_path'] = image_path


class BatchProcessingError(YOLODetectorError):
    """批量处理错误"""
    
    def __init__(self, message: str, batch_size: int = None, failed_count: int = None, **kwargs):
        super().__init__(message, error_code='BATCH_PROCESSING_ERROR', **kwargs)
        if batch_size is not None:
            self.details['batch_size'] = batch_size
        if failed_count is not None:
            self.details['failed_count'] = failed_count


class ValidationError(YOLODetectorError):
    """验证错误"""
    
    def __init__(self, message: str, validation_type: str = None, **kwargs):
        super().__init__(message, error_code='VALIDATION_ERROR', **kwargs)
        if validation_type:
            self.details['validation_type'] = validation_type


class FileOperationError(YOLODetectorError):
    """文件操作错误"""
    
    def __init__(self, message: str, file_path: str = None, operation: str = None, **kwargs):
        super().__init__(message, error_code='FILE_OPERATION_ERROR', **kwargs)
        if file_path:
            self.details['file_path'] = file_path
        if operation:
            self.details['operation'] = operation


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化错误处理器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_stats = {
            'total_errors': 0,
            'error_types': {},
            'recent_errors': []
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文
            
        Returns:
            错误处理结果
        """
        # 更新统计
        self.error_stats['total_errors'] += 1
        error_type = type(error).__name__
        self.error_stats['error_types'][error_type] = self.error_stats['error_types'].get(error_type, 0) + 1
        
        # 创建错误记录
        error_record = {
            'error_type': error_type,
            'message': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        # 如果是自定义异常，添加详细信息
        if isinstance(error, YOLODetectorError):
            error_record.update(error.to_dict())
        
        # 记录到最近错误列表
        self.error_stats['recent_errors'].append(error_record)
        if len(self.error_stats['recent_errors']) > 10:
            self.error_stats['recent_errors'].pop(0)
        
        # 记录日志
        self.logger.error(f"Error handled: {error_type} - {str(error)}", extra={'error_context': context})
        
        return error_record
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return self.error_stats.copy()
    
    def clear_stats(self):
        """清除错误统计"""
        self.error_stats = {
            'total_errors': 0,
            'error_types': {},
            'recent_errors': []
        }


def handle_exceptions(error_handler: ErrorHandler = None, 
                     reraise: bool = True,
                     default_return: Any = None,
                     log_errors: bool = True):
    """
    异常处理装饰器
    
    Args:
        error_handler: 错误处理器
        reraise: 是否重新抛出异常
        default_return: 异常时的默认返回值
        log_errors: 是否记录错误日志
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取函数上下文
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                # 处理错误
                if error_handler:
                    error_handler.handle_error(e, context)
                elif log_errors:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
                
                # 决定是否重新抛出异常
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, 
                error_handler: ErrorHandler = None,
                default_return: Any = None,
                context: Dict[str, Any] = None,
                **kwargs) -> tuple[bool, Any]:
    """
    安全执行函数
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        error_handler: 错误处理器
        default_return: 默认返回值
        context: 执行上下文
        **kwargs: 函数关键字参数
        
    Returns:
        (是否成功, 结果或错误信息)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        # 处理错误
        if error_handler:
            error_record = error_handler.handle_error(e, context)
            return False, error_record
        else:
            return False, {'error': str(e), 'type': type(e).__name__}


def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
    """
    创建错误上下文
    
    Args:
        operation: 操作名称
        **kwargs: 额外的上下文信息
        
    Returns:
        错误上下文字典
    """
    context = {
        'operation': operation,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }
    context.update(kwargs)
    return context


def validate_and_raise(condition: bool, 
                      error_class: type = ValidationError,
                      message: str = "Validation failed",
                      **error_kwargs):
    """
    验证条件并在失败时抛出异常
    
    Args:
        condition: 验证条件
        error_class: 异常类
        message: 错误消息
        **error_kwargs: 异常的额外参数
    """
    if not condition:
        raise error_class(message, **error_kwargs)


# 全局错误处理器实例
_global_error_handler = None


def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def setup_error_handling(logger: Optional[logging.Logger] = None) -> ErrorHandler:
    """
    设置全局错误处理
    
    Args:
        logger: 日志记录器
        
    Returns:
        错误处理器实例
    """
    global _global_error_handler
    _global_error_handler = ErrorHandler(logger)
    return _global_error_handler
