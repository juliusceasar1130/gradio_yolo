# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建文件操作工具函数模块

"""
文件操作工具函数

提供文件和目录操作的常用功能
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging
import shutil

logger = logging.getLogger(__name__)


def get_image_files(folder_path: str, 
                   extensions: List[str] = None,
                   recursive: bool = False) -> List[str]:
    """
    获取文件夹中的所有图像文件
    
    Args:
        folder_path: 文件夹路径
        extensions: 支持的文件扩展名列表
        recursive: 是否递归搜索子文件夹
        
    Returns:
        图像文件路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    image_files = []
    
    try:
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            logger.warning(f"文件夹不存在: {folder_path}")
            return []
        
        if not folder_path.is_dir():
            logger.warning(f"路径不是文件夹: {folder_path}")
            return []
        
        # 搜索模式
        pattern = "**/*" if recursive else "*"
        
        for ext in extensions:
            # 搜索指定扩展名的文件
            files = folder_path.glob(f"{pattern}{ext}")
            files_lower = folder_path.glob(f"{pattern}{ext.lower()}")
            files_upper = folder_path.glob(f"{pattern}{ext.upper()}")
            
            # 合并结果并转换为字符串路径
            for file_path in list(files) + list(files_lower) + list(files_upper):
                if file_path.is_file():
                    str_path = str(file_path)
                    if str_path not in image_files:  # 避免重复
                        image_files.append(str_path)
        
        # 排序文件列表
        image_files.sort()
        
        logger.info(f"找到 {len(image_files)} 个图像文件在 {folder_path}")
        return image_files
        
    except Exception as e:
        logger.error(f"获取图像文件失败: {folder_path}, 错误: {e}")
        return []


def ensure_dir(dir_path: Union[str, Path]) -> bool:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        dir_path: 目录路径
        
    Returns:
        是否成功创建或目录已存在
    """
    try:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"目录确保存在: {dir_path}")
        return True
    except Exception as e:
        logger.error(f"创建目录失败: {dir_path}, 错误: {e}")
        return False


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    获取文件信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        包含文件信息的字典
    """
    info = {}
    
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return {}
        
        stat = file_path.stat()
        
        info.update({
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix,
            'parent': str(file_path.parent),
            'absolute_path': str(file_path.absolute()),
            'size_bytes': stat.st_size,
            'size_kb': stat.st_size / 1024,
            'size_mb': stat.st_size / (1024 * 1024),
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir()
        })
        
        logger.debug(f"获取文件信息成功: {file_path}")
        return info
        
    except Exception as e:
        logger.error(f"获取文件信息失败: {file_path}, 错误: {e}")
        return {}


def get_example_images(folder_path: str, 
                      max_count: int = 10,
                      extensions: List[str] = None) -> List[str]:
    """
    获取示例图像文件列表
    
    Args:
        folder_path: 图像文件夹路径
        max_count: 最大返回数量
        extensions: 支持的文件扩展名
        
    Returns:
        示例图像文件路径列表
    """
    try:
        image_files = get_image_files(folder_path, extensions)
        
        if not image_files:
            logger.warning(f"未找到示例图像: {folder_path}")
            return []
        
        # 限制数量
        example_files = image_files[:max_count]
        
        logger.info(f"获取 {len(example_files)} 个示例图像")
        return example_files
        
    except Exception as e:
        logger.error(f"获取示例图像失败: {folder_path}, 错误: {e}")
        return []


def safe_filename(filename: str, replacement: str = "_") -> str:
    """
    生成安全的文件名，移除或替换非法字符
    
    Args:
        filename: 原始文件名
        replacement: 替换字符
        
    Returns:
        安全的文件名
    """
    import re
    
    # 定义非法字符
    illegal_chars = r'[<>:"/\\|?*]'
    
    # 替换非法字符
    safe_name = re.sub(illegal_chars, replacement, filename)
    
    # 移除首尾空格和点
    safe_name = safe_name.strip(' .')
    
    # 确保文件名不为空
    if not safe_name:
        safe_name = "unnamed"
    
    logger.debug(f"生成安全文件名: {filename} -> {safe_name}")
    return safe_name


def copy_file(src_path: Union[str, Path], 
              dst_path: Union[str, Path],
              create_dirs: bool = True) -> bool:
    """
    复制文件
    
    Args:
        src_path: 源文件路径
        dst_path: 目标文件路径
        create_dirs: 是否创建目标目录
        
    Returns:
        是否复制成功
    """
    try:
        src_path = Path(src_path)
        dst_path = Path(dst_path)
        
        if not src_path.exists():
            logger.error(f"源文件不存在: {src_path}")
            return False
        
        if create_dirs:
            ensure_dir(dst_path.parent)
        
        shutil.copy2(src_path, dst_path)
        logger.info(f"文件复制成功: {src_path} -> {dst_path}")
        return True
        
    except Exception as e:
        logger.error(f"文件复制失败: {src_path} -> {dst_path}, 错误: {e}")
        return False


def move_file(src_path: Union[str, Path], 
              dst_path: Union[str, Path],
              create_dirs: bool = True) -> bool:
    """
    移动文件
    
    Args:
        src_path: 源文件路径
        dst_path: 目标文件路径
        create_dirs: 是否创建目标目录
        
    Returns:
        是否移动成功
    """
    try:
        src_path = Path(src_path)
        dst_path = Path(dst_path)
        
        if not src_path.exists():
            logger.error(f"源文件不存在: {src_path}")
            return False
        
        if create_dirs:
            ensure_dir(dst_path.parent)
        
        shutil.move(str(src_path), str(dst_path))
        logger.info(f"文件移动成功: {src_path} -> {dst_path}")
        return True
        
    except Exception as e:
        logger.error(f"文件移动失败: {src_path} -> {dst_path}, 错误: {e}")
        return False


def delete_file(file_path: Union[str, Path]) -> bool:
    """
    删除文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否删除成功
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return True  # 文件不存在也算删除成功
        
        file_path.unlink()
        logger.info(f"文件删除成功: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"文件删除失败: {file_path}, 错误: {e}")
        return False


def get_unique_filename(file_path: Union[str, Path]) -> Path:
    """
    获取唯一的文件名，如果文件已存在则添加数字后缀
    
    Args:
        file_path: 原始文件路径
        
    Returns:
        唯一的文件路径
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return file_path
    
    stem = file_path.stem
    suffix = file_path.suffix
    parent = file_path.parent
    
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name
        
        if not new_path.exists():
            logger.debug(f"生成唯一文件名: {file_path} -> {new_path}")
            return new_path
        
        counter += 1
        
        # 防止无限循环
        if counter > 9999:
            import time
            timestamp = int(time.time())
            new_name = f"{stem}_{timestamp}{suffix}"
            new_path = parent / new_name
            logger.debug(f"使用时间戳生成唯一文件名: {new_path}")
            return new_path


def clean_directory(dir_path: Union[str, Path], 
                   pattern: str = "*",
                   keep_dirs: bool = True) -> int:
    """
    清理目录中的文件
    
    Args:
        dir_path: 目录路径
        pattern: 文件匹配模式
        keep_dirs: 是否保留子目录
        
    Returns:
        删除的文件数量
    """
    try:
        dir_path = Path(dir_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"目录不存在或不是目录: {dir_path}")
            return 0
        
        deleted_count = 0
        
        for item in dir_path.glob(pattern):
            if item.is_file():
                item.unlink()
                deleted_count += 1
            elif item.is_dir() and not keep_dirs:
                shutil.rmtree(item)
                deleted_count += 1
        
        logger.info(f"清理目录完成: {dir_path}, 删除 {deleted_count} 个项目")
        return deleted_count
        
    except Exception as e:
        logger.error(f"清理目录失败: {dir_path}, 错误: {e}")
        return 0
