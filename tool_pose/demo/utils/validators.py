#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证工具模块

主要修改内容：
1. 文件/路径验证
2. 配置验证
修改时间：2025年1月31日
最后更新：2025年11月2日
"""

from pathlib import Path
from typing import Optional, Union


def validate_path(path: Union[str, Path], path_type: str = "文件") -> tuple[bool, Optional[str]]:
    """
    验证路径是否存在
    
    Args:
        path: 路径
        path_type: 路径类型（用于错误提示）
        
    Returns:
        (是否有效, 错误信息)
    """
    path_obj = Path(path) if isinstance(path, str) else path
    
    if not path_obj.exists():
        return False, f"❌ {path_type}不存在: {path_obj}"
    
    return True, None


def validate_paths(*paths: tuple[Union[str, Path], str]) -> tuple[bool, Optional[str]]:
    """
    批量验证路径
    
    Args:
        *paths: 路径元组列表，每个元组为 (path, path_type)
        
    Returns:
        (是否全部有效, 错误信息)
    """
    for path, path_type in paths:
        is_valid, error_msg = validate_path(path, path_type)
        if not is_valid:
            return False, error_msg
    
    return True, None

