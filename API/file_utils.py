"""
文件工具模块，用于从文件名中提取雪橇号码和日期信息
"""

import os
import re


def extract_skid_number(file_path):
    """
    从文件路径中提取雪橇号码
    
    参数:
        file_path (str): 文件路径，例如 'D:\\Python\\00雪橇打标\\20250521\\1006_20250521_021853.jpg'
        
    返回:
        str: 雪橇号码，例如 '1006'
    """
    # 获取文件名（不含路径）
    file_name = os.path.basename(file_path)
    
    # 使用正则表达式匹配文件名开头的数字部分
    match = re.match(r'^(\d+)_', file_name)
    if match:
        return match.group(1)
    
    return None


def extract_date(file_path):
    """
    从文件路径中提取日期
    
    参数:
        file_path (str): 文件路径，例如 'D:\\Python\\00雪橇打标\\20250521\\1006_20250521_021853.jpg'
        
    返回:
        str: 日期，例如 '20250521'
    """
    # 获取文件名（不含路径）
    file_name = os.path.basename(file_path)
    
    # 使用正则表达式匹配文件名中的日期部分（通常是8位数字格式的日期）
    match = re.search(r'_(\d{8})_', file_name)
    if match:
        return match.group(1)
    
    # 如果文件名中没有找到，尝试从路径中提取
    dir_name = os.path.dirname(file_path)
    last_dir = os.path.basename(dir_name)
    if re.match(r'^\d{8}$', last_dir):  # 检查目录名是否为8位数字（日期格式）
        return last_dir
    
    return None 


def extract_time(file_path):
    """
    从文件路径中提取时间
    
    参数:
        file_path (str): 文件路径，例如 'D:\\Python\\00雪橇打标\\20250521\\1006_20250521_021853.jpg'
        
    返回:
        str: 时间，例如 '021853'
    """
    # 获取文件名（不含路径）
    file_name = os.path.basename(file_path)
    
    # 使用正则表达式匹配文件名中的时间部分（通常是6位数字格式的时间）
    match = re.search(r'_\d{8}_(\d{6})', file_name)
    if match:
        return match.group(1)
    
    return None 