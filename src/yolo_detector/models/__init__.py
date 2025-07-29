# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建models包初始化文件
"""
模型管理模块

负责YOLO模型的加载、管理和初始化
"""

from .model_loader import ModelLoader

__all__ = ["ModelLoader"]
