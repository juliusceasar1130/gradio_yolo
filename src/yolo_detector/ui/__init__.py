# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建ui包初始化文件
"""
用户界面模块

包含Gradio界面和相关的UI组件
"""

from .gradio_app import create_gradio_interface, GradioApp

__all__ = [
    "create_gradio_interface",
    "GradioApp"
]
