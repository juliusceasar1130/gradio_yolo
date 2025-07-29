# 创建者/修改者: chenliang；修改时间：2025年7月27日 23:20；主要修改内容：创建统一的Gradio界面

"""
Gradio界面模块

提供统一的Web界面，支持目标检测和图像分割功能
"""

import gradio as gr
import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Any
import logging

from ..config.settings import Config
from ..models.model_loader import ModelLoader
from ..core import ObjectDetector, SegmentationDetector, BatchProcessor
from ..core.image_processor import ImageProcessor
from ..core.result_processor import ResultProcessor
from ..utils import get_image_files, get_example_images, format_image_info, get_logger, setup_logging

logger = get_logger(__name__)


class GradioApp:
    """Gradio应用类"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化Gradio应用
        
        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.model_loader = ModelLoader(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.result_processor = ResultProcessor(self.config)
        
        # 初始化检测器
        self.object_detector = ObjectDetector(self.model_loader, self.config)
        self.segmentation_detector = SegmentationDetector(self.model_loader, self.config)
        
        # 当前状态
        self.current_detector = "detection"  # "detection" 或 "segmentation"
        self.current_filename = ""
        
        # UI配置
        ui_config = self.config.get_ui_config()
        self.theme = ui_config.get('theme', 'soft')
        self.title = ui_config.get('title', 'YOLO 目标检测与分割系统')
        self.description = ui_config.get('description', '上传图片或使用示例图片进行物体检测')
        
        logger.info("Gradio应用初始化完成")
    
    def get_current_detector(self):
        """获取当前检测器"""
        if self.current_detector == "detection":
            return self.object_detector
        else:
            return self.segmentation_detector
    
    def switch_detector(self, detector_type: str) -> str:
        """
        切换检测器类型
        
        Args:
            detector_type: 检测器类型
            
        Returns:
            状态信息
        """
        try:
            if detector_type == self.current_detector:
                return f"当前已是{detector_type}模式"
            
            self.current_detector = detector_type
            detector = self.get_current_detector()
            
            # 预加载模型
            if not detector.is_model_loaded():
                success = detector.load_model()
                if not success:
                    return f"切换到{detector_type}模式失败：模型加载失败"
            
            mode_name = "目标检测" if detector_type == "detection" else "图像分割"
            logger.info(f"切换到{mode_name}模式")
            return f"已切换到{mode_name}模式"
            
        except Exception as e:
            logger.error(f"切换检测器失败: {e}")
            return f"切换失败: {str(e)}"
    
    def detect_image(self, image: Optional[Image.Image], 
                    confidence: float, 
                    detector_type: str) -> Tuple[Optional[np.ndarray], str, str]:
        """
        检测图像
        
        Args:
            image: 输入图像
            confidence: 置信度阈值
            detector_type: 检测器类型
            
        Returns:
            (结果图像, 统计信息, 状态信息)
        """
        try:
            if image is None:
                return None, "未上传图片", "请上传图片"
            
            # 切换检测器
            switch_msg = self.switch_detector(detector_type)
            
            # 获取当前检测器
            detector = self.get_current_detector()
            
            # 执行检测
            result = detector.detect(image, conf=confidence)
            
            if result is None:
                return None, "检测失败", switch_msg + "\n检测失败，请检查图像和模型"
            
            # 获取可视化结果
            vis_image = result.get_visualization(convert_to_rgb=True)
            
            # 获取统计信息
            stats_text = result.format_statistics()
            
            # 记录检测结果
            self.result_processor.process_single_result(result, self.current_filename)
            
            logger.info(f"检测完成: {result.get_statistics()['total_detections']} 个对象")
            return vis_image, stats_text, switch_msg + "\n检测完成"
            
        except Exception as e:
            logger.error(f"检测图像失败: {e}")
            return None, f"检测失败: {str(e)}", f"检测过程中出现错误: {str(e)}"
    
    def update_image_info(self, image: Optional[Image.Image]) -> str:
        """
        更新图像信息
        
        Args:
            image: 输入图像
            
        Returns:
            图像信息文本
        """
        try:
            if image is None:
                self.current_filename = ""
                return "未上传图片"
            
            # 使用工具函数格式化图像信息
            info_text = format_image_info(image, self.current_filename)
            return info_text
            
        except Exception as e:
            logger.error(f"更新图像信息失败: {e}")
            return f"获取图像信息失败: {str(e)}"
    
    def use_example_image(self, evt: gr.SelectData) -> Tuple[Optional[Image.Image], str]:
        """
        使用示例图像
        
        Args:
            evt: 选择事件
            
        Returns:
            (图像, 图像信息)
        """
        try:
            # 获取示例图像列表
            input_folder = self.config.get('data.input_folder')
            if not os.path.exists(input_folder):
                return None, "示例图像文件夹不存在"
            
            example_files = get_example_images(input_folder, max_count=12)
            
            if not example_files or evt.index >= len(example_files):
                return None, "示例图像不可用"
            
            # 加载选中的图像
            selected_path = example_files[evt.index]
            image = Image.open(selected_path)
            
            # 更新文件名
            self.current_filename = os.path.basename(selected_path)
            
            # 获取图像信息
            info_text = format_image_info(image, self.current_filename)
            
            logger.info(f"使用示例图像: {self.current_filename}")
            return image, info_text
            
        except Exception as e:
            logger.error(f"使用示例图像失败: {e}")
            return None, f"加载示例图像失败: {str(e)}"
    
    def clear_all(self) -> Tuple[None, str, str, str]:
        """
        清除所有内容
        
        Returns:
            (None, 空字符串, 图像信息, 状态信息)
        """
        self.current_filename = ""
        return None, "", "未上传图片", "已清除所有内容"
    
    def get_model_info(self) -> str:
        """获取模型信息"""
        try:
            available_models = self.model_loader.list_available_models()
            
            info_parts = ["### 模型信息"]
            
            for model_type, model_info in available_models.items():
                status = "✅ 可用" if model_info['exists'] else "❌ 不可用"
                loaded = "已加载" if model_info['loaded'] else "未加载"
                
                info_parts.append(f"**{model_type.title()}模型**: {status} ({loaded})")
                if model_info['exists']:
                    info_parts.append(f"- 路径: {model_info['path']}")
                    info_parts.append(f"- 描述: {model_info.get('description', '无描述')}")
            
            return "\n".join(info_parts)
            
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return f"获取模型信息失败: {str(e)}"
    
    def get_example_gallery(self) -> list:
        """获取示例图像画廊"""
        try:
            input_folder = self.config.get('data.input_folder')
            if not os.path.exists(input_folder):
                return []
            
            example_files = get_example_images(input_folder, max_count=12)
            
            gallery_items = []
            for img_path in example_files:
                try:
                    # 创建缩略图
                    image = Image.open(img_path)
                    filename = os.path.basename(img_path)
                    gallery_items.append((img_path, filename))
                except Exception as e:
                    logger.warning(f"加载示例图像失败: {img_path}, 错误: {e}")
                    continue
            
            return gallery_items
            
        except Exception as e:
            logger.error(f"获取示例画廊失败: {e}")
            return []
    
    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        try:
            # 设置主题
            if self.theme == "soft":
                theme = gr.themes.Soft()
            elif self.theme == "monochrome":
                theme = gr.themes.Monochrome()
            else:
                theme = gr.themes.Default()
            
            with gr.Blocks(theme=theme, title=self.title) as demo:
                # 标题和描述
                gr.Markdown(f"# {self.title}")
                gr.Markdown(f"### {self.description}")
                
                # 检查示例图像
                example_gallery = self.get_example_gallery()
                if not example_gallery:
                    gr.Markdown("⚠️ 警告: 未找到示例图像，请检查配置中的输入文件夹路径")
                
                with gr.Row():
                    # 左侧：输入区域
                    with gr.Column(scale=1):
                        # 检测器选择
                        detector_radio = gr.Radio(
                            choices=[("目标检测", "detection"), ("图像分割", "segmentation")],
                            value="detection",
                            label="检测模式",
                            info="选择检测类型"
                        )
                        
                        # 图像输入
                        input_image = gr.Image(
                            label="输入图片", 
                            type="pil",
                            height=300
                        )
                        
                        # 图像信息
                        image_info = gr.Markdown("未上传图片", label="图片信息")
                        
                        # 置信度滑块
                        conf_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.25,
                            step=0.05,
                            label="置信度阈值",
                            info="调整检测阈值（值越低检测越多）"
                        )
                        
                        # 操作按钮
                        with gr.Row():
                            detect_btn = gr.Button("开始检测", variant="primary")
                            clear_btn = gr.Button("清除", variant="secondary")
                        
                        # 状态信息
                        status_info = gr.Markdown("就绪", label="状态")
                        
                        # 模型信息（折叠）
                        with gr.Accordion("模型信息", open=False):
                            model_info = gr.Markdown(self.get_model_info())
                    
                    # 右侧：输出区域
                    with gr.Column(scale=1):
                        # 输出图像
                        output_image = gr.Image(
                            label="检测结果",
                            height=400
                        )
                        
                        # 统计信息
                        stats_output = gr.Markdown("", label="检测统计")
                        
                        # 示例图像画廊
                        if example_gallery:
                            gr.Markdown("### 示例图片")
                            examples_gallery = gr.Gallery(
                                value=example_gallery,
                                columns=3,
                                rows=4,
                                object_fit="scale-down",
                                height="300px",
                                label="点击选择示例图片"
                            )
                
                # 事件绑定
                # 检测按钮点击
                detect_btn.click(
                    fn=self.detect_image,
                    inputs=[input_image, conf_slider, detector_radio],
                    outputs=[output_image, stats_output, status_info]
                )
                
                # 清除按钮点击
                clear_btn.click(
                    fn=self.clear_all,
                    outputs=[input_image, stats_output, image_info, status_info]
                )
                
                # 图像变化时更新信息
                input_image.change(
                    fn=self.update_image_info,
                    inputs=[input_image],
                    outputs=[image_info]
                )
                
                # 图像变化时自动检测
                input_image.change(
                    fn=self.detect_image,
                    inputs=[input_image, conf_slider, detector_radio],
                    outputs=[output_image, stats_output, status_info]
                )
                
                # 示例图像选择
                if example_gallery:
                    examples_gallery.select(
                        fn=self.use_example_image,
                        outputs=[input_image, image_info]
                    )
            
            logger.info("Gradio界面创建完成")
            return demo
            
        except Exception as e:
            logger.error(f"创建Gradio界面失败: {e}")
            raise


def create_gradio_interface(config: Optional[Config] = None) -> gr.Blocks:
    """
    创建Gradio界面的便捷函数
    
    Args:
        config: 配置对象
        
    Returns:
        Gradio界面对象
    """
    # 设置日志
    setup_logging(config)
    
    # 创建应用
    app = GradioApp(config)
    
    # 创建界面
    return app.create_interface()
