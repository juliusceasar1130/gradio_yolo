# 创建者/修改者: chenliang
# 修改时间：2025年11月02日 10:00
# 主要修改内容：
# 1. 完全改造为工具关键点检测界面
# 2. 删除目标检测、图像分割功能
# 3. 实现摄像头实时预览
# 4. 实现检测和结果保存功能
# 5. 支持 Axis MJPEG 摄像头连接
# 6. 添加摄像头类型选择和 Axis 配置界面
# 7. 修复自动刷新功能：改进JavaScript元素查找逻辑，添加多种查找方法、重试机制和DOM监听
# 8. 采用 Generator 方案实现实时摄像头预览（使用生成器函数 + .then() 触发）
# 9. 优化刷新速度：从配置文件读取preview_interval，默认30fps（0.033秒间隔）
# 10. 实现预览图像缩放优化：支持配置预览最大宽度/高度，减少网络传输数据量
# 11. 添加预览性能优化界面控件，支持实时调整预览尺寸和缩放设置

"""
工具关键点检测系统 - Gradio界面模块

提供基于Gradio的Web界面，支持：
- 摄像头实时采集和预览
- 工具关键点检测
- 角度计算和标注
- 结果保存（原始图、结果图、JSON）
"""

import gradio as gr
import os
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime
import json
import threading
import time

from ..core import CameraCapture
from ..utils import ToolPoseDetector, get_logger
from ..utils.file_utils import ensure_dir

logger = get_logger(__name__)


class ToolPoseGradioApp:
    """
    工具关键点检测Gradio应用类
    """
    
    def __init__(self, config_path: Optional[str] = None, output_dir: str = "outputs/tool_pose"):
        """
        初始化应用
        
        Args:
            config_path: 工具关键点配置文件路径，如果为None则使用默认路径
                        (src/yolo_detector/config/tool_pose_config.yaml)
            output_dir: 输出目录路径
        """
        # 初始化检测器
        self.detector = ToolPoseDetector(config_path)
        
        # 摄像头管理器
        self.camera: Optional[CameraCapture] = None
        
        # 输出目录
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir / "raw_images")
        ensure_dir(self.output_dir / "result_images")
        ensure_dir(self.output_dir / "json_results")
        
        # 自动刷新相关状态
        self.auto_refresh_enabled = False
        self.auto_refresh_thread: Optional[threading.Thread] = None
        self.auto_refresh_stop_event = threading.Event()
        
        # 从配置文件读取预览相关配置
        # 注意：更小的刷新间隔意味着更高的刷新率，但也会增加CPU和网络负载
        try:
            import yaml
            if config_path is None:
                config_dir = Path(__file__).parent.parent / "config"
                config_path = config_dir / "tool_pose_config.yaml"
            else:
                config_path = Path(config_path)
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    camera_config = config.get('camera', {})
                    
                    # 读取刷新间隔
                    self.preview_interval = camera_config.get('preview_interval', 0.033)
                    
                    # 读取预览优化配置
                    preview_config = camera_config.get('preview', {})
                    self.preview_max_width = preview_config.get('max_width')
                    self.preview_max_height = preview_config.get('max_height')
                    self.preview_enable_scale = preview_config.get('enable_scale', True)
                    interpolation_name = preview_config.get('interpolation', 'INTER_LINEAR')
                    
                    # 将字符串转换为OpenCV插值常量
                    interpolation_map = {
                        'INTER_NEAREST': cv2.INTER_NEAREST,
                        'INTER_LINEAR': cv2.INTER_LINEAR,
                        'INTER_CUBIC': cv2.INTER_CUBIC,
                        'INTER_AREA': cv2.INTER_AREA,
                        'INTER_LANCZOS4': cv2.INTER_LANCZOS4
                    }
                    self.preview_interpolation = interpolation_map.get(interpolation_name, cv2.INTER_LINEAR)
            else:
                self.preview_interval = 0.033  # 默认约30fps
                self.preview_max_width = None
                self.preview_max_height = None
                self.preview_enable_scale = True
                self.preview_interpolation = cv2.INTER_LINEAR
        except Exception as e:
            logger.warning(f"读取配置文件失败，使用默认配置: {e}")
            self.preview_interval = 0.033  # 默认约30fps
            self.preview_max_width = None
            self.preview_max_height = None
            self.preview_enable_scale = True
            self.preview_interpolation = cv2.INTER_LINEAR
        
        # 记录配置信息
        fps_info = f"约{1.0/self.preview_interval:.1f}fps"
        scale_info = ""
        if self.preview_enable_scale and self.preview_max_width:
            scale_info = f"，预览尺寸: 最大宽度{self.preview_max_width}px"
        logger.info(f"工具关键点检测应用初始化完成，输出目录: {self.output_dir}，刷新间隔: {self.preview_interval}秒（{fps_info}）{scale_info}")
    
    def connect_camera(
        self,
        camera_type: str,
        camera_index: int = 0,
        axis_ip: str = "",
        axis_username: str = "root",
        axis_password: str = "root"
    ) -> Tuple[str, str]:
        """
        连接摄像头
        
        Args:
            camera_type: 摄像头类型，"本机摄像头" 或 "Axis 摄像头"
            camera_index: 本机摄像头索引（仅本机摄像头使用）
            axis_ip: Axis 摄像头 IP 地址
            axis_username: Axis 摄像头用户名
            axis_password: Axis 摄像头密码
        
        Returns:
            (状态消息, 状态显示文本)
        """
        try:
            # 如果已连接，先断开
            if self.camera is not None and self.camera.is_connected:
                self.disconnect_camera()
            
            # 根据类型创建摄像头对象
            if camera_type == "本机摄像头":
                logger.info(f"连接本机摄像头（索引: {camera_index}）")
                self.camera = CameraCapture(
                    camera_index=camera_index,
                    camera_type="local"
                )
            elif camera_type == "Axis 摄像头":
                if not axis_ip or not axis_ip.strip():
                    error_msg = "Axis 摄像头 IP 地址不能为空"
                    logger.error(error_msg)
                    status_text = f"状态: ○ 未连接\n\n❌ {error_msg}"
                    return f"❌ {error_msg}", status_text
                
                logger.info(f"连接 Axis 摄像头（IP: {axis_ip}）")
                self.camera = CameraCapture(
                    camera_index=0,  # Axis 不使用索引
                    camera_type="axis",
                    axis_ip=axis_ip.strip(),
                    axis_username=axis_username.strip() or "root",
                    axis_password=axis_password.strip() or "root"
                )
            else:
                error_msg = f"不支持的摄像头类型: {camera_type}"
                logger.error(error_msg)
                status_text = f"状态: ○ 未连接\n\n❌ {error_msg}"
                return f"❌ {error_msg}", status_text
            
            # 连接摄像头
            success, msg = self.camera.connect()
            
            if success:
                logger.info(f"摄像头连接成功: {msg}")
                
                # 等待一小段时间，确保采集线程开始工作并获取第一帧
                # 特别是对于 Axis 摄像头，网络连接可能需要一些时间
                import time
                time.sleep(0.5)  # 等待500ms
                
                # 检查是否能获取到帧
                test_frame = self.camera.get_frame()
                if test_frame is not None:
                    camera_type = self.camera.camera_type
                    logger.info(f"摄像头连接成功，已获取到第一帧（类型: {camera_type}，尺寸: {test_frame.shape}）")
                else:
                    logger.warning("摄像头连接成功，但暂时无法获取帧，可能需要等待采集线程启动")
                
                status_text = f"状态: ● 已连接\n\n✅ {msg}"
                return f"✅ {msg}", status_text
            else:
                logger.error(f"摄像头连接失败: {msg}")
                status_text = f"状态: ○ 未连接\n\n❌ {msg}"
                return f"❌ {msg}", status_text
                
        except Exception as e:
            error_msg = f"连接摄像头时发生错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            status_text = f"状态: ○ 未连接\n\n❌ {error_msg}"
            return f"❌ {error_msg}", status_text
    
    def disconnect_camera(self) -> Tuple[str, str]:
        """
        断开摄像头连接
        
        Returns:
            (状态消息, 状态显示文本)
        """
        try:
            if self.camera is not None:
                success, msg = self.camera.disconnect()
                self.camera = None
                logger.info("摄像头已断开")
                status_text = f"状态: ○ 未连接\n\n✅ {msg}"
                return f"✅ {msg}", status_text
            else:
                status_text = "状态: ○ 未连接\n\n摄像头未连接"
                return "摄像头未连接", status_text
                
        except Exception as e:
            error_msg = f"断开摄像头时发生错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            status_text = f"状态: ○ 未连接\n\n❌ {error_msg}"
            return f"❌ {error_msg}", status_text
    
    def update_preview(self) -> Optional[np.ndarray]:
        """
        更新实时预览（单次更新，用于手动刷新）
        
        Returns:
            当前帧图像，如果未连接则返回None
        """
        try:
            if self.camera is None or not self.camera.is_connected:
                return None
            
            frame = self.camera.get_frame()
            if frame is not None:
                # 转换为RGB格式（Gradio Image组件需要RGB）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            
            return None
            
        except Exception as e:
            logger.error(f"更新预览失败: {e}")
            return None
    
    def stream_camera_frames(self):
        """
        流式生成摄像头帧（Generator 生成器函数，用于实时预览）
        
        该方法会持续生成摄像头帧，直到摄像头断开或流式传输停止。
        使用 Generator 方案，通过 yield 持续输出帧数据。
        
        Yields:
            numpy.ndarray: RGB格式的帧图像，如果未连接或出错则返回None
        """
        logger.info("Generator 流式传输启动...")
        frame_count = 0
        consecutive_none_count = 0
        max_consecutive_none = 50  # 连续50次None后记录警告（约5秒）
        initial_wait_completed = False  # 标记是否已完成初始等待
        
        try:
            # 初始等待：确保摄像头连接完成，特别是 Axis 摄像头需要网络连接时间
            if self.camera and self.camera.is_connected:
                camera_type = self.camera.camera_type if self.camera else "unknown"
                logger.info(f"等待摄像头准备就绪（类型: {camera_type}）...")
                
                # 等待采集线程启动并获取第一帧（最多等待3秒）
                max_initial_wait = 3.0
                wait_interval = 0.1
                waited = 0
                first_frame = None
                
                while waited < max_initial_wait:
                    if self.camera and self.camera.is_connected:
                        first_frame = self.camera.get_frame()
                        if first_frame is not None:
                            logger.info(f"摄像头已就绪，获取到第一帧（尺寸: {first_frame.shape}）")
                            initial_wait_completed = True
                            break
                    time.sleep(wait_interval)
                    waited += wait_interval
                
                if not initial_wait_completed:
                    logger.warning("摄像头连接成功，但初始等待期间未获取到帧，流式传输将继续尝试")
            
            # 开始持续流式传输
            while True:
                try:
                    # 检查摄像头连接状态
                    if self.camera is None or not self.camera.is_connected:
                        time.sleep(0.1)  # 等待摄像头连接
                        yield None
                        continue
                    
                    # 获取帧
                    frame = self.camera.get_frame()
                    if frame is not None:
                        # 转换为RGB格式（Gradio Image组件需要RGB）
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # 预览图像缩放优化（如果启用）
                        if self.preview_enable_scale:
                            h, w = frame_rgb.shape[:2]
                            scale_factor = 1.0
                            target_width = w
                            target_height = h
                            
                            # 根据最大宽度缩放
                            if self.preview_max_width and w > self.preview_max_width:
                                scale_factor = self.preview_max_width / w
                                target_width = self.preview_max_width
                                target_height = int(h * scale_factor)
                            
                            # 根据最大高度缩放（如果设置了）
                            if self.preview_max_height and target_height > self.preview_max_height:
                                height_scale = self.preview_max_height / target_height
                                scale_factor *= height_scale
                                target_width = int(target_width * height_scale)
                                target_height = self.preview_max_height
                            
                            # 如果需要缩放，执行缩放操作
                            if scale_factor < 1.0:
                                frame_rgb = cv2.resize(
                                    frame_rgb,
                                    (target_width, target_height),
                                    interpolation=self.preview_interpolation
                                )
                        
                        frame_count += 1
                        consecutive_none_count = 0  # 重置None计数
                        
                        # 记录日志（首次成功或每100帧）
                        if frame_count == 1:
                            camera_type = self.camera.camera_type if self.camera else "unknown"
                            preview_size = f"{frame_rgb.shape[1]}x{frame_rgb.shape[0]}"
                            logger.info(f"Generator 流式传输开始输出帧（类型: {camera_type}，预览尺寸: {preview_size}）")
                        elif frame_count % 100 == 0:
                            camera_type = self.camera.camera_type if self.camera else "unknown"
                            logger.info(f"Generator 流式传输中（{camera_type}），已传输 {frame_count} 帧")
                        
                        yield frame_rgb
                    else:
                        # 如果没有帧，返回None（Gradio会保持上一帧）
                        consecutive_none_count += 1
                        if consecutive_none_count == max_consecutive_none:
                            camera_type = self.camera.camera_type if self.camera else "unknown"
                            logger.warning(f"Generator 流式传输持续 {max_consecutive_none} 次未获取到帧（摄像头类型: {camera_type}），可能存在问题")
                            # 检查摄像头状态
                            if self.camera:
                                status = self.camera.get_status()
                                logger.warning(f"摄像头状态: {status}")
                        yield None
                    
                    # 控制帧率（使用配置文件中的刷新间隔，默认约30fps）
                    time.sleep(self.preview_interval)
                    
                except Exception as e:
                    logger.error(f"Generator 流式传输错误: {e}", exc_info=True)
                    time.sleep(0.5)  # 出错时等待更长时间
                    yield None
                    
        except GeneratorExit:
            logger.info("Generator 流式传输已停止（GeneratorExit）")
        except Exception as e:
            logger.error(f"Generator 流式传输发生严重错误: {e}", exc_info=True)
        finally:
            logger.info(f"Generator 流式传输结束，共传输 {frame_count} 帧")
    
    def toggle_auto_refresh(self, enabled: bool) -> str:
        """
        切换自动刷新状态
        
        Args:
            enabled: 是否启用自动刷新
        
        Returns:
            状态消息
        """
        try:
            self.auto_refresh_enabled = enabled
            
            if enabled:
                logger.info("自动刷新已开启")
                fps = 1.0 / self.preview_interval if self.preview_interval > 0 else 30.0
                return f"✅ 自动刷新已开启，预览将定期更新（约{fps:.0f}fps）"
            else:
                logger.info("自动刷新已关闭")
                return "ℹ️ 自动刷新已关闭，请手动点击刷新按钮"
                
        except Exception as e:
            logger.error(f"切换自动刷新状态失败: {e}")
            return f"❌ 切换失败: {str(e)}"
    
    def start_detection(
        self,
        conf: float,
        imgsz: int,
        save_raw: bool,
        save_result: bool,
        save_json: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, str]:
        """
        开始检测
        
        Args:
            conf: 置信度阈值
            imgsz: 图像尺寸
            save_raw: 是否保存原始图片
            save_result: 是否保存结果图片
            save_json: 是否保存JSON数据
            
        Returns:
            (预览图像, 结果图像, 统计信息, 状态信息)
        """
        try:
            # 检查摄像头连接
            if self.camera is None or not self.camera.is_connected:
                return None, None, "", "❌ 请先连接摄像头"
            
            # 获取当前帧
            frame = self.camera.get_frame()
            if frame is None:
                return None, None, "", "❌ 无法获取摄像头帧，请检查连接"
            
            # 保存原始图片（如果需要）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_image_path = None
            if save_raw:
                try:
                    raw_image_path = self.output_dir / "raw_images" / f"raw_{timestamp}.jpg"
                    cv2.imwrite(str(raw_image_path), frame)
                    if not raw_image_path.exists():
                        logger.warning(f"原始图片保存可能失败: {raw_image_path}")
                    else:
                        logger.info(f"原始图片已保存: {raw_image_path}")
                except Exception as e:
                    logger.error(f"保存原始图片失败: {e}")
                    # 继续执行，不因为保存失败而中断检测
            
            # 执行检测
            logger.info("开始执行检测...")
            result_dict = self.detector.detect_and_annotate(frame, conf=conf, imgsz=imgsz)
            
            if not result_dict['success']:
                return None, None, "", f"❌ 检测失败: {result_dict.get('error_message', '未知错误')}"
            
            # 获取结果
            annotated_image = result_dict['annotated_image']
            angles_results = result_dict['angles_results']
            yolo_result = result_dict['result']
            
            # 转换为RGB格式用于显示
            result_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            preview_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 生成统计信息
            stats_text = self._format_statistics(yolo_result, angles_results)
            
            # 保存结果（如果需要）
            saved_files = []
            
            if save_result:
                try:
                    result_image_path = self.output_dir / "result_images" / f"result_{timestamp}.jpg"
                    cv2.imwrite(str(result_image_path), annotated_image)
                    if result_image_path.exists():
                        saved_files.append(f"结果图: {result_image_path}")
                        logger.info(f"结果图片已保存: {result_image_path}")
                    else:
                        logger.warning(f"结果图片保存可能失败: {result_image_path}")
                except Exception as e:
                    logger.error(f"保存结果图片失败: {e}")
            
            if save_json:
                try:
                    json_data = self.detector.prepare_json_data(
                        yolo_result,
                        angles_results,
                        image_path=str(raw_image_path) if raw_image_path else "camera_frame"
                    )
                    json_path = self.output_dir / "json_results" / f"result_{timestamp}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    if json_path.exists():
                        saved_files.append(f"JSON: {json_path}")
                        logger.info(f"JSON数据已保存: {json_path}")
                    else:
                        logger.warning(f"JSON数据保存可能失败: {json_path}")
                except Exception as e:
                    logger.error(f"保存JSON数据失败: {e}")
            
            # 生成状态信息
            status_msg = "✅ 检测完成\n\n"
            if saved_files:
                status_msg += "✅ 文件已保存:\n"
                for file_info in saved_files:
                    status_msg += f"- {file_info}\n"
            else:
                status_msg += "ℹ️ 未启用自动保存"
            
            logger.info("检测完成")
            return preview_image_rgb, result_image_rgb, stats_text, status_msg
            
        except Exception as e:
            error_msg = f"检测过程中发生错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, None, "", f"❌ {error_msg}"
    
    def _format_statistics(self, result: Any, angles_results: list) -> str:
        """
        格式化统计信息
        
        Args:
            result: YOLO预测结果对象
            angles_results: 角度计算结果列表
            
        Returns:
            格式化的统计信息文本
        """
        try:
            if result is None:
                return "未检测到任何对象"
            
            stats_parts = []
            
            # 检测统计
            if result.boxes is not None:
                num_objects = len(result.boxes)
                stats_parts.append(f"### 📊 检测统计\n")
                stats_parts.append(f"检测到 **{num_objects}** 个工具对象\n\n")
                
                # 按类别统计
                if result.boxes.cls is not None:
                    boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
                    class_counts = {}
                    for cls_id in boxes_cls:
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    
                    for cls_id, count in class_counts.items():
                        class_name = self.detector._class_names.get(cls_id, f"class_{cls_id}")
                        stats_parts.append(f"- **{class_name}**: {count} 个\n")
            
            # 角度统计
            if angles_results:
                stats_parts.append(f"\n### 📐 角度统计\n\n")
                for obj_info in angles_results:
                    class_name = obj_info['class_name']
                    angles = obj_info['angles']
                    
                    # 统计有效角度
                    valid_angles = {k: v for k, v in angles.items() if isinstance(v, (int, float)) and v is not None}
                    
                    if valid_angles:
                        stats_parts.append(f"**{class_name}** (对象 {obj_info['object_id']}): {len(valid_angles)} 个有效角度\n")
                        for angle_name, angle_value in list(valid_angles.items())[:5]:  # 只显示前5个
                            stats_parts.append(f"  - {angle_name}: {angle_value:.2f}°\n")
                        
                        if len(valid_angles) > 5:
                            stats_parts.append(f"  - ... (还有 {len(valid_angles) - 5} 个角度)\n")
                        stats_parts.append("\n")
            
            return "".join(stats_parts) if stats_parts else "无统计信息"
            
        except Exception as e:
            logger.error(f"格式化统计信息失败: {e}")
            return f"统计信息生成失败: {str(e)}"
    
    def set_output_dir(self, output_dir: str) -> str:
        """
        设置输出目录
        
        Args:
            output_dir: 输出目录路径
            
        Returns:
            状态消息
        """
        try:
            self.output_dir = Path(output_dir)
            ensure_dir(self.output_dir / "raw_images")
            ensure_dir(self.output_dir / "result_images")
            ensure_dir(self.output_dir / "json_results")
            logger.info(f"输出目录已设置为: {self.output_dir}")
            return f"✅ 输出目录已设置: {self.output_dir}"
        except Exception as e:
            error_msg = f"设置输出目录失败: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def create_interface(self) -> gr.Blocks:
        """
        创建Gradio界面
        """
        try:
            with gr.Blocks(theme=gr.themes.Soft(), title="工具关键点检测系统") as demo:
                # 标题
                gr.Markdown("# 🔧 工具关键点检测系统")
                gr.Markdown("### 实时检测工具关键点并计算角度")
                
                with gr.Row():
                    # 左侧：配置区域
                    with gr.Column(scale=1):
                        # 摄像头配置
                        gr.Markdown("## 📹 摄像头配置")
                        camera_status = gr.Markdown("状态: ○ 未连接", label="摄像头状态")
                        
                        # 摄像头类型选择
                        camera_type_dropdown = gr.Dropdown(
                            choices=["本机摄像头", "Axis 摄像头"],
                            value="本机摄像头",
                            label="摄像头类型",
                            info="选择要使用的摄像头类型"
                        )
                        
                        # 本机摄像头配置（默认显示）
                        camera_index_input = gr.Number(
                            value=0,
                            label="摄像头索引",
                            info="本机摄像头索引，通常为 0",
                            visible=True,
                            precision=0
                        )
                        
                        # Axis 摄像头配置（默认隐藏）
                        axis_ip_input = gr.Textbox(
                            value="192.168.39.253",
                            label="Axis 摄像头 IP 地址",
                            placeholder="192.168.39.253",
                            visible=False
                        )
                        axis_username_input = gr.Textbox(
                            value="root",
                            label="Axis 用户名",
                            placeholder="root",
                            visible=False
                        )
                        axis_password_input = gr.Textbox(
                            value="root",
                            label="Axis 密码",
                            placeholder="root",
                            type="password",
                            visible=False
                        )
                        
                        with gr.Row():
                            connect_btn = gr.Button("连接摄像头", variant="primary")
                            disconnect_btn = gr.Button("断开连接", variant="secondary")
                        
                        camera_msg = gr.Markdown("", label="连接状态", visible=False)
                        
                        # 摄像头类型切换时的界面更新函数
                        def update_camera_config(camera_type):
                            """根据摄像头类型显示/隐藏相应配置项"""
                            if camera_type == "本机摄像头":
                                return [
                                    gr.update(visible=True),   # camera_index_input
                                    gr.update(visible=False), # axis_ip_input
                                    gr.update(visible=False), # axis_username_input
                                    gr.update(visible=False)  # axis_password_input
                                ]
                            else:  # Axis 摄像头
                                return [
                                    gr.update(visible=False), # camera_index_input
                                    gr.update(visible=True),  # axis_ip_input
                                    gr.update(visible=True),  # axis_username_input
                                    gr.update(visible=True)   # axis_password_input
                                ]
                        
                        # 绑定摄像头类型切换事件
                        camera_type_dropdown.change(
                            fn=update_camera_config,
                            inputs=[camera_type_dropdown],
                            outputs=[
                                camera_index_input,
                                axis_ip_input,
                                axis_username_input,
                                axis_password_input
                            ]
                        )
                        
                        # 预览性能优化配置
                        gr.Markdown("## ⚡ 预览性能优化")
                        preview_fps_slider = gr.Slider(
                            minimum=5,
                            maximum=60,
                            value=int(1.0 / self.preview_interval) if self.preview_interval > 0 else 20,
                            step=1,
                            label="刷新率 (fps)",
                            info=f"当前配置文件: {1.0/self.preview_interval:.1f}fps（需修改配置文件生效）"
                        )
                        preview_width_slider = gr.Slider(
                            minimum=320,
                            maximum=1920,
                            value=self.preview_max_width if self.preview_max_width else 1280,
                            step=80,
                            label="预览最大宽度 (px)",
                            info="较小的值可提高刷新速度，但会降低预览质量"
                        )
                        preview_scale_checkbox = gr.Checkbox(
                            value=self.preview_enable_scale,
                            label="启用预览缩放优化",
                            info="启用后将在传输前缩放图像，减少网络传输量"
                        )
                        apply_preview_settings_btn = gr.Button("应用预览设置", variant="secondary")
                        preview_settings_status = gr.Markdown("", visible=False)
                        
                        # 检测参数
                        gr.Markdown("## ⚙️ 检测参数")
                        conf_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.25,
                            step=0.05,
                            label="置信度阈值",
                            info="调整检测阈值（值越低检测越多）"
                        )
                        
                        imgsz_dropdown = gr.Dropdown(
                            choices=[416, 640, 1280],
                            value=640,
                            label="图像尺寸",
                            info="推荐：640"
                        )
                        
                        # 输出配置
                        gr.Markdown("## 💾 输出配置")
                        output_dir_input = gr.Textbox(
                            value=str(self.output_dir),
                            label="输出目录",
                            placeholder="outputs/tool_pose"
                        )
                        set_output_btn = gr.Button("设置输出目录", variant="secondary")
                        output_dir_status = gr.Markdown("", label="输出目录状态")
                        
                        save_raw_checkbox = gr.Checkbox(
                            value=True,
                            label="保存检测前图片"
                        )
                        save_result_checkbox = gr.Checkbox(
                            value=True,
                            label="保存检测结果图片"
                        )
                        save_json_checkbox = gr.Checkbox(
                            value=True,
                            label="保存JSON数据"
                        )
                        
                        # 操作按钮
                        gr.Markdown("## 🔍 操作")
                        detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")
                        
                        # 状态信息
                        status_info = gr.Markdown("就绪", label="状态信息")
                    
                    # 右侧：预览和结果区域
                    with gr.Column(scale=1):
                        # 实时预览
                        gr.Markdown("## 📺 实时预览")
                        # 动态计算并显示刷新率信息
                        fps = 1.0 / self.preview_interval if self.preview_interval > 0 else 30.0
                        gr.Markdown(f"💡 **提示**: 连接摄像头后将自动开始实时流式传输（约{fps:.0f}fps）")
                        with gr.Row():
                            refresh_preview_btn = gr.Button("🔄 手动刷新", variant="secondary", size="sm")
                        preview_image = gr.Image(
                            label="摄像头画面",
                            height=400,
                            type="numpy",
                            streaming=True  # 启用流式传输
                        )
                        
                        # 检测结果
                        gr.Markdown("## 📊 检测结果")
                        result_image = gr.Image(
                            label="检测结果（含关键点和角度标注）",
                            height=400,
                            type="numpy"
                        )
                        
                        # 统计信息
                        gr.Markdown("## 📈 检测统计")
                        stats_output = gr.Markdown("", label="统计信息")
                
                # 事件绑定
                # 连接摄像头
                # 使用 Generator 方案：连接成功后自动启动生成器流式传输
                def connect_camera_wrapper(camera_type, camera_index, axis_ip, axis_username, axis_password):
                    """连接摄像头包装函数，用于触发流式传输"""
                    msg, status = self.connect_camera(
                        camera_type, camera_index, axis_ip, axis_username, axis_password
                    )
                    return msg, status
                
                # 连接摄像头按钮事件
                connect_btn.click(
                    fn=connect_camera_wrapper,
                    inputs=[
                        camera_type_dropdown,
                        camera_index_input,
                        axis_ip_input,
                        axis_username_input,
                        axis_password_input
                    ],
                    outputs=[camera_msg, camera_status]
                ).then(
                    # 连接成功后，启动 Generator 流式传输
                    fn=self.stream_camera_frames,
                    inputs=[],
                    outputs=[preview_image]
                )
                
                # 断开摄像头
                disconnect_btn.click(
                    fn=self.disconnect_camera,
                    outputs=[camera_msg, camera_status]
                )
                
                # 设置输出目录
                set_output_btn.click(
                    fn=self.set_output_dir,
                    inputs=[output_dir_input],
                    outputs=[output_dir_status]
                )
                
                # 应用预览设置
                def apply_preview_settings(enable_scale, max_width):
                    """应用预览性能优化设置"""
                    try:
                        self.preview_enable_scale = enable_scale
                        self.preview_max_width = int(max_width) if max_width else None
                        fps = 1.0 / self.preview_interval if self.preview_interval > 0 else 20.0
                        status_msg = f"✅ 预览设置已应用：缩放优化={'启用' if enable_scale else '禁用'}"
                        if enable_scale and max_width:
                            status_msg += f"，最大宽度={max_width}px"
                        status_msg += f"，刷新率={fps:.1f}fps"
                        logger.info(f"预览设置已更新: enable_scale={enable_scale}, max_width={max_width}")
                        return gr.update(value=status_msg, visible=True)
                    except Exception as e:
                        error_msg = f"❌ 应用预览设置失败: {str(e)}"
                        logger.error(error_msg)
                        return gr.update(value=error_msg, visible=True)
                
                apply_preview_settings_btn.click(
                    fn=apply_preview_settings,
                    inputs=[preview_scale_checkbox, preview_width_slider],
                    outputs=[preview_settings_status]
                )
                
                # 开始检测
                detect_btn.click(
                    fn=self.start_detection,
                    inputs=[
                        conf_slider,
                        imgsz_dropdown,
                        save_raw_checkbox,
                        save_result_checkbox,
                        save_json_checkbox
                    ],
                    outputs=[preview_image, result_image, stats_output, status_info]
                )
                
                # 刷新预览按钮（手动刷新，用于单次更新）
                refresh_preview_btn.click(
                    fn=self.update_preview,
                    outputs=[preview_image]
                )
                
                # Generator 方案：流式传输摄像头画面
                # 注意：Generator 函数会在连接摄像头后通过 .then() 自动启动
                # 生成器会持续 yield 帧数据，直到摄像头断开或用户断开连接
            
            # 启用队列功能（Generator 方案需要）
            # 队列功能允许 Gradio 处理长时间的生成器函数
            demo.queue()
            
            logger.info("Gradio界面创建完成（Generator 方案，已启用队列）")
            return demo
            
        except Exception as e:
            logger.error(f"创建Gradio界面失败: {e}", exc_info=True)
            raise
    
    def __del__(self):
        """析构函数，确保摄像头资源被释放"""
        try:
            if self.camera is not None:
                self.camera.disconnect()
        except Exception:
            pass


def create_gradio_interface(config_path: Optional[str] = None, output_dir: str = "outputs/tool_pose") -> gr.Blocks:
    """
    创建Gradio界面的便捷函数
    
    Args:
        config_path: 工具关键点配置文件路径，如果为None则使用默认路径
                    (src/yolo_detector/config/tool_pose_config.yaml)
        output_dir: 输出目录路径
        
    Returns:
        Gradio界面对象
    """
    # 创建应用
    app = ToolPoseGradioApp(config_path=config_path, output_dir=output_dir)
    
    # 创建界面
    return app.create_interface()
