# 创建者/修改者: chenliang
# 修改时间：2025年11月02日 10:00
# 主要修改内容：
# 1. 创建摄像头采集模块，支持本机摄像头
# 2. 实现实时帧采集（后台线程）
# 3. 线程安全的帧缓存
# 4. 资源释放和清理
# 5. 连接/断开功能
# 6. 支持 Axis MJPEG 摄像头（HTTP/HTTPS MJPEG 流）
# 7. 简化 Axis 连接方式，仅使用 OpenCV 连接

"""
摄像头采集模块

提供摄像头连接、帧采集、资源管理等功能
支持：
- 本机摄像头（VideoCapture(0)）
- Axis MJPEG 摄像头（HTTP/HTTPS MJPEG 流）
"""

import cv2
import numpy as np
import threading
from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import requests  # 用于 test_connection 方法

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CameraCapture:
    """
    摄像头采集类
    
    功能：
    - 连接本机摄像头或 Axis MJPEG 摄像头
    - 后台线程持续采集帧
    - 线程安全的帧缓存
    - 资源自动释放和清理
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        camera_type: str = "local",
        axis_ip: Optional[str] = None,
        axis_username: str = "root",
        axis_password: str = "root"
    ):
        """
        初始化摄像头采集器
        
        Args:
            camera_index: 摄像头索引，默认为0（本机第一个摄像头，仅 local 类型使用）
            camera_type: 摄像头类型，"local" 或 "axis"
            axis_ip: Axis 摄像头 IP 地址（仅 axis 类型使用）
            axis_username: Axis 摄像头用户名，默认 "root"
            axis_password: Axis 摄像头密码，默认 "root"
        """
        self.camera_type = camera_type
        self.camera_index = camera_index
        
        # Axis 配置
        self.axis_ip = axis_ip
        self.axis_username = axis_username
        self.axis_password = axis_password
        
        # 连接方式标记（用于日志）
        self.connection_method: Optional[str] = None  # "opencv"
        
        # 视频捕获对象（OpenCV 方式）
        self.cap: Optional[cv2.VideoCapture] = None
        
        self.is_connected = False
        self.is_capturing = False
        
        # 线程和锁
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_timestamp: Optional[datetime] = None
        
        # 统计信息
        self.frame_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None
        
        logger.info(f"摄像头采集器初始化完成，类型: {camera_type}, 索引: {camera_index}")
    
    def connect(self) -> Tuple[bool, str]:
        """
        连接摄像头
        
        Returns:
            (是否成功, 状态消息)
        """
        # 如果已经连接，先断开
        if self.is_connected:
            self.disconnect()
        
        try:
            if self.camera_type == "local":
                return self._connect_local()
            elif self.camera_type == "axis":
                return self._connect_axis()
            else:
                error_msg = f"不支持的摄像头类型: {self.camera_type}"
                logger.error(error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"连接摄像头时发生错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._cleanup_capture()
            return False, error_msg
    
    def _connect_local(self) -> Tuple[bool, str]:
        """连接本机摄像头"""
        logger.info(f"正在连接本机摄像头（索引: {self.camera_index}）...")
        
        # 创建VideoCapture对象
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # 检查是否成功打开
        if not self.cap.isOpened():
            error_msg = f"无法打开摄像头（索引: {self.camera_index}），请检查摄像头是否连接"
            logger.error(error_msg)
            self._cleanup_capture()
            return False, error_msg
        
        # 设置摄像头参数（可选，提高稳定性）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置宽度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # 设置高度
        self.cap.set(cv2.CAP_PROP_FPS, 30)            # 设置帧率
        
        # 验证能否读取帧
        ret, frame = self.cap.read()
        if not ret or frame is None:
            error_msg = "摄像头连接成功，但无法读取帧数据"
            logger.error(error_msg)
            self._cleanup_capture()
            return False, error_msg
        
        self.is_connected = True
        self.connection_method = "opencv"
        logger.info(f"本机摄像头连接成功（索引: {self.camera_index}）")
        
        # 启动采集线程
        self._start_capture_thread()
        
        return True, f"本机摄像头连接成功（索引: {self.camera_index}）"
    
    def _connect_axis(self) -> Tuple[bool, str]:
        """连接 Axis MJPEG 摄像头（使用 OpenCV 方式）"""
        if not self.axis_ip:
            error_msg = "Axis 摄像头 IP 地址未设置"
            logger.error(error_msg)
            return False, error_msg
        
        logger.info(f"正在连接 Axis 摄像头（IP: {self.axis_ip}）...")
        
        # 使用 OpenCV 方式连接
        success, msg = self._connect_axis_opencv()
        if success:
            self.connection_method = "opencv"
            logger.info(f"Axis 摄像头连接成功: {msg}")
            return True, f"Axis 摄像头连接成功: {msg}"
        
        error_msg = f"Axis 摄像头连接失败: {msg}"
        logger.error(error_msg)
        return False, error_msg
    
    def _connect_axis_opencv(self) -> Tuple[bool, str]:
        """
        使用 OpenCV VideoCapture 连接 Axis MJPEG 流
        
        URL 格式: http://[username]:[password]@[ip]/axis-cgi/mjpg/video.cgi
        OpenCV 支持 HTTP Basic Auth（通过 URL 嵌入认证信息）
        """
        try:
            # 构建 MJPEG 流 URL（标准 Axis 格式）
            # 使用 URL 嵌入认证信息（OpenCV 支持 HTTP Basic Auth）
            mjpeg_url = f"http://{self.axis_username}:{self.axis_password}@{self.axis_ip}/axis-cgi/mjpg/video.cgi"
            
            # 隐藏密码在日志中
            safe_url = mjpeg_url.replace(f"{self.axis_password}", "***").replace(f"{self.axis_username}:***@", "***@")
            logger.info(f"正在连接 Axis MJPEG 流: {safe_url}")
            
            # 创建 VideoCapture 对象
            self.cap = cv2.VideoCapture(mjpeg_url)
            
            # 检查是否成功打开
            if not self.cap.isOpened():
                error_msg = "无法打开 MJPEG 流，请检查 IP 地址和网络连接"
                logger.error(error_msg)
                self._cleanup_capture()
                return False, error_msg
            
            # 尝试读取一帧以验证连接
            ret, frame = self.cap.read()
            if not ret or frame is None:
                error_msg = "连接成功但无法读取帧数据，可能是认证失败或流格式不支持"
                logger.error(error_msg)
                self._cleanup_capture()
                return False, error_msg
            
            # 连接成功
            logger.info(f"Axis 摄像头连接成功")
            logger.info(f"图像尺寸: {frame.shape[1]}x{frame.shape[0]}, 通道数: {frame.shape[2] if len(frame.shape) > 2 else 1}")
            
            self.is_connected = True
            
            # 启动采集线程
            self._start_capture_thread()
            
            return True, "Axis 摄像头连接成功"
            
        except Exception as e:
            error_msg = f"连接 Axis 摄像头时发生异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._cleanup_capture()
            return False, error_msg
    
    
    def disconnect(self) -> Tuple[bool, str]:
        """
        断开摄像头连接
        
        Returns:
            (是否成功, 状态消息)
        """
        try:
            logger.info("正在断开摄像头连接...")
            
            # 停止采集线程
            self._stop_capture_thread()
            
            # 释放资源
            self._cleanup_capture()
            
            self.is_connected = False
            self.connection_method = None
            logger.info("摄像头已断开连接")
            
            return True, "摄像头已断开连接"
            
        except Exception as e:
            error_msg = f"断开摄像头连接时发生错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        获取最新的帧
        
        Returns:
            最新的帧图像（BGR格式），如果未连接或没有帧则返回None
        """
        if not self.is_connected:
            return None
        
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """
        获取最新的帧和时间戳
        
        Returns:
            (帧图像, 时间戳)，如果未连接或没有帧则返回(None, None)
        """
        if not self.is_connected:
            return None, None
        
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.frame_timestamp
            return None, None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        手动捕获一帧（同步方式）
        
        注意：这个方法会阻塞，不推荐在实时预览中使用
        主要用于单次抓取或测试
        
        Returns:
            捕获的帧图像，如果失败则返回None
        """
        if not self.is_connected or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            return None
        except Exception as e:
            logger.error(f"捕获帧失败: {e}")
            return None
    
    def is_frame_available(self) -> bool:
        """
        检查是否有可用的帧
        
        Returns:
            如果有可用帧返回True，否则返回False
        """
        if not self.is_connected:
            return False
        
        with self.frame_lock:
            return self.latest_frame is not None
    
    def get_status(self) -> dict:
        """
        获取摄像头状态信息

        Returns:
            包含状态信息的字典
        """
        status = {
            "is_connected": self.is_connected,
            "is_capturing": self.is_capturing,
            "camera_type": self.camera_type,
            "camera_index": self.camera_index,
            "frame_count": self.frame_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "has_frame": self.is_frame_available()
        }

        # 添加连接方式信息
        if self.connection_method:
            status["connection_method"] = self.connection_method

        # 添加 Axis 配置信息（如果使用）
        if self.camera_type == "axis" and self.axis_ip:
            status["axis_ip"] = self.axis_ip

        return status

    def test_connection(self) -> dict:
        """
        测试摄像头连接（不实际建立连接，仅测试网络连通性）

        Returns:
            包含测试结果的字典
        """
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "camera_type": self.camera_type,
            "tests": []
        }

        if self.camera_type == "local":
            # 测试本机摄像头
            try:
                cap = cv2.VideoCapture(self.camera_index)
                if cap.isOpened():
                    test_results["tests"].append({
                        "name": "本地摄像头访问",
                        "status": "SUCCESS",
                        "message": f"摄像头索引 {self.camera_index} 可访问"
                    })
                    cap.release()
                else:
                    test_results["tests"].append({
                        "name": "本地摄像头访问",
                        "status": "FAILED",
                        "message": f"无法打开摄像头索引 {self.camera_index}"
                    })
            except Exception as e:
                test_results["tests"].append({
                    "name": "本地摄像头访问",
                    "status": "ERROR",
                    "message": f"测试时发生错误: {str(e)}"
                })

        elif self.camera_type == "axis":
            # 测试 Axis 摄像头
            if not self.axis_ip:
                test_results["tests"].append({
                    "name": "IP 地址检查",
                    "status": "ERROR",
                    "message": "Axis IP 地址未设置"
                })
                return test_results

            test_results["axis_ip"] = self.axis_ip

            # 测试 1: 网络连通性
            import socket
            try:
                sock = socket.create_connection((self.axis_ip, 80), timeout=5)
                sock.close()
                test_results["tests"].append({
                    "name": "网络连通性",
                    "status": "SUCCESS",
                    "message": f"可以连接到 {self.axis_ip}:80"
                })
            except Exception as e:
                test_results["tests"].append({
                    "name": "网络连通性",
                    "status": "FAILED",
                    "message": f"无法连接到 {self.axis_ip}:80 - {str(e)}"
                })

            # 测试 2: HTTP 访问
            try:
                response = requests.get(
                    f"http://{self.axis_ip}",
                    timeout=5,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                test_results["tests"].append({
                    "name": "HTTP 访问",
                    "status": "SUCCESS",
                    "message": f"HTTP 响应码: {response.status_code}",
                    "details": {
                        "server": response.headers.get('Server', 'Unknown'),
                        "content_type": response.headers.get('Content-Type', 'Unknown')
                    }
                })
            except Exception as e:
                test_results["tests"].append({
                    "name": "HTTP 访问",
                    "status": "FAILED",
                    "message": f"HTTP 请求失败: {str(e)}"
                })

            # 测试 3: MJPEG 端点
            mjpeg_urls = [
                f"http://{self.axis_ip}/axis-cgi/mjpg/video.cgi",
                f"http://{self.axis_ip}/mjpg/video.mjpg",
            ]

            for url in mjpeg_urls:
                try:
                    safe_url = url.replace(f"http://{self.axis_ip}", "http://***.***.***.***")
                    response = requests.get(
                        url,
                        timeout=10,
                        headers={'User-Agent': 'Axis Camera Test'},
                        stream=True
                    )

                    # 检查响应
                    if response.status_code == 200:
                        content_type = response.headers.get('Content-Type', '').lower()
                        if 'jpeg' in content_type or 'multipart' in content_type:
                            test_results["tests"].append({
                                "name": f"MJPEG 端点: {safe_url}",
                                "status": "SUCCESS",
                                "message": f"MJPEG 流可用 (Content-Type: {response.headers.get('Content-Type')})"
                            })
                        else:
                            test_results["tests"].append({
                                "name": f"MJPEG 端点: {safe_url}",
                                "status": "WARNING",
                                "message": f"端点存在但 Content-Type 不正确: {response.headers.get('Content-Type')}"
                            })
                    else:
                        test_results["tests"].append({
                            "name": f"MJPEG 端点: {safe_url}",
                            "status": "FAILED",
                            "message": f"HTTP 错误 {response.status_code}"
                        })

                    response.close()
                    break  # 如果成功了，就不需要测试其他URL

                except Exception as e:
                    continue  # 尝试下一个URL

            # 测试 4: 认证测试（可选，不显示密码）
            test_results["tests"].append({
                "name": "认证配置",
                "status": "INFO",
                "message": f"用户名: {self.axis_username}, 密码: {'*' * len(self.axis_password)}"
            })

        # 计算总体状态
        failed_tests = [t for t in test_results["tests"] if t["status"] == "FAILED"]
        warning_tests = [t for t in test_results["tests"] if t["status"] == "WARNING"]

        if len(failed_tests) == 0 and len(warning_tests) == 0:
            test_results["overall_status"] = "SUCCESS"
            test_results["overall_message"] = "所有测试通过"
        elif len(failed_tests) > 0:
            test_results["overall_status"] = "FAILED"
            test_results["overall_message"] = f"{len(failed_tests)} 项测试失败"
        else:
            test_results["overall_status"] = "WARNING"
            test_results["overall_message"] = f"测试通过但有 {len(warning_tests)} 项警告"

        return test_results
    
    def _start_capture_thread(self):
        """启动采集线程"""
        if self.is_capturing:
            return
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,  # 设置为守护线程，主程序退出时自动结束
            name="CameraCaptureThread"
        )
        self.capture_thread.start()
        logger.info("摄像头采集线程已启动")
    
    def _stop_capture_thread(self):
        """停止采集线程"""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        
        # 等待线程结束（最多等待2秒）
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                logger.warning("采集线程未在预期时间内结束")
            else:
                logger.info("摄像头采集线程已停止")
        
        self.capture_thread = None
    
    def _capture_loop(self):
        """
        采集循环（在线程中运行，用于 OpenCV 方式）
        
        持续从摄像头读取帧并更新缓存
        """
        logger.info("开始摄像头采集循环（OpenCV方式）")
        consecutive_errors = 0
        max_consecutive_errors = 10  # 连续错误最大次数
        
        while self.is_capturing and self.is_connected:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.warning("摄像头对象无效，停止采集")
                    break
                
                # 读取帧
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # 更新帧缓存（线程安全）
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.frame_timestamp = datetime.now()
                    
                    self.frame_count += 1
                    consecutive_errors = 0  # 重置错误计数
                    
                    # 每100帧记录一次日志（避免日志过多）
                    if self.frame_count % 100 == 0:
                        logger.debug(f"已采集 {self.frame_count} 帧")
                
                else:
                    consecutive_errors += 1
                    self.error_count += 1
                    self.last_error = "读取帧失败"
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"连续 {max_consecutive_errors} 次读取帧失败，停止采集")
                        self.is_connected = False
                        break
                
                # 控制帧率，避免CPU占用过高
                # OpenCV的read()已经有一定的延迟，这里可以适当调整
                import time
                time.sleep(0.01)  # 约100fps的循环频率
                
            except Exception as e:
                consecutive_errors += 1
                self.error_count += 1
                self.last_error = str(e)
                logger.error(f"采集帧时发生错误: {e}", exc_info=True)
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"连续 {max_consecutive_errors} 次错误，停止采集")
                    self.is_connected = False
                    break
                
                import time
                time.sleep(0.1)  # 出错时等待更长时间
        
        logger.info(f"摄像头采集循环结束，总共采集 {self.frame_count} 帧")
        self.is_capturing = False
    
    
    def _cleanup_capture(self):
        """
        清理摄像头资源（OpenCV 方式）
        """
        if self.cap is not None:
            try:
                self.cap.release()
                logger.debug("摄像头资源已释放")
            except Exception as e:
                logger.error(f"释放摄像头资源时发生错误: {e}")
            finally:
                self.cap = None
        
        # 清空帧缓存
        with self.frame_lock:
            self.latest_frame = None
            self.frame_timestamp = None
    
    
    def __enter__(self):
        """上下文管理器入口"""
        success, msg = self.connect()
        if not success:
            raise RuntimeError(f"无法连接摄像头: {msg}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
        return False
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:
            pass  # 忽略析构函数中的错误

