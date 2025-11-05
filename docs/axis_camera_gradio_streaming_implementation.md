# Axis 网络摄像头连接与 Gradio 流式输出实现指南

**创建者/修改者**: chenliang  
**修改时间**: 2025年11月02日  
**主要修改内容**: Axis 网络摄像头连接与 Gradio 流式输出完整实现方案

---

## 📋 概述

本文档详细说明如何在项目中实现 Axis 网络摄像头（MJPEG 流）的连接，并在 Gradio Web 界面中实现实时流式输出。该方案支持本机摄像头和 Axis 网络摄像头两种类型，使用统一的接口进行管理。

---

## 🏗️ 架构设计

### 核心组件

1. **CameraCapture 类** (`src/yolo_detector/core/camera_capture.py`)
   - 负责摄像头连接、帧采集、资源管理
   - 支持本机摄像头和 Axis MJPEG 摄像头
   - 后台线程持续采集帧，提供线程安全的帧缓存

2. **ToolPoseGradioApp 类** (`src/yolo_detector/ui/gradio_app.py`)
   - Gradio Web 界面应用
   - 实现流式传输生成器函数
   - 处理摄像头连接和预览显示

### 数据流

```
Axis 摄像头 (MJPEG 流)
    ↓
OpenCV VideoCapture (HTTP Basic Auth)
    ↓
后台采集线程 (持续读取帧)
    ↓
线程安全帧缓存 (latest_frame)
    ↓
Generator 生成器函数 (stream_camera_frames)
    ↓
Gradio Image 组件 (streaming=True)
    ↓
Web 浏览器实时预览
```

---

## 🔧 实现细节

### 1. Axis 摄像头连接实现

#### 1.1 URL 构建

Axis 摄像头使用标准的 MJPEG 流 URL，支持在 URL 中嵌入认证信息：

```python
# URL 格式
mjpeg_url = f"http://{username}:{password}@{ip}/axis-cgi/mjpg/video.cgi"

# 示例
mjpeg_url = "http://root:root@192.168.39.253/axis-cgi/mjpg/video.cgi"
```

#### 1.2 连接代码

```python
def _connect_axis_opencv(self) -> Tuple[bool, str]:
    """
    使用 OpenCV VideoCapture 连接 Axis MJPEG 流
    
    URL 格式: http://[username]:[password]@[ip]/axis-cgi/mjpg/video.cgi
    OpenCV 支持 HTTP Basic Auth（通过 URL 嵌入认证信息）
    """
    try:
        # 构建 MJPEG 流 URL
        mjpeg_url = f"http://{self.axis_username}:{self.axis_password}@{self.axis_ip}/axis-cgi/mjpg/video.cgi"
        
        # 创建 VideoCapture 对象
        self.cap = cv2.VideoCapture(mjpeg_url)
        
        # 检查是否成功打开
        if not self.cap.isOpened():
            return False, "无法打开 MJPEG 流，请检查 IP 地址和网络连接"
        
        # 尝试读取一帧以验证连接
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, "连接成功但无法读取帧数据，可能是认证失败或流格式不支持"
        
        # 连接成功
        self.is_connected = True
        self._start_capture_thread()  # 启动后台采集线程
        
        return True, "Axis 摄像头连接成功"
        
    except Exception as e:
        return False, f"连接异常: {str(e)}"
```

#### 1.3 后台采集线程

为了保证流畅的帧采集，使用后台线程持续读取帧：

```python
def _start_capture_thread(self):
    """启动后台采集线程"""
    if self.is_capturing:
        return
    
    self.is_capturing = True
    self.capture_thread = threading.Thread(
        target=self._capture_loop,
        daemon=True,
        name="CameraCaptureThread"
    )
    self.capture_thread.start()

def _capture_loop(self):
    """后台采集循环（在线程中运行）"""
    while self.is_capturing and self.is_connected:
        try:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.frame_timestamp = datetime.now()
                        self.frame_count += 1
                else:
                    self.error_count += 1
                    self.last_error = "读取帧失败"
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"采集帧时出错: {e}")
        
        time.sleep(0.01)  # 避免CPU占用过高
```

#### 1.4 线程安全的帧获取

```python
def get_frame(self) -> Optional[np.ndarray]:
    """
    获取最新的帧（线程安全）
    
    Returns:
        最新的帧图像（BGR格式），如果未连接或没有帧则返回None
    """
    if not self.is_connected:
        return None
    
    with self.frame_lock:
        return self.latest_frame.copy() if self.latest_frame is not None else None
```

---

### 2. Gradio 流式输出实现

#### 2.1 Generator 生成器函数

使用 Python Generator 实现持续流式输出：

```python
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
    
    try:
        # 初始等待：确保摄像头连接完成
        if self.camera and self.camera.is_connected:
            max_initial_wait = 3.0
            wait_interval = 0.1
            waited = 0
            
            while waited < max_initial_wait:
                if self.camera and self.camera.is_connected:
                    first_frame = self.camera.get_frame()
                    if first_frame is not None:
                        logger.info(f"摄像头已就绪，获取到第一帧")
                        break
                time.sleep(wait_interval)
                waited += wait_interval
        
        # 开始持续流式传输
        while True:
            try:
                # 检查摄像头连接状态
                if self.camera is None or not self.camera.is_connected:
                    time.sleep(0.1)
                    yield None
                    continue
                
                # 获取帧
                frame = self.camera.get_frame()
                if frame is not None:
                    # 转换为RGB格式（Gradio Image组件需要RGB）
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 预览图像缩放优化（如果启用）
                    if self.preview_enable_scale and self.preview_max_width:
                        h, w = frame_rgb.shape[:2]
                        if w > self.preview_max_width:
                            scale_factor = self.preview_max_width / w
                            target_width = self.preview_max_width
                            target_height = int(h * scale_factor)
                            
                            frame_rgb = cv2.resize(
                                frame_rgb,
                                (target_width, target_height),
                                interpolation=self.preview_interpolation
                            )
                    
                    frame_count += 1
                    yield frame_rgb
                else:
                    yield None
                    
            except Exception as e:
                logger.error(f"流式传输错误: {e}")
                yield None
                time.sleep(0.1)
                
    except Exception as e:
        logger.error(f"流式传输异常: {e}")
        yield None
```

#### 2.2 Gradio 界面集成

```python
# 创建 Image 组件，启用流式传输
preview_image = gr.Image(
    label="摄像头画面",
    height=400,
    type="numpy",
    streaming=True  # 启用流式传输
)

# 连接摄像头按钮事件
connect_btn.click(
    fn=connect_camera_wrapper,
    inputs=[camera_type_dropdown, camera_index_input, ...],
    outputs=[camera_msg, camera_status]
).then(
    # 连接成功后，启动 Generator 流式传输
    fn=self.stream_camera_frames,
    inputs=[],
    outputs=[preview_image]
)
```

---

## 📝 配置说明

### 配置文件位置

`src/yolo_detector/config/tool_pose_config.yaml`

### 摄像头配置

```yaml
camera:
  # 预览刷新间隔（秒）
  preview_interval: 0.05  # 约20fps
  
  # 预览图像优化配置
  preview:
    # 预览图像最大宽度（像素）
    max_width: 640  # 推荐值：640/800/1280/1920
    
    # 预览图像最大高度（像素，null表示按宽度比例自动计算）
    max_height: null
    
    # 是否启用预览缩放优化
    enable_scale: true
    
    # 图像插值方法
    interpolation: "INTER_LINEAR"  # INTER_LINEAR/INTER_AREA/INTER_NEAREST/INTER_CUBIC
```

### 使用示例

```python
from yolo_detector.core import CameraCapture

# 创建 Axis 摄像头对象
camera = CameraCapture(
    camera_type="axis",
    axis_ip="192.168.39.253",
    axis_username="root",
    axis_password="root"
)

# 连接摄像头
success, msg = camera.connect()
if success:
    print("连接成功")
    
    # 获取帧
    frame = camera.get_frame()
    if frame is not None:
        print(f"获取到帧，尺寸: {frame.shape}")
    
    # 断开连接
    camera.disconnect()
```

---

## 🎯 完整使用示例

### 示例 1: 基础连接和预览

```python
from yolo_detector.core import CameraCapture
import cv2

# 初始化 Axis 摄像头
camera = CameraCapture(
    camera_type="axis",
    axis_ip="192.168.39.253",
    axis_username="root",
    axis_password="root"
)

# 连接
success, msg = camera.connect()
print(f"连接结果: {success}, 消息: {msg}")

if success:
    # 持续获取帧并显示
    while True:
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow("Axis Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("等待帧...")
    
    camera.disconnect()
    cv2.destroyAllWindows()
```

### 示例 2: Gradio Web 界面集成

```python
from yolo_detector.ui import create_gradio_interface

# 创建 Gradio 界面
demo = create_gradio_interface(
    config_path="src/yolo_detector/config/tool_pose_config.yaml",
    output_dir="outputs/tool_pose"
)

# 启动界面
demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
```

在 Web 界面中：
1. 选择摄像头类型为 "Axis 摄像头"
2. 输入 IP 地址：`192.168.39.253`
3. 输入用户名：`root`
4. 输入密码：`root`
5. 点击 "连接摄像头"
6. 系统会自动开始流式传输预览

---

## 🔍 关键技术点

### 1. HTTP Basic Auth

OpenCV 的 `VideoCapture` 支持在 URL 中嵌入认证信息：

```python
# ✅ 支持的方式
url = f"http://username:password@ip/path"

# ❌ 不支持的方式
url = "http://ip/path"
# 然后单独设置认证头（OpenCV VideoCapture 不支持）
```

### 2. MJPEG 流格式

Axis 摄像头使用标准的 MJPEG 流格式：
- 通过 HTTP 协议传输
- 使用 `multipart/x-mixed-replace` 内容类型
- 每个 JPEG 帧之间有边界标记

### 3. 线程安全

使用 `threading.Lock` 确保帧访问的线程安全：

```python
self.frame_lock = threading.Lock()

# 写入时加锁
with self.frame_lock:
    self.latest_frame = frame

# 读取时加锁
with self.frame_lock:
    return self.latest_frame.copy()
```

### 4. Generator 流式传输

使用 Python Generator 实现持续输出：

```python
def generator_function():
    while True:
        frame = get_frame()
        yield frame  # 持续输出帧
```

Gradio 的 `streaming=True` 会自动调用生成器函数并持续更新显示。

---

## ⚙️ 性能优化

### 1. 预览图像缩放

减少网络传输和显示处理的数据量：

```python
# 原始尺寸：1920x1080 = 6,220,800 像素
# 缩放后：640x360 = 230,400 像素
# 数据量减少：约 96%
```

### 2. 刷新率控制

通过 `preview_interval` 控制刷新频率：

```yaml
preview_interval: 0.05  # 20fps
preview_interval: 0.033 # 30fps
preview_interval: 0.016 # 60fps
```

### 3. 插值算法选择

```python
# INTER_LINEAR: 速度快，质量好（推荐）
# INTER_AREA: 适合缩小图像，质量最好
# INTER_NEAREST: 最快但质量较差
# INTER_CUBIC: 质量最好但速度较慢
```

---

## 🐛 故障排除

### 问题 1: 连接失败

**症状**: `无法打开 MJPEG 流`

**可能原因**:
- IP 地址错误
- 网络不通
- 防火墙阻止
- 用户名/密码错误

**解决方案**:
```bash
# 1. 测试网络连通性
ping 192.168.39.253

# 2. 测试 HTTP 端口
telnet 192.168.39.253 80

# 3. 在浏览器中测试 URL
# http://192.168.39.253/axis-cgi/mjpg/video.cgi
```

### 问题 2: 认证失败

**症状**: `连接成功但无法读取帧数据`

**可能原因**:
- 用户名/密码错误
- 摄像头禁用匿名访问

**解决方案**:
- 确认用户名和密码正确
- 检查摄像头用户权限设置
- 尝试在浏览器中手动输入认证信息

### 问题 3: 流式传输卡顿

**症状**: 预览画面不流畅

**可能原因**:
- 网络带宽不足
- 预览尺寸过大
- 刷新率过高

**解决方案**:
```yaml
# 降低预览尺寸
preview:
  max_width: 640  # 从 1280 降低到 640

# 降低刷新率
preview_interval: 0.1  # 从 0.05 降低到 0.1 (10fps)
```

---

## 📚 相关文档

- [Axis 摄像头配置指南](axis_camera_setup.md)
- [Gradio 流式传输文档](gradio_upgrade_guide.md)
- [摄像头采集模块 API 文档](../src/yolo_detector/core/camera_capture.py)

---

## 🔄 更新日志

- **2025-11-02**: 初始版本，完整实现 Axis 摄像头连接和 Gradio 流式输出
- **2025-11-02**: 添加预览图像缩放优化功能

---

**总结**: 本方案实现了 Axis 网络摄像头的连接和 Gradio 实时流式输出，通过 OpenCV VideoCapture、后台采集线程、Generator 生成器和 Gradio streaming 功能，实现了流畅的实时预览体验。

