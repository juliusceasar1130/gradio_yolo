# 工具关键点检测系统

**创建者/修改者**: chenliang  
**修改时间**: 2025年11月02日 10:00  
**主要修改内容**: 
1. 重构为工具关键点检测系统，更新完整的架构和使用文档
2. 配置文件迁移：将 `tool_pose/demo/angle_config.yaml` 迁移到 `src/yolo_detector/config/tool_pose_config.yaml`
3. 合并所有配置参数：路径、预测、输出、类别、关键点、角度、可视化、摄像头、日志等
4. **2025年11月02日 10:00** - 新增 Axis MJPEG 摄像头支持
   - 扩展 `CameraCapture` 类支持 Axis MJPEG 流（HTTP/HTTPS）
   - 使用 OpenCV VideoCapture 连接（支持 HTTP Basic Auth）
   - 更新 Gradio 界面，添加摄像头类型选择和 Axis 配置
5. **2025年11月02日 10:00** - 实现 Gradio 流式传输实时预览
   - 使用 `streaming=True` + Generator 生成器函数实现实时摄像头预览
   - 自动持续更新预览画面（可配置刷新率，默认约20fps）
   - 简化代码，移除复杂的 JavaScript 自动刷新逻辑
   - 保留手动刷新按钮作为备选方案
6. **2025年11月02日 10:00** - 实现 Axis 网络摄像头支持和预览优化
   - 支持 Axis MJPEG 摄像头连接（HTTP Basic Auth）
   - 使用 OpenCV VideoCapture 连接网络摄像头流
   - 后台线程持续采集帧，提供线程安全的帧缓存
   - 实现预览图像缩放优化，减少网络传输数据量（可配置预览尺寸）
   - 详细实现文档：`docs/axis_camera_gradio_streaming_implementation.md`

基于YOLO11 Pose的工具关键点检测系统，提供Web界面和实时检测功能。支持摄像头实时采集、工具关键点检测、角度计算和结果保存。

## 🚀 项目特性

- **实时检测**: 支持本机摄像头和 Axis MJPEG 摄像头实时采集和预览
- **关键点检测**: 基于YOLO11 Pose的工具关键点检测
- **角度计算**: 自动计算工具关键点之间的角度
- **结果可视化**: 实时显示检测结果，包含关键点和角度标注
- **自动保存**: 支持保存原始图片、结果图片和JSON数据
- **模块化设计**: 清晰的代码结构，易于维护和扩展
- **完善的日志系统**: 详细的运行日志和错误追踪

## 📁 项目结构

```
gradio_tool/
├── src/yolo_detector/           # 主要源代码
│   ├── config/                  # 配置管理
│   │   ├── settings.py         # 配置管理模块
│   │   └── tool_pose_config.yaml  # 工具关键点配置文件（已迁移）
│   ├── core/                    # 核心功能模块
│   │   ├── camera_capture.py    # 摄像头采集模块
│   │   ├── tool_pose_model_loader.py    # 工具关键点模型加载器（已迁移）
│   │   ├── tool_pose_predictor.py        # 工具关键点预测器（已迁移）
│   │   └── ...                  # 其他核心模块（历史遗留，当前未使用）
│   ├── utils/                   # 工具函数
│   │   ├── tool_pose_utils.py  # 工具关键点检测封装模块
│   │   ├── tool_pose_angle_calculator.py # 角度计算工具（已迁移）
│   │   ├── tool_pose_json_utils.py       # JSON工具（已迁移）
│   │   ├── image_utils.py      # 图像处理工具
│   │   ├── file_utils.py       # 文件操作工具
│   │   └── logger.py           # 日志系统
│   └── ui/                      # 用户界面
│       └── gradio_app.py       # Gradio Web界面
├── tool_pose/                   # 工具检测相关
│   ├── demo/                    # 检测配置和历史实现（已迁移）
│   │   └── ...                  # 其他文件（训练、测试等用途）
│   └── tools/                   # 训练相关工具
├── configs/                     # 配置文件目录（历史遗留）
├── outputs/                     # 输出结果目录
│   └── tool_pose/              # 工具检测结果
│       ├── raw_images/         # 检测前原始图片
│       ├── result_images/      # 检测结果图片（带标注）
│       └── json_results/       # JSON数据
├── logs/                        # 日志文件
├── main.py                      # 主入口文件
└── README.md                    # 本文档
```

## 🛠️ 安装和配置

### 1. 环境要求

- Python 3.8+
- PyTorch
- Ultralytics YOLO (YOLO11)
- Gradio 5.31.0
- OpenCV
- PIL/Pillow

### 2. 安装依赖

```bash
# 激活conda环境（如果使用）
conda activate gradioflask

# 安装依赖
pip install -r requirements.txt
```

主要依赖包：
- `gradio` (5.31.0)
- `gradio_client` (1.10.1)
- `ultralytics`
- `opencv-python`
- `pillow`
- `pyyaml`
- `numpy`

### 3. 配置模型路径

编辑 `src/yolo_detector/config/tool_pose_config.yaml` 文件，设置模型路径和检测参数：

```yaml
paths:
  model: "path/to/your/model.pt"  # 修改为您的模型路径
  output: "outputs/tool_pose"     # 输出目录

predict:
  conf: 0.25      # 置信度阈值
  iou: 0.45       # IoU阈值
  imgsz: 640      # 图像尺寸
  device: ""      # 设备选择（空字符串表示自动选择）

names:
  0: tool1        # 类别名称映射
  1: tool2

kpt_names:
  0: [t1_1, t1_2, ...]  # 关键点名称
  1: [t2_1, t2_2, ...]

angles:
  tool1:
    angle1: [t1_1, t1_2, t1_3]  # 角度定义
    ...
```

详细配置说明请参考 `src/yolo_detector/config/tool_pose_config.yaml` 文件中的注释。

**注意**: 配置文件已从 `tool_pose/demo/angle_config.yaml` 迁移到 `src/yolo_detector/config/tool_pose_config.yaml`，请使用新的配置文件路径。

## 🎯 使用方法

### 启动Web界面

```bash
# 启动Web界面（默认配置）
python main.py web

# 启动Web界面（自定义端口）
python main.py web --port 7862

# 启动Web界面（创建公共链接）
python main.py web --share

# 启动Web界面（启用调试模式）
python main.py web --debug

# 启动Web界面（指定配置文件）
python main.py web --config src/yolo_detector/config/tool_pose_config.yaml

# 启动Web界面（指定输出目录）
python main.py web --output-dir outputs/tool_pose
```

### Web界面使用流程

1. **连接摄像头**
   - 点击"连接摄像头"按钮
   - 系统会自动连接本机摄像头（VideoCapture(0)）
   - 连接成功后显示摄像头状态

2. **实时预览**
   - 摄像头连接后，实时预览区域会显示摄像头画面
   - 可以手动点击"刷新预览"按钮更新画面
   - 可以勾选"自动刷新"实现定期自动更新

3. **配置检测参数**
   - **置信度阈值**: 调整检测阈值（值越低检测越多）
   - **图像尺寸**: 选择检测时的图像尺寸（推荐：640）

4. **配置输出选项**
   - **输出目录**: 设置结果保存路径（默认：outputs/tool_pose）
   - **保存检测前图片**: 是否保存检测前的原始图片
   - **保存检测结果图片**: 是否保存带标注的结果图片
   - **保存JSON数据**: 是否保存JSON格式的检测数据

5. **开始检测**
   - 点击"🔍 开始检测"按钮
   - 系统会：
     - 自动保存检测前图片（如果启用）
     - 执行关键点检测和角度计算
     - 显示检测结果（含关键点和角度标注）
     - 自动保存结果图片和JSON数据（如果启用）

6. **查看结果**
   - 右侧"检测结果"区域显示标注后的图像
   - "检测统计"区域显示检测到的对象数量和角度信息
   - "状态信息"区域显示检测状态和保存的文件路径

### 输出文件结构

检测完成后，会在输出目录生成以下文件：

```
outputs/tool_pose/
├── raw_images/
│   └── raw_YYYYMMDD_HHMMSS.jpg      # 检测前原始图片
├── result_images/
│   └── result_YYYYMMDD_HHMMSS.jpg   # 检测结果图片（带标注）
└── json_results/
    └── result_YYYYMMDD_HHMMSS.json  # JSON数据（包含检测结果和角度信息）
```

JSON数据格式包含：
- 图像信息（路径、尺寸、时间戳）
- 检测结果（对象ID、类别、置信度、边界框）
- 关键点坐标（每个关键点的x、y坐标和可见性）
- 角度计算结果（每个定义角度的度数）

## 📊 功能模块

### 1. 摄像头采集模块 (`src/yolo_detector/core/camera_capture.py`)

**CameraCapture类** - 摄像头采集管理器

**当前实现**:
- ✅ 支持本机摄像头（VideoCapture(0)）
- ✅ 支持 Axis MJPEG 摄像头（HTTP/HTTPS MJPEG 流）
- ✅ 使用 OpenCV VideoCapture 连接（支持 HTTP Basic Auth）
- ✅ 后台线程实时采集帧，提供线程安全的帧缓存
- ✅ 自动资源释放和清理
- ✅ 支持预览图像缩放优化，减少网络传输数据量

**详细实现文档**: 请参考 `docs/axis_camera_gradio_streaming_implementation.md`

**主要方法**:
- `__init__(camera_index=0, camera_type="local", axis_ip=None, axis_username="root", axis_password="root")` - 初始化摄像头采集器
  - `camera_type`: 摄像头类型，"local" 或 "axis"
  - `axis_ip`: Axis 摄像头 IP 地址（仅 axis 类型使用）
  - `axis_username`: Axis 摄像头用户名，默认 "root"
  - `axis_password`: Axis 摄像头密码，默认 "root"
- `connect() -> Tuple[bool, str]` - 连接摄像头
- `disconnect() -> Tuple[bool, str]` - 断开连接
- `get_frame() -> Optional[np.ndarray]` - 获取最新帧（线程安全）
- `get_status() -> dict` - 获取状态信息（包含连接方式、摄像头类型等）

**内部方法**（用于 Axis 摄像头）:
- `_connect_local() -> Tuple[bool, str]` - 连接本机摄像头
- `_connect_axis() -> Tuple[bool, str]` - 连接 Axis 摄像头（使用 OpenCV 方式）
- `_connect_axis_opencv() -> Tuple[bool, str]` - 使用 OpenCV VideoCapture 连接 Axis MJPEG 流

**使用示例**:

```python
from yolo_detector.core import CameraCapture

# 1. 本机摄像头
camera = CameraCapture(camera_index=0, camera_type="local")
success, msg = camera.connect()

# 2. Axis 摄像头
camera = CameraCapture(
    camera_type="axis",
    axis_ip="192.168.39.253",
    axis_username="root",
    axis_password="root"
)
success, msg = camera.connect()

if success:
    # 获取帧
    frame = camera.get_frame()
    # 处理帧...
    
    # 断开连接
    camera.disconnect()

# 使用上下文管理器（推荐）
with CameraCapture(camera_type="axis", axis_ip="192.168.39.253") as camera:
    frame = camera.get_frame()
    # 处理帧...
```

### 2. Gradio 界面模块 (`src/yolo_detector/ui/gradio_app.py`)

**ToolPoseGradioApp类** - Gradio Web界面应用

**当前实现**:
- ✅ 支持本机摄像头和 Axis 摄像头连接
- ✅ 实时流式传输摄像头画面（使用 Gradio 流式传输功能）
- ✅ 工具关键点检测和结果可视化
- ✅ 角度计算和统计信息显示
- ✅ 结果保存（原始图、结果图、JSON）
- ✅ 检测参数配置（置信度、图像尺寸等）

**主要方法**:
- `__init__(config_path=None, output_dir="outputs/tool_pose")` - 初始化应用
- `connect_camera(camera_type, camera_index=0, axis_ip="", axis_username="root", axis_password="root")` - 连接摄像头
- `disconnect_camera()` - 断开摄像头连接
- `update_preview() -> Optional[np.ndarray]` - 更新预览（单次更新，用于手动刷新）
- `stream_camera_frames()` - 流式生成摄像头帧（生成器函数，用于实时预览）
  - 持续生成摄像头帧，直到摄像头断开或流式传输停止
  - 可配置刷新间隔（默认约20fps，0.05秒间隔）
  - 包含错误处理和资源管理
  - 支持预览图像缩放优化（可配置预览最大宽度）
- `start_detection(conf, imgsz, save_raw, save_result, save_json)` - 开始检测
- `set_output_dir(output_dir)` - 设置输出目录

**流式传输功能**:
- 使用 Gradio 的 `streaming=True` 和 Generator 生成器函数实现实时预览
- 连接摄像头后自动开始流式传输
- 自动处理摄像头断开和错误情况
- 帧率控制：可配置刷新间隔（默认约20fps，0.05秒间隔）
- 预览图像缩放优化：可配置预览最大宽度，减少网络传输数据量

**使用示例**:

```python
from yolo_detector.ui import create_gradio_interface

# 创建界面（使用默认配置）
demo = create_gradio_interface()

# 使用自定义配置和输出目录
demo = create_gradio_interface(
    config_path="path/to/config.yaml",
    output_dir="custom_outputs"
)

# 启动界面
demo.launch(server_name="0.0.0.0", server_port=7861)
```

### 3. 工具关键点检测模块

#### 3.1 ToolPoseDetector (`src/yolo_detector/utils/tool_pose_utils.py`)

**ToolPoseDetector类** - 工具关键点检测器

封装工具关键点检测逻辑，提供统一的接口。所有相关模块已迁移到 `src/yolo_detector` 目录。

#### 2.2 工具关键点核心模块

- **ToolPoseModelLoader** (`src/yolo_detector/core/tool_pose_model_loader.py`) - 模型加载器（单例模式）
- **ToolPosePredictor** (`src/yolo_detector/core/tool_pose_predictor.py`) - 预测执行器
- **角度计算工具** (`src/yolo_detector/utils/tool_pose_angle_calculator.py`) - 关键点角度计算和标注
- **JSON工具** (`src/yolo_detector/utils/tool_pose_json_utils.py`) - JSON序列化和结果保存

**主要方法**:

- `__init__(config_path=None)` - 初始化检测器
  - `config_path`: 工具关键点配置文件路径（默认：src/yolo_detector/config/tool_pose_config.yaml）
  
- `load_model() -> bool` - 加载模型（延迟加载）
  - 返回: 是否成功
  - 功能: 加载YOLO Pose模型，支持模型缓存
  
- `detect(image, conf=None, iou=None, imgsz=None) -> Optional[Any]` - 执行关键点检测
  - `image`: 输入图像（numpy数组，BGR或RGB格式）
  - `conf`: 置信度阈值（可选）
  - `iou`: IoU阈值（可选）
  - `imgsz`: 图像尺寸（可选）
  - 返回: YOLO预测结果对象
  
- `calculate_angles(result) -> List[Dict[str, Any]]` - 计算角度
  - `result`: YOLO预测结果对象
  - 返回: 角度计算结果列表
  
- `annotate_image(image, result, angles_results) -> np.ndarray` - 在图像上标注检测结果和角度
  - `image`: 原始图像（BGR格式）
  - `result`: YOLO预测结果对象
  - `angles_results`: 角度计算结果列表
  - 返回: 标注后的图像（BGR格式）
  
- `detect_and_annotate(image, conf=None, iou=None, imgsz=None) -> Dict[str, Any]` - 一站式检测和标注接口
  - 返回: 包含检测结果的字典（annotated_image, result, angles_results等）
  
- `prepare_json_data(result, angles_results, image_path=None) -> Dict[str, Any]` - 准备JSON数据（用于保存）
  - 返回: JSON数据字典
  
- `get_status() -> Dict[str, Any]` - 获取检测器状态信息
  - 返回: 状态信息字典

**使用示例**:
```python
from yolo_detector.utils import ToolPoseDetector
import cv2

# 创建检测器（使用默认配置文件）
detector = ToolPoseDetector()
# 或指定自定义配置文件
# detector = ToolPoseDetector(config_path="path/to/custom_config.yaml")

# 加载模型（延迟加载，第一次检测时自动加载）
detector.load_model()

# 读取图像
image = cv2.imread("test.jpg")

# 执行检测和标注（一站式）
result = detector.detect_and_annotate(image, conf=0.25, imgsz=640)

if result['success']:
    annotated_image = result['annotated_image']
    angles_results = result['angles_results']
    
    # 保存标注后的图像
    cv2.imwrite("result.jpg", annotated_image)
    
    # 准备JSON数据
    json_data = detector.prepare_json_data(
        result['result'],
        angles_results,
        image_path="test.jpg"
    )
```

### 3. Gradio界面模块 (`src/yolo_detector/ui/gradio_app.py`)

**ToolPoseGradioApp类** - 工具关键点检测Gradio应用

提供完整的Web界面，包括：
- 摄像头连接和实时预览
- 检测参数配置
- 输出配置
- 检测执行和结果显示
- 结果保存

**主要方法**:
- `__init__(config_path, output_dir)` - 初始化应用
- `connect_camera()` - 连接摄像头
- `disconnect_camera()` - 断开摄像头
- `update_preview()` - 更新实时预览
- `start_detection(...)` - 执行检测
- `create_interface()` - 创建Gradio界面

**使用方式**:
```python
from yolo_detector.ui import create_gradio_interface

# 创建界面（使用默认配置文件）
demo = create_gradio_interface(
    output_dir="outputs/tool_pose"
)
# 或指定自定义配置文件
# demo = create_gradio_interface(
#     config_path="path/to/custom_config.yaml",
#     output_dir="outputs/tool_pose"
# )

# 启动界面
demo.launch(server_port=7861)
```

### 4. 工具函数模块 (`src/yolo_detector/utils/`)

- **图像工具** (`image_utils.py`): 图像加载、处理、验证
- **文件工具** (`file_utils.py`): 文件操作、路径处理
- **日志系统** (`logger.py`): 统一的日志记录
- **异常处理** (`exceptions.py`): 完善的错误处理机制

## 🔧 开发指南

### 核心模块函数清单

#### Gradio 界面模块 (`src/yolo_detector/ui/gradio_app.py`)

**ToolPoseGradioApp 类**:
- `connect_camera()` - 连接摄像头（支持本机摄像头和 Axis 摄像头）
- `disconnect_camera()` - 断开摄像头连接
- `update_preview()` - 更新预览（单次更新，用于手动刷新）
- `stream_camera_frames()` - 流式生成摄像头帧（生成器函数，用于实时预览）**【新增】**
- `start_detection()` - 开始检测
- `set_output_dir()` - 设置输出目录
- `toggle_auto_refresh()` - 切换自动刷新状态（保留用于兼容性）

详细的核心模块函数清单请参考 README.md 中的"功能模块"部分。

### 扩展开发

#### 添加新的摄像头类型

1. 在 `camera_capture.py` 中添加新的摄像头类
2. 实现 `connect()`, `disconnect()`, `get_frame()` 等方法
3. 在 `gradio_app.py` 中添加对应的配置界面

#### 添加新的检测功能

1. 在 `tool_pose_utils.py` 中添加新的方法
2. 在 `src/yolo_detector/core/` 或 `src/yolo_detector/utils/` 中添加相关模块
3. 更新 `gradio_app.py` 添加对应的界面组件

#### 模块组织结构说明

所有工具关键点检测相关的模块已统一迁移到 `src/yolo_detector` 目录：

- **核心模块** (`src/yolo_detector/core/`): 模型加载、预测执行等
- **工具模块** (`src/yolo_detector/utils/`): 角度计算、JSON处理、检测封装等
- **配置文件**: 统一位于 `src/yolo_detector/config/tool_pose_config.yaml`

#### 自定义配置

可以通过以下方式自定义配置：

```python
from yolo_detector.utils import ToolPoseDetector

# 使用自定义配置文件
detector = ToolPoseDetector(config_path="path/to/custom_config.yaml")

# 在检测时动态调整参数
result = detector.detect_and_annotate(
    image,
    conf=0.3,      # 自定义置信度（覆盖配置文件中的值）
    imgsz=1280     # 自定义图像尺寸（覆盖配置文件中的值）
)

# 查看配置状态
status = detector.get_status()
print(f"模型路径: {status['model_path']}")
print(f"配置文件: {status['config_path']}")
```

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
python -m pytest

# 运行特定测试
python -m pytest tests/test_config.py

# 生成覆盖率报告
python -m pytest --cov=src/yolo_detector --cov-report=html
```

## 📈 性能优化

### 1. 模型优化

- 使用适当的模型大小（YOLO11n/s/m/l/x）
- 启用GPU加速（如果可用）
- 合理设置置信度阈值和图像尺寸

### 2. 摄像头采集优化

- 当前使用后台线程采集，避免阻塞主线程
- 帧缓存使用锁机制保证线程安全
- 支持帧率控制，避免CPU占用过高

### 3. 内存管理

- 模型延迟加载（第一次检测时加载）
- 及时释放不需要的资源
- 使用适当的图像尺寸限制

## 🐛 故障排除

### 常见问题

1. **摄像头连接失败**
   - 检查摄像头是否正确连接
   - 确认摄像头未被其他程序占用
   - 查看日志文件获取详细错误信息

2. **模型加载失败**
   - 检查模型文件路径是否正确（在 `src/yolo_detector/config/tool_pose_config.yaml` 中配置）
   - 确认模型文件格式是否支持（.pt文件）
   - 查看日志文件获取详细错误信息

3. **检测失败或结果不准确**
   - 调整置信度阈值（降低阈值可能检测更多，但也可能包含误检）
   - 尝试不同的图像尺寸（640/1280）
   - 检查关键点定义是否正确（在 `src/yolo_detector/config/tool_pose_config.yaml` 中配置）

4. **Web界面无法访问**
   - 检查端口是否被占用（默认：7861）
   - 确认防火墙设置
   - 查看控制台错误信息

### 日志文件

日志文件位于 `logs/` 目录：
- `yolo_detector.log`: 应用程序日志

## 📝 更新日志

### v3.0.0 (2025-11-02)

- 🎉 重构为工具关键点检测系统
- ✨ 实现摄像头实时采集和预览功能
- 🔧 封装工具关键点检测模块
- 📊 实现角度计算和结果可视化
- 💾 实现自动保存功能（原始图、结果图、JSON）
- 📚 更新完整的文档和使用指南
- 🔄 **代码结构优化**: 将 `tool_pose/demo` 中被使用的模块迁移到 `src/yolo_detector` 目录，程序结构更加清晰
- 📝 **配置文件迁移**: 将 `tool_pose/demo/angle_config.yaml` 迁移到 `src/yolo_detector/config/tool_pose_config.yaml`，统一配置管理，并合并了所有配置参数（路径、预测、输出、类别、关键点、角度、可视化、摄像头、日志等）

### v2.0.0 (2025-07-27)

- 🎉 完全重构项目架构
- ✨ 统一的检测和分割界面
- 🔧 模块化设计和配置管理
- 📊 完善的日志和错误处理系统
- 🧪 全面的单元测试覆盖

### v1.0.0 (原始版本)

- 基础的检测和分割功能
- 简单的Gradio界面
- 硬编码的配置

## 📋 开发计划

详细开发计划请参考 `tool_pose/工具检测系统开发计划.md`。

### 当前阶段

- ✅ **阶段1**: 前端原型设计（已完成）
- 🔄 **阶段2**: 核心模块开发（进行中）
  - ✅ 摄像头模块（本机摄像头）
  - ✅ 检测封装模块
- 📋 **阶段3**: 界面开发（待开始）
- 📋 **阶段4**: 保存功能实现（待开始）
- 📋 **阶段5**: 主程序改造（已完成）

### 后续完善

- 📝 **Axis RTSP摄像头支持**: 支持RTSP协议连接网络摄像头
- 📝 **连接稳定性优化**: RTSP连接缓冲和重连机制
- 📝 **性能优化**: 模型加载优化、帧率自适应调整

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 强大的YOLO实现
- [Gradio](https://gradio.app/) - 简单易用的Web界面框架
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)

---

**最后更新时间**: 2025年11月02日  
**项目状态**: ✅ 开发中（核心功能已完成）  
**当前版本**: v3.0.0