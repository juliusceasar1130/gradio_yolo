# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**工具关键点检测系统** - 基于YOLOv11的工具关键点检测系统，支持实时摄像头采集、关键点检测、角度计算和结果保存。当前已从通用的YOLO检测系统（目标检测、分割、姿态、分类）重构为专门的工具关键点检测系统。

### 核心架构

```
src/yolo_detector/
├── config/              # 配置管理
│   ├── settings.py     # 通用配置管理模块
│   └── tool_pose_config.yaml  # 工具关键点专用配置
├── core/                # 核心功能
│   ├── camera_capture.py    # 摄像头采集（支持本机摄像头和Axis MJPEG）
│   ├── tool_pose_model_loader.py  # 模型加载器
│   ├── tool_pose_predictor.py     # 预测器
│   ├── detector.py         # 历史通用检测器（未使用）
│   ├── image_processor.py  # 图像处理
│   ├── result_processor.py # 结果处理
│   └── batch_processor.py  # 批量处理（未使用）
├── models/              # 模型管理
├── utils/               # 工具函数
│   ├── tool_pose_utils.py         # 工具关键点封装
│   ├── tool_pose_angle_calculator.py  # 角度计算
│   ├── tool_pose_json_utils.py    # JSON处理
│   ├── image_utils.py             # 图像工具
│   ├── file_utils.py              # 文件工具
│   └── logger.py                  # 日志系统
└── ui/                  # 用户界面
    └── gradio_app.py     # Gradio Web界面
```

**重要提示**: 当前项目专注于工具关键点检测，core/中未使用的模块（如detector.py, batch_processor.py）为历史遗留代码。

## 常用开发命令

### 环境设置
```bash
# 激活conda环境（项目使用此环境）
conda activate gradioflask

# 安装依赖
pip install -r requirements.txt

# 开发依赖（代码格式化和测试）
pip install -r requirements-dev.txt
```

### 运行应用
```bash
# 启动Web界面（默认配置，自动端口检测）
python main.py web

# 指定端口启动
python main.py web --port 7862

# 启用调试模式
python main.py web --debug

# 创建公共链接
python main.py web --share

# 指定配置文件
python main.py web --config src/yolo_detector/config/tool_pose_config.yaml

# 指定输出目录
python main.py web --output-dir outputs/tool_pose
```

### 代码质量检查
```bash
# 使用ruff格式化代码
ruff format .

# 使用ruff检查代码
ruff check .
```

### 测试
```bash
# 运行所有测试
python -m pytest

# 运行特定测试文件
python -m pytest tests/test_config.py

# 运行单个测试函数
python -m pytest tests/test_config.py::test_config_load

# 运行带标记的测试
python -m pytest -m unit          # 仅单元测试
python -m pytest -m integration   # 仅集成测试
python -m pytest -m "not slow"    # 跳过慢速测试

# 生成覆盖率报告
python -m pytest --cov=src/yolo_detector --cov-report=html --cov-report=term

# 运行集成测试脚本
python scripts/test_integration.py
```

### 安装开发版本
```bash
# 以开发模式安装
pip install -e .
```

## 配置管理

### 主配置文件
- **通用配置**: `src/yolo_detector/config/settings.py`
- **工具关键点配置**: `src/yolo_detector/config/tool_pose_config.yaml`

### 关键配置项
```yaml
# 模型路径
paths:
  model: "path/to/your/model.pt"

# 检测参数
prediction:
  confidence_threshold: 0.5
  iou_threshold: 0.7

# 输出配置
output:
  save_raw_images: true
  save_result_images: true
  save_json: true

# 摄像头配置
camera:
  type: "webcam"  # webcam 或 axis_mjpeg
  axis_config:
    url: "http://camera-ip/axis-cgi/mjpg/video.cgi"
    username: "user"
    password: "pass"
```

## 开发指南

### 修改文件要求
根据 `.cursor/rules/fileupdate.mdc` 规定：
- **必须**在文件头部添加修改记录注释：
  ```python
  # 创建者/修改者: [姓名]
  # 修改时间: YYYY年MM月DD日 HH:MM
  # 主要修改内容:
  # 1. [修改内容]
  ```

### 代码规范
- 使用 `ruff` 进行代码格式化和检查（版本 0.12.1）
- 在 `README.md` 中记录新增函数清单
- 尽量复用现有函数，避免重复实现
- **包版本要求**:
  - gradio: 5.31.0
  - gradio_client: 1.10.1

### 重要模块

#### 摄像头模块 (src/yolo_detector/core/camera_capture.py)
- 支持本机摄像头（OpenCV）
- 支持Axis MJPEG摄像头（HTTP/HTTPS，带认证）
- 双重回退机制确保兼容性

#### 工具关键点检测 (src/yolo_detector/core/tool_pose_predictor.py)
- 基于YOLOv11 Pose模型
- 关键点检测和角度计算
- 结果可视化

#### Web界面 (src/yolo_detector/ui/gradio_app.py)
- 基于Gradio 5.31.0
- 实时摄像头预览
- 关键点标注显示
- 角度计算结果展示

### 日志系统
- 使用 `loguru` 库
- 日志文件: `logs/yolo_detector.log`
- 通过 `src/yolo_detector/utils/logger.py` 配置
- 支持不同级别的日志记录

## 项目特性

### 当前功能
- ✅ **实时检测**: 本机摄像头和Axis MJPEG摄像头支持
- ✅ **关键点检测**: 基于YOLOv11 Pose的工具关键点检测
- ✅ **角度计算**: 自动计算关键点间角度
- ✅ **结果可视化**: 实时标注显示
- ✅ **自动保存**: 原始图像、结果图像、JSON数据
- ✅ **模块化设计**: 清晰的代码结构

### 输出目录结构
```
outputs/tool_pose/
├── raw_images/         # 检测前原始图片
├── result_images/      # 检测结果图片（带标注）
└── json_results/       # JSON数据文件
```

## 测试指南

### 测试标记 (pytest.ini)
- `unit`: 单元测试
- `integration`: 集成测试
- `slow`: 慢速测试
- `model`: 需要模型文件的测试

### 测试文件位置
- `tests/`: pytest测试目录
- `scripts/`: 集成测试脚本

### 编写测试
```python
import pytest

@pytest.mark.unit
def test_new_function():
    """测试新功能"""
    assert True
```

## 故障排除

### 常见问题
1. **端口占用**: 系统会自动检测可用端口（7861-7870）
2. **模型文件**: 确保 `tool_pose_config.yaml` 中的模型路径正确
3. **摄像头权限**: 确保摄像头访问权限充足
4. **依赖版本**: 严格遵循 `requirements.txt` 中的版本要求

### 调试
- 使用 `--debug` 模式启动Web界面
- 查看 `logs/yolo_detector.log` 日志文件
- Axis摄像头问题：参考 `axis_camera_diagnostic_summary.md`

## 性能优化建议

1. **GPU加速**: 配置PyTorch使用CUDA（如果有GPU）
2. **模型选择**: 根据精度/速度需求选择合适的YOLOv11模型大小
3. **内存管理**: 及时释放不使用的模型和图像数据
4. **批量处理**: 当前版本专注于实时单帧检测，批量处理功能未激活
