# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目架构

这是一个基于YOLOv11的目标检测、图像分割和姿态识别系统，采用模块化设计：

### 核心模块结构
- `src/yolo_detector/` - 主要源代码目录
  - `config/` - 配置管理模块，基于YAML文件的统一配置系统
  - `models/` - 模型加载器，支持多种YOLO模型（检测、分割、姿态、分类）
  - `core/` - 核心检测逻辑，包含各种检测器和处理器
  - `utils/` - 工具函数（图像处理、文件操作、日志、异常处理）
  - `ui/` - Gradio Web界面

### 主要组件
- **Config**: 统一配置管理器，从 `configs/default.yaml` 加载配置
- **ModelLoader**: 模型加载和管理，支持模型缓存和状态监控
- **检测器类**: ObjectDetector, SegmentationDetector, PoseDetector
- **处理器类**: ImageProcessor, ResultProcessor, BatchProcessor

## 常用开发命令

### 环境设置
```bash
# 安装依赖
pip install -r requirements.txt

# 查看系统信息和模型状态
python main.py info
```

### 运行应用
```bash
# 启动Web界面（开发模式）
python main.py web --debug

# 启动Web界面（生产模式）
python main.py web

# 单张图像检测
python main.py detect --image path/to/image.jpg --output result.jpg

# 批量处理
python main.py batch --input path/to/images/ --output path/to/results/
```

### 测试
```bash
# 运行所有测试
python -m pytest

# 运行特定测试
python -m pytest tests/test_config.py

# 生成覆盖率报告
python -m pytest --cov=src/yolo_detector --cov-report=html

# 运行集成测试
python scripts/test_integration.py
```

## 开发指南

### 配置管理
- 主配置文件: `configs/default.yaml`
- 包含模型路径、数据路径、检测参数、UI配置等
- 使用 Config 类进行统一管理
- 支持通过代码动态修改配置

### 模型路径配置
模型路径在 `configs/default.yaml` 中配置：
- 检测模型: `models.detection.path`
- 分割模型: `models.segmentation.path`
- 姿态模型: `models.pose.path`
- 分类模型: `models.classification.path`

### 日志系统
- 日志文件: `logs/yolo_detector.log`
- 使用 loguru 库进行日志管理
- 支持日志轮转和自动清理
- 可通过配置调整日志级别和格式

### 添加新功能
1. 在相应模块中添加新类或函数
2. 更新 `__init__.py` 导出新组件
3. 在配置文件中添加相关配置（如需要）
4. 编写单元测试
5. 更新文档

### 错误处理
- 使用 `src/yolo_detector/utils/exceptions.py` 中的自定义异常类
- 所有主要函数都应包含适当的异常处理
- 错误信息会记录到日志文件

## 项目特性

### 支持的检测类型
- 目标检测 (Object Detection)
- 图像分割 (Segmentation)
- 姿态识别 (Pose Detection)
- 图像分类 (Classification)

### Web界面功能
- 基于 Gradio 的直观 Web 界面
- 支持拖拽上传图片
- 实时结果展示
- 示例图像画廊
- 多种检测模式切换

### 批量处理功能
- 支持文件夹批量处理
- 进度回调显示
- 结果统计和导出
- 多线程处理优化

## 性能优化建议

1. **模型选择**: 根据需求选择合适的模型大小（YOLOv11n/s/m/l/x）
2. **GPU加速**: 确保PyTorch支持CUDA，配置文件中设置 `system.device: "cuda"`
3. **批量处理**: 调整 `batch_processing.max_workers` 和 `chunk_size` 参数
4. **内存管理**: 监控内存使用，及时释放不需要的模型