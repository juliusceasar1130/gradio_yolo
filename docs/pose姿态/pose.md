# YOLO Pose 姿态检测使用指南

<!-- 
主要修改内容：更新姿态检测的正确使用方法和参数说明
修改时间：2025年10月25日
-->

## 🎯 概述

YOLO Pose 是 Ultralytics 提供的人体姿态检测模型，能够检测人体关键点并绘制骨架连线。本项目已集成 YOLO11 Pose 模型，支持实时姿态检测。

## 📁 模型文件

- **模型位置**: `D:\00deeplearn\yolo11\【2】训练模型\pose\yolo11s-pose.pt`
- **模型类型**: YOLO11s Pose (轻量级)
- **关键点数量**: 17个 (COCO格式)
- **支持格式**: 图像、视频、实时摄像头

## 🚀 快速开始

### 1. Web界面使用

```bash
# 启动Web界面
python main.py web
```

在Web界面中：
1. 选择 **"姿态检测"** 模式
2. 上传包含人物的图像或视频
3. 点击 **"开始检测"**
4. 查看检测结果和关键点标注

### 2. 命令行使用

```bash
# 检测单张图像
python main.py detect --mode pose --source path/to/image.jpg

# 检测视频
python main.py detect --mode pose --source path/to/video.mp4

# 实时摄像头检测
python main.py detect --mode pose --source 0
```

## ⚙️ 配置参数

### 有效参数 (用于预测)

| 参数 | 类型 | 默认值 | 说明 | 推荐值 |
|------|------|--------|------|--------|
| `conf` | float | 0.25 | 置信度阈值 | 0.25-0.5 |
| `iou` | float | 0.45 | IoU阈值 | 0.3-0.7 |
| `max_det` | int | 1000 | 最大检测数 | 100-500 |
| `imgsz` | int | 640 | 图像尺寸 | 416/640/1280 |
| `device` | str | "" | 设备选择 | "cuda"/"cpu" |
| `half` | bool | False | 半精度推理 | True(速度)/False(精度) |
| `verbose` | bool | True | 详细输出 | True/False |

### 可视化参数 (用于结果展示)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `show_labels` | bool | True | 显示类别标签 |
| `show_conf` | bool | True | 显示置信度 |
| `show_boxes` | bool | True | 显示边界框 |
| `kpt_shape` | tuple | (17, 3) | 关键点形状 |
| `kpt_radius` | int | 5 | 关键点半径 |
| `kpt_line` | bool | True | 骨架连线 |

## 🦴 关键点说明

### COCO 17关键点定义

```python
KEYPOINT_NAMES = [
    "鼻子", "左眼", "右眼", "左耳", "右耳",           # 头部 (0-4)
    "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",    # 上肢 (5-10)
    "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"     # 下肢 (11-16)
]
```

### 骨架连接

```python
SKELETON_CONNECTIONS = [
    # 头部连接
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 上肢连接
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # 躯干连接
    (5, 11), (6, 12), (11, 12),
    # 下肢连接
    (11, 13), (13, 15), (12, 14), (14, 16)
]
```

## 💡 使用示例

### 基础检测

```python
from yolo_detector.core import YOLODetector
from yolo_detector.config import Settings

# 创建检测器
config = Settings()
detector = YOLODetector(detector_type='pose', config=config)

# 加载模型
detector.load_model()

# 执行检测
result = detector.detect('path/to/image.jpg')
```

### 高级配置

```python
# 高精度检测
result = detector.detect(
    'image.jpg',
    confidence_threshold=0.1,    # 低置信度阈值
    imgsz=1280,                  # 高分辨率
    max_det=500                  # 更多检测
)

# 快速检测
result = detector.detect(
    'image.jpg',
    confidence_threshold=0.5,    # 高置信度阈值
    imgsz=416,                   # 低分辨率
    half=True                    # 半精度推理
)
```

## 🎨 可视化选项

### 显示控制

```python
# 完整显示
show_boxes=True, show_labels=True, show_conf=True

# 仅关键点和骨架
show_boxes=False, show_labels=False, show_keypoints=True, show_skeleton=True

# 仅骨架连线
show_boxes=False, show_keypoints=False, show_skeleton=True
```

## ⚡ 性能优化

### 速度优先配置

```yaml
pose:
  confidence_threshold: 0.3
  imgsz: 416
  device: "cuda"
  half: true
  max_det: 100
```

### 精度优先配置

```yaml
pose:
  confidence_threshold: 0.1
  imgsz: 1280
  device: "cuda"
  half: false
  max_det: 500
```

### 平衡配置 (推荐)

```yaml
pose:
  confidence_threshold: 0.25
  imgsz: 640
  device: "cuda"
  half: true
  max_det: 300
```

## 🔧 故障排除

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 检测不到人 | 置信度太高 | 降低 `conf` 到 0.1-0.2 |
| 误检太多 | 置信度太低 | 提高 `conf` 到 0.3-0.5 |
| 速度太慢 | 图像尺寸太大 | 降低 `imgsz` 到 416 |
| 精度不够 | 图像尺寸太小 | 提高 `imgsz` 到 1280 |
| 内存不足 | 批处理太大 | 使用 `stream=True` |

### 错误排查

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型文件存在且完整

2. **检测结果为空**
   - 降低置信度阈值
   - 检查输入图像质量
   - 确认图像中包含人物

3. **性能问题**
   - 使用GPU加速 (`device="cuda"`)
   - 启用半精度推理 (`half=True`)
   - 减小输入图像尺寸

## 📊 输出格式

### 检测结果结构

```python
{
    'boxes': [...],           # 边界框信息
    'keypoints': [...],       # 关键点坐标
    'conf': [...],            # 置信度分数
    'class_ids': [...],       # 类别ID
    'names': [...]            # 类别名称
}
```

### 关键点坐标格式

```python
# 每个关键点包含 [x, y, visibility]
keypoint = [x_coord, y_coord, visibility_score]
# visibility: 0=不可见, 1=遮挡, 2=可见
```

## 📚 相关文档

- [姿态检测参数快速参考](pose_parameters_quick_reference.md)
- [姿态检测参数完整指南](pose_prediction_parameters.md)
- [姿态检测修复报告](pose_fix_report.md)
- [项目使用说明](README.md)

## 🔄 更新日志

- **2025-10-25**: 更新文档，修正参数使用说明
- **2025-10-25**: 添加故障排除和性能优化建议
- **2025-01-27**: 初始版本创建

---

*本文档基于 YOLO11 Pose 模型编写，适用于当前项目版本。如有疑问，请参考相关技术文档。*
 