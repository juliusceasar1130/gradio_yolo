# YOLO Pose 预测参数完整指南

<!-- 
主要修改内容：更新YOLO Pose预测参数文档，移除无效参数说明
修改时间：2025年10月25日
-->

## 📋 目录

- [概述](#概述)
- [核心检测参数](#核心检测参数)
- [Pose专用参数](#pose专用参数)
- [可视化参数](#可视化参数)
- [高级参数](#高级参数)
- [参数使用示例](#参数使用示例)
- [性能优化建议](#性能优化建议)
- [常见问题解答](#常见问题解答)

## 🎯 概述

YOLO Pose模型支持丰富的预测参数，可以精确控制检测行为、可视化效果和性能表现。本文档基于[Ultralytics官方文档](https://docs.ultralytics.com/zh/modes/predict/#inference-arguments)整理，提供完整的参数说明和使用指南。

### 关键特性

- **17个关键点检测**：基于COCO数据集的标准化人体关键点
- **实时推理**：支持GPU加速和半精度推理
- **灵活可视化**：可控制边界框、关键点、骨架连线的显示
- **批量处理**：支持多图像和视频流处理

## 🔧 核心检测参数

### 基础检测控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `conf` | float | 0.25 | 置信度阈值，过滤低置信度检测 |
| `iou` | float | 0.45 | IoU阈值，用于非极大值抑制(NMS) |
| `max_det` | int | 1000 | 单张图像最大检测数量 |
| `imgsz` | int | 640 | 输入图像尺寸(像素) |
| `device` | str | "" | 推理设备("cpu"/"cuda"/"0"/"1"等) |
| `half` | bool | False | 半精度推理(FP16)，提升速度 |

### 参数详解

#### `conf` - 置信度阈值
```python
# 高精度检测(减少误检)
result = model.predict(image, conf=0.5)

# 宽松检测(增加召回率)
result = model.predict(image, conf=0.1)
```

#### `iou` - IoU阈值
```python
# 严格NMS(减少重叠检测)
result = model.predict(image, iou=0.3)

# 宽松NMS(保留更多检测)
result = model.predict(image, iou=0.8)
```

#### `imgsz` - 图像尺寸
```python
# 高分辨率检测(更准确)
result = model.predict(image, imgsz=1280)

# 快速检测(较低分辨率)
result = model.predict(image, imgsz=416)
```

## 🦴 Pose专用参数

### ⚠️ 重要说明

以下参数**不是** `predict()` 方法的有效参数，它们用于结果可视化：

| 参数 | 类型 | 默认值 | 说明 | 用途 |
|------|------|--------|------|------|
| `kpt_shape` | tuple | (17, 3) | 关键点形状(关键点数, 坐标维度) | 可视化 |
| `kpt_radius` | int | 5 | 关键点绘制半径(像素) | 可视化 |
| `kpt_line` | bool | True | 是否绘制骨架连线 | 可视化 |

**注意**: 这些参数不应传递给 `model.predict()` 方法，否则会报错：
```
'kpt_shape' is not a valid YOLO argument.
'kpt_radius' is not a valid YOLO argument.
'kpt_line' is not a valid YOLO argument.
```

### COCO 17关键点定义

```python
KEYPOINT_NAMES = [
    "鼻子", "左眼", "右眼", "左耳", "右耳",           # 头部 (0-4)
    "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",    # 上肢 (5-10)
    "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"     # 下肢 (11-16)
]

KEYPOINT_GROUPS = {
    "头部": [0, 1, 2, 3, 4],
    "躯干": [5, 6, 11, 12],
    "上肢": [5, 6, 7, 8, 9, 10],
    "下肢": [11, 12, 13, 14, 15, 16]
}
```

### 骨架连接定义

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

## 🎨 可视化参数

### 显示控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `show_labels` | bool | True | 显示类别标签 |
| `show_conf` | bool | True | 显示置信度分数 |
| `show_boxes` | bool | True | 显示边界框 |
| `show_keypoints` | bool | True | 显示关键点 |
| `show_skeleton` | bool | True | 显示骨架连线 |

### 颜色和样式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `color_mode` | str | 'class' | 颜色模式('class'/'instance') |
| `txt_color` | tuple | (255,255,255) | 文本颜色(RGB) |

### 可视化示例

```python
# 执行预测（只使用有效参数）
results = model.predict(
    image, 
    conf=0.25,
    imgsz=640,
    device="cuda"
)

# 在结果可视化时使用可视化参数
if results:
    for result in results:
        # 使用 plot() 方法进行可视化
        annotated_image = result.plot(
            show_boxes=True,      # 显示边界框
            show_labels=True,     # 显示标签
            show_conf=True,       # 显示置信度
            line_width=2,         # 线条宽度
            font_size=12          # 字体大小
        )
        
        # 或者使用自定义可视化
        # 注意：kpt_radius, kpt_line 等参数在自定义绘制时使用
        # 而不是在 predict() 方法中
```

## ⚡ 高级参数

### 性能优化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dnn` | bool | False | 使用OpenCV DNN后端 |
| `data` | str | "" | 数据集配置文件路径 |
| `verbose` | bool | True | 详细输出信息 |

### 流式处理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `stream` | bool | False | 流式处理模式(内存优化) |

```python
# 流式处理长视频
for result in model.predict(video_path, stream=True):
    # 处理每一帧结果
    process_frame(result)
```

## 💡 参数使用示例

### 基础使用

```python
from ultralytics import YOLO

# 加载Pose模型
model = YOLO("yolo11s-pose.pt")

# 基础检测
results = model.predict("image.jpg")
```

### 高级配置

```python
# 高精度多人检测
results = model.predict(
    source="image.jpg",
    conf=0.3,                    # 置信度阈值
    max_det=500,                 # 最大检测数量
    imgsz=1280,                  # 高分辨率
    device="cuda",               # GPU加速
    half=True,                   # 半精度推理
    verbose=False                # 关闭详细输出
)

# 注意：kpt_radius, show_boxes 等参数用于结果可视化，
# 不应传递给 predict() 方法
```

### 批量处理

```python
# 批量处理多张图像
image_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = model.predict(
    source=image_list,
    conf=0.25,
    imgsz=640,
    save=True,                   # 保存结果
    save_txt=True               # 保存标签文件
)
```

### 视频处理

```python
# 视频流处理
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )
        # 处理结果
        annotated_frame = results[0].plot()
        cv2.imshow("Pose Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
```

## 🚀 性能优化建议

### 1. 设备选择

```python
# GPU加速(推荐)
results = model.predict(image, device="cuda")

# 多GPU支持
results = model.predict(image, device="0,1,2,3")

# CPU推理(兼容性)
results = model.predict(image, device="cpu")
```

### 2. 精度与速度平衡

```python
# 速度优先(实时应用)
results = model.predict(
    image,
    imgsz=416,      # 较小输入尺寸
    half=True,      # 半精度
    conf=0.3        # 适中置信度
)

# 精度优先(离线分析)
results = model.predict(
    image,
    imgsz=1280,     # 大输入尺寸
    half=False,     # 全精度
    conf=0.1        # 低置信度阈值
)
```

### 3. 内存优化

```python
# 流式处理(大视频文件)
for result in model.predict(video_path, stream=True):
    # 逐帧处理，避免内存溢出
    process_result(result)

# 批量大小控制
results = model.predict(
    source=image_list,
    batch=4,        # 控制批处理大小
    imgsz=640
)
```

## ❓ 常见问题解答

### Q1: 如何提高检测精度？

**A:** 可以通过以下方式提高精度：
- 增加输入图像尺寸(`imgsz=1280`)
- 降低置信度阈值(`conf=0.1`)
- 使用全精度推理(`half=False`)
- 选择更大型号的模型(`yolo11m-pose.pt`)

### Q2: 如何提升推理速度？

**A:** 速度优化建议：
- 使用GPU加速(`device="cuda"`)
- 启用半精度推理(`half=True`)
- 减小输入尺寸(`imgsz=416`)
- 提高置信度阈值(`conf=0.5`)

### Q3: 关键点检测不准确怎么办？

**A:** 关键点精度优化：
- 确保输入图像质量良好
- 增加图像尺寸(`imgsz=1280`)
- 检查光照条件
- 考虑使用更大型号的模型

### Q4: 如何处理多人场景？

**A:** 多人检测配置：
- 增加最大检测数量(`max_det=500`)
- 调整IoU阈值(`iou=0.5`)
- 确保图像分辨率足够高
- 考虑使用更大的模型

### Q5: 内存不足怎么办？

**A:** 内存优化方案：
- 使用流式处理(`stream=True`)
- 减小批处理大小
- 降低图像尺寸
- 使用半精度推理

## 📚 参考资源

- [Ultralytics官方文档](https://docs.ultralytics.com/zh/modes/predict/#inference-arguments)
- [YOLO Pose模型下载](https://github.com/ultralytics/assets/releases)
- [COCO关键点格式说明](https://cocodataset.org/#format-data)

## 🔄 更新日志

- **2025-10-25**: 修正参数说明，移除无效的kpt_shape、kpt_radius、kpt_line参数
- **2025-10-25**: 更新可视化示例，明确参数用途区分
- **2025-01-27**: 初始版本，包含完整参数说明和使用示例
- **2025-01-27**: 添加性能优化建议和常见问题解答

---

*本文档基于Ultralytics YOLO官方文档整理，适用于YOLO11 Pose模型。如有疑问，请参考官方文档或提交Issue。*
