# YOLO Pose 参数快速参考

<!-- 
主要修改内容：更新YOLO Pose参数快速参考，修正无效参数说明
修改时间：2025年10月25日
-->

## 🚀 常用参数组合

### 实时检测(速度优先)
```python
results = model.predict(
    image,
    conf=0.3,
    imgsz=416,
    device="cuda",
    half=True,
    max_det=100
)
```

### 高精度检测(精度优先)
```python
results = model.predict(
    image,
    conf=0.1,
    imgsz=1280,
    device="cuda",
    half=False,
    max_det=500
)
```

### 平衡模式(推荐)
```python
results = model.predict(
    image,
    conf=0.25,
    imgsz=640,
    device="cuda",
    half=True,
    max_det=300
)
```

## 📋 参数速查表

### ✅ 有效参数 (用于 predict())

| 参数 | 类型 | 默认值 | 说明 | 推荐值 |
|------|------|--------|------|--------|
| `conf` | float | 0.25 | 置信度阈值 | 0.25-0.5 |
| `iou` | float | 0.45 | IoU阈值 | 0.3-0.7 |
| `max_det` | int | 1000 | 最大检测数 | 100-500 |
| `imgsz` | int | 640 | 图像尺寸 | 416/640/1280 |
| `device` | str | "" | 设备 | "cuda"/"cpu" |
| `half` | bool | False | 半精度 | True(速度)/False(精度) |
| `verbose` | bool | True | 详细输出 | True/False |

### ⚠️ 可视化参数 (不用于 predict())

| 参数 | 类型 | 默认值 | 说明 | 用途 |
|------|------|--------|------|------|
| `kpt_radius` | int | 5 | 关键点半径 | 可视化绘制 |
| `kpt_line` | bool | True | 骨架连线 | 可视化绘制 |
| `kpt_shape` | tuple | (17, 3) | 关键点形状 | 可视化绘制 |

## 🎨 可视化控制

```python
# 执行预测（只使用有效参数）
results = model.predict(image, conf=0.25, imgsz=640)

# 在结果可视化时使用可视化参数
if results:
    annotated_image = results[0].plot(
        show_boxes=True,      # 显示边界框
        show_labels=True,    # 显示标签
        show_conf=True       # 显示置信度
    )
```

### 可视化选项

```python
# 完整显示
show_boxes=True, show_labels=True, show_conf=True

# 仅关键点和骨架
show_boxes=False, show_labels=False, show_keypoints=True, show_skeleton=True

# 仅骨架连线
show_boxes=False, show_keypoints=False, show_skeleton=True
```

## ⚡ 性能对比

| 配置 | 速度 | 精度 | 内存 | 适用场景 |
|------|------|------|------|----------|
| 实时模式 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 实时应用 |
| 平衡模式 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 通用场景 |
| 精度模式 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 离线分析 |

## 🔧 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 检测不到人 | conf太高 | 降低conf到0.1-0.2 |
| 检测太多误报 | conf太低 | 提高conf到0.3-0.5 |
| 速度太慢 | imgsz太大 | 降低imgsz到416 |
| 精度不够 | imgsz太小 | 提高imgsz到1280 |
| 内存不足 | 批处理太大 | 使用stream=True |
| **参数错误** | **使用无效参数** | **移除kpt_shape、kpt_radius、kpt_line** |

### ⚠️ 常见错误

```python
# ❌ 错误用法 - 会导致参数错误
results = model.predict(
    image,
    kpt_shape=(17, 3),    # 无效参数
    kpt_radius=5,         # 无效参数
    kpt_line=True         # 无效参数
)

# ✅ 正确用法
results = model.predict(
    image,
    conf=0.25,
    imgsz=640,
    device="cuda"
)
```

---

*快速参考 - 详细说明请参考 `pose_prediction_parameters.md`*
