# YOLO Pose 姿态检测完整使用指南

<!-- 
主要修改内容：创建YOLO Pose姿态检测的完整使用指南
修改时间：2025年10月25日
-->

## 📋 目录

- [快速开始](#快速开始)
- [Web界面使用](#web界面使用)
- [命令行使用](#命令行使用)
- [API使用](#api使用)
- [参数配置](#参数配置)
- [常见问题](#常见问题)
- [最佳实践](#最佳实践)

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖：

```bash
# 激活conda环境
conda activate gradioflask

# 检查依赖
pip list | grep ultralytics
```

### 2. 模型文件

- **模型路径**: `D:\00deeplearn\yolo11\【2】训练模型\pose\yolo11s-pose.pt`
- **模型类型**: YOLO11s Pose (轻量级，适合实时检测)
- **关键点**: 17个COCO格式关键点

### 3. 快速测试

```bash
# 启动Web界面
python main.py web

# 或使用命令行
python main.py detect --mode pose --source path/to/image.jpg
```

## 🌐 Web界面使用

### 启动Web界面

```bash
python main.py web
```

访问 `http://localhost:7860` 打开Web界面。

### 使用步骤

1. **选择模式**: 在界面顶部选择 "姿态检测" 模式
2. **上传文件**: 
   - 点击 "上传图像" 或 "上传视频"
   - 支持格式：JPG, PNG, MP4, AVI
3. **配置参数**:
   - 置信度阈值：0.1-0.9 (默认0.25)
   - 图像尺寸：416/640/1280 (默认640)
   - 设备：CPU/GPU (自动检测)
4. **开始检测**: 点击 "开始检测" 按钮
5. **查看结果**: 
   - 检测结果图像
   - 关键点坐标信息
   - 检测统计信息

### Web界面功能

- ✅ **实时预览**: 检测结果实时显示
- ✅ **参数调节**: 动态调整检测参数
- ✅ **结果下载**: 保存检测结果图像
- ✅ **批量处理**: 支持多文件上传
- ✅ **进度显示**: 显示检测进度

## 💻 命令行使用

### 基础命令

```bash
# 检测单张图像
python main.py detect --mode pose --source image.jpg

# 检测视频
python main.py detect --mode pose --source video.mp4

# 实时摄像头检测
python main.py detect --mode pose --source 0

# 检测文件夹
python main.py detect --mode pose --source /path/to/images/
```

### 高级参数

```bash
# 高精度检测
python main.py detect --mode pose --source image.jpg --conf 0.1 --imgsz 1280

# 快速检测
python main.py detect --mode pose --source image.jpg --conf 0.5 --imgsz 416

# 指定输出目录
python main.py detect --mode pose --source image.jpg --output-dir results/

# 保存结果
python main.py detect --mode pose --source image.jpg --save --save-txt
```

### 命令行参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--mode` | 检测模式 | pose | `--mode pose` |
| `--source` | 输入源 | - | `--source image.jpg` |
| `--conf` | 置信度阈值 | 0.25 | `--conf 0.3` |
| `--imgsz` | 图像尺寸 | 640 | `--imgsz 1280` |
| `--device` | 设备 | auto | `--device cuda` |
| `--save` | 保存结果 | False | `--save` |
| `--save-txt` | 保存标签 | False | `--save-txt` |

## 🔧 API使用

### 基础API调用

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

### 高级API使用

```python
# 自定义参数检测
result = detector.detect(
    'image.jpg',
    confidence_threshold=0.3,    # 置信度阈值
    imgsz=1280,                  # 图像尺寸
    max_det=500,                 # 最大检测数
    device='cuda',               # 设备选择
    half=True                    # 半精度推理
)

# 批量检测
image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = []
for image_path in image_list:
    result = detector.detect(image_path)
    results.append(result)
```

### 结果处理

```python
# 获取检测结果
if result:
    boxes = result.boxes          # 边界框
    keypoints = result.keypoints  # 关键点
    conf = result.conf           # 置信度
    class_ids = result.class_ids # 类别ID
    
    # 处理关键点数据
    for i, kpts in enumerate(keypoints):
        print(f"Person {i+1}:")
        for j, kpt in enumerate(kpts):
            x, y, visibility = kpt
            print(f"  Keypoint {j}: ({x:.1f}, {y:.1f}), visibility: {visibility}")
```

## ⚙️ 参数配置

### 配置文件

编辑 `configs/default.yaml` 文件：

```yaml
pose:
  confidence_threshold: 0.25    # 置信度阈值
  max_det: 1000                 # 最大检测数
  iou: 0.45                     # IoU阈值
  imgsz: 640                     # 图像尺寸
  device: ""                     # 设备选择
  half: false                    # 半精度推理
  show_labels: true              # 显示标签
  show_conf: true                # 显示置信度
  show_boxes: true               # 显示边界框
  # 以下参数用于可视化，不传递给predict()
  kpt_shape: [17, 3]             # 关键点形状
  kpt_radius: 5                  # 关键点半径
  kpt_line: true                 # 骨架连线
```

### 性能配置

#### 速度优先配置

```yaml
pose:
  confidence_threshold: 0.3
  imgsz: 416
  device: "cuda"
  half: true
  max_det: 100
```

#### 精度优先配置

```yaml
pose:
  confidence_threshold: 0.1
  imgsz: 1280
  device: "cuda"
  half: false
  max_det: 500
```

#### 平衡配置 (推荐)

```yaml
pose:
  confidence_threshold: 0.25
  imgsz: 640
  device: "cuda"
  half: true
  max_det: 300
```

## ❓ 常见问题

### Q1: 检测不到人物怎么办？

**A:** 尝试以下解决方案：
- 降低置信度阈值 (`conf=0.1`)
- 增加图像尺寸 (`imgsz=1280`)
- 检查图像质量和光照条件
- 确认图像中包含完整的人物

### Q2: 检测结果不准确怎么办？

**A:** 提高检测精度：
- 使用高分辨率输入 (`imgsz=1280`)
- 降低置信度阈值 (`conf=0.1`)
- 关闭半精度推理 (`half=false`)
- 使用更大型号的模型

### Q3: 检测速度太慢怎么办？

**A:** 优化检测速度：
- 使用GPU加速 (`device="cuda"`)
- 启用半精度推理 (`half=true`)
- 减小输入尺寸 (`imgsz=416`)
- 提高置信度阈值 (`conf=0.5`)

### Q4: 出现参数错误怎么办？

**A:** 检查参数使用：
```python
# ❌ 错误 - 这些参数不能传递给predict()
results = model.predict(image, kpt_shape=(17, 3), kpt_radius=5)

# ✅ 正确 - 只使用有效参数
results = model.predict(image, conf=0.25, imgsz=640)
```

### Q5: 如何处理多人场景？

**A:** 多人检测配置：
- 增加最大检测数 (`max_det=500`)
- 调整IoU阈值 (`iou=0.5`)
- 确保图像分辨率足够高
- 考虑使用更大的模型

## 💡 最佳实践

### 1. 图像预处理

```python
# 推荐的图像预处理
import cv2

def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 调整图像尺寸（可选）
    height, width = image.shape[:2]
    if width > 1920:  # 限制最大宽度
        scale = 1920 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    return image
```

### 2. 结果后处理

```python
def process_pose_results(result):
    """处理姿态检测结果"""
    if not result:
        return None
    
    processed_results = []
    
    for i, (box, keypoints, conf) in enumerate(zip(result.boxes, result.keypoints, result.conf)):
        # 过滤低置信度检测
        if conf < 0.3:
            continue
            
        # 提取关键点
        person_data = {
            'id': i,
            'confidence': float(conf),
            'bbox': box.tolist(),
            'keypoints': keypoints.tolist()
        }
        
        processed_results.append(person_data)
    
    return processed_results
```

### 3. 性能监控

```python
import time

def benchmark_detection(detector, image_path, iterations=10):
    """性能基准测试"""
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        result = detector.detect(image_path)
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    
    print(f"平均检测时间: {avg_time:.3f}s")
    print(f"检测帧率: {fps:.1f} FPS")
    
    return avg_time, fps
```

### 4. 错误处理

```python
def safe_detect(detector, image_path):
    """安全的检测函数，包含错误处理"""
    try:
        result = detector.detect(image_path)
        return result, None
    except FileNotFoundError:
        return None, "图像文件不存在"
    except Exception as e:
        return None, f"检测失败: {str(e)}"
```

## 📚 相关文档

- [姿态检测使用指南](pose.md)
- [参数快速参考](pose_parameters_quick_reference.md)
- [参数完整指南](pose_prediction_parameters.md)
- [修复报告](pose_fix_report.md)
- [项目说明](README.md)

## 🔄 更新日志

- **2025-10-25**: 创建完整使用指南
- **2025-10-25**: 添加最佳实践和错误处理
- **2025-10-25**: 更新参数配置说明

---

*本文档提供YOLO Pose姿态检测的完整使用指南，包括Web界面、命令行和API使用方法。如有疑问，请参考相关技术文档。*
