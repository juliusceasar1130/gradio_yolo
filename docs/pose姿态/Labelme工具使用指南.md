# Labelme数据标注工具使用指南

## 📋 文档信息
- **创建者/修改者**: chenliang
- **修改时间**: 2025年1月27日
- **主要修改内容**: Labelme数据标注工具使用指南和示例

---

## 🎯 概述

本指南提供使用Labelme进行数据标注的完整工具链，包括标注、转换、训练和验证的完整流程。

---

## 📁 文件说明

### 核心文件
- `docs/Labelme数据标注完整指南.md` - 详细的标注指南文档
- `labelme_to_yolo_pose_converter.py` - Labelme到YOLO格式转换工具
- `pose_training_tool.py` - Pose模型训练和验证工具

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate gradioflask

# 安装labelme
pip install labelme
```

### 2. 标注数据

#### 2.1 创建标签文件

在图像文件夹中创建 `labels.txt`：

```txt
top_left
top_right
bottom_right
bottom_left
```

#### 2.2 使用Labelme标注

```bash
# 启动labelme
labelme

# 或指定图像文件夹
labelme /path/to/your/images
```

**标注步骤：**
1. 点击 "Open Dir" 选择图像文件夹
2. 点击 "Tools" → "Load Labels" 加载 `labels.txt`
3. 使用 "Create Rectangle" 框选目标
4. 使用 "Create Point" 标注关键点
5. 保存标注结果

### 3. 转换格式

#### 3.1 基础转换

```bash
# 转换Labelme JSON文件为YOLO格式
python labelme_to_yolo_pose_converter.py \
    --input_dir /path/to/json/files \
    --output_dir /path/to/yolo/labels \
    --dim 3
```

#### 3.2 带可见性的转换

```bash
# 使用可见性标注
python labelme_to_yolo_pose_converter.py \
    --input_dir /path/to/json/files \
    --output_dir /path/to/yolo/labels \
    --use_visibility \
    --dim 3
```

#### 3.3 自定义关键点

```bash
# 自定义关键点名称
python labelme_to_yolo_pose_converter.py \
    --input_dir /path/to/json/files \
    --output_dir /path/to/yolo/labels \
    --keypoints corner1 corner2 corner3 corner4 \
    --dim 3
```

#### 3.4 创建数据集配置

```bash
# 转换并创建YAML配置文件
python labelme_to_yolo_pose_converter.py \
    --input_dir /path/to/json/files \
    --output_dir /path/to/yolo/labels \
    --create_yaml \
    --dataset_name quadrilateral_dataset \
    --dim 3
```

### 4. 训练模型

#### 4.1 创建数据集配置

```bash
# 创建示例数据集配置
python pose_training_tool.py create_config \
    --output dataset.yaml \
    --dataset_name quadrilateral \
    --keypoints top_left top_right bottom_right bottom_left \
    --kpt_shape 4 3
```

#### 4.2 训练模型

```bash
# 训练自定义Pose模型
python pose_training_tool.py train \
    --data dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --model_size s
```

#### 4.3 验证模型

```bash
# 验证单个图像
python pose_training_tool.py validate \
    --model pose_training/custom_model/weights/best.pt \
    --image test_image.jpg

# 批量验证
python pose_training_tool.py validate \
    --model pose_training/custom_model/weights/best.pt \
    --dir test_images/
```

---

## 📋 详细使用示例

### 示例1：四边形关键点检测

#### 1. 准备数据
```
quadrilateral_dataset/
├── images/
│   ├── train/
│   │   ├── quad1.jpg
│   │   ├── quad2.jpg
│   │   └── ...
│   └── val/
│       ├── val_quad1.jpg
│       └── ...
├── labels.txt
└── annotations/
    ├── quad1.json
    ├── quad2.json
    └── ...
```

#### 2. 创建标签文件
```txt
top_left
top_right
bottom_right
bottom_left
```

#### 3. 转换格式
```bash
python labelme_to_yolo_pose_converter.py \
    --input_dir quadrilateral_dataset/annotations \
    --output_dir quadrilateral_dataset/labels \
    --create_yaml \
    --dataset_name quadrilateral \
    --dim 3
```

#### 4. 训练模型
```bash
python pose_training_tool.py train \
    --data quadrilateral_dataset/quadrilateral.yaml \
    --epochs 100 \
    --batch 16
```

### 示例2：COCO 17关键点检测

#### 1. 创建标签文件
```txt
nose
left_eye
right_eye
left_ear
right_ear
left_shoulder
right_shoulder
left_elbow
right_elbow
left_wrist
right_wrist
left_hip
right_hip
left_knee
right_knee
left_ankle
right_ankle
```

#### 2. 转换格式
```bash
python labelme_to_yolo_pose_converter.py \
    --input_dir coco_annotations \
    --output_dir coco_labels \
    --keypoints nose left_eye right_eye left_ear right_ear left_shoulder right_shoulder left_elbow right_elbow left_wrist right_wrist left_hip right_hip left_knee right_knee left_ankle right_ankle \
    --create_yaml \
    --dataset_name coco_pose \
    --dim 3
```

---

## 🔧 高级用法

### 1. 可见性标注

#### 创建带可见性的标签文件
```txt
top_left_visible
top_left_occluded
top_left_invisible
top_right_visible
top_right_occluded
top_right_invisible
bottom_right_visible
bottom_right_occluded
bottom_right_invisible
bottom_left_visible
bottom_left_occluded
bottom_left_invisible
```

#### 转换时使用可见性
```bash
python labelme_to_yolo_pose_converter.py \
    --input_dir annotations \
    --output_dir labels \
    --use_visibility \
    --dim 3
```

### 2. 自定义骨架连接

在YAML配置文件中自定义骨架连接：

```yaml
# 自定义骨架连接
skeleton:
  - [0, 1]  # 连接关键点0和1
  - [1, 2]  # 连接关键点1和2
  - [2, 3]  # 连接关键点2和3
  - [3, 0]  # 连接关键点3和0
```

### 3. 多类别检测

```yaml
# 多类别配置
nc: 3
names:
  0: person
  1: car
  2: quadrilateral

kpt_names:
  0:  # person的关键点
    - nose
    - left_eye
    - right_eye
    # ... 更多关键点
  1:  # car的关键点
    - front_left
    - front_right
    - rear_left
    - rear_right
  2:  # quadrilateral的关键点
    - top_left
    - top_right
    - bottom_right
    - bottom_left
```

---

## 🎯 最佳实践

### 1. 标注质量
- 确保关键点标注在像素级别准确
- 保持所有图像中关键点顺序一致
- 定期检查标注质量

### 2. 数据平衡
- 包含不同角度、光照条件的图像
- 建议验证集占20%
- 确保训练集和验证集分布相似

### 3. 模型选择
- 小数据集：使用 `yolo11s-pose.pt`
- 大数据集：使用 `yolo11m-pose.pt` 或 `yolo11l-pose.pt`

### 4. 训练参数
- 根据数据集大小调整batch size
- 调整学习率和训练轮数
- 使用数据增强提高泛化能力

---

## 🐛 常见问题

### 1. 关键点顺序错误
**问题**: 转换后的关键点顺序不正确
**解决**: 检查 `labels.txt` 文件，确保标注时按照正确顺序

### 2. 坐标归一化问题
**问题**: 坐标超出0-1范围
**解决**: 检查图像尺寸读取是否正确

### 3. 可见性标注问题
**问题**: 可见性值不正确
**解决**: 使用标签命名约定或检查转换脚本

### 4. 训练不收敛
**问题**: 模型训练效果差
**解决**: 
- 检查数据质量
- 调整学习率
- 增加训练数据
- 使用预训练模型

---

## 📚 参考资料

- [Ultralytics YOLO Pose数据集格式](https://docs.ultralytics.com/zh/datasets/pose/#ultralytics-yolo-format)
- [Labelme官方文档](https://github.com/wkentaro/labelme)
- [YOLO Pose官方文档](https://docs.ultralytics.com/tasks/pose/)

---

## 📝 更新日志

### v1.0 (2025-01-27)
- 初始版本
- 支持Labelme到YOLO格式转换
- 支持自定义关键点标注
- 支持可见性标注
- 提供完整的训练和验证工具

---

*文档版本: v1.0*  
*最后更新: 2025年1月27日*
