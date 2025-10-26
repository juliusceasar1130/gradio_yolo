# Labelme数据标注完整指南

## 📋 文档信息
- **创建者/修改者**: chenliang
- **修改时间**: 2025年1月27日
- **主要修改内容**: Labelme数据标注完整指南，包含环境配置、标注流程、格式转换等

---

## 🎯 概述

本文档详细介绍如何使用Labelme工具进行数据标注，特别是针对YOLO Pose模型的自定义关键点标注。包含从环境安装到模型训练的完整流程。

---

## 📦 第一部分：环境准备

### 1.1 安装Labelme

```bash
# 激活conda环境
conda activate gradioflask

# 安装labelme
pip install labelme
```

### 1.2 验证安装

```bash
# 启动labelme验证安装
labelme
```

如果成功启动GUI界面，说明安装成功。

---

## 🏗️ 第二部分：标注准备

### 2.1 创建标签配置文件

在图像文件夹中创建 `labels.txt` 文件，定义关键点名称：

#### 2.1.1 四边形关键点标注
```txt
top_left
top_right
bottom_right
bottom_left
```

#### 2.1.2 带可见性的关键点标注
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

#### 2.1.3 COCO 17关键点标注
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

### 2.2 数据集目录结构

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── val_image1.jpg
│       ├── val_image2.jpg
│       └── ...
├── labels.txt
└── annotations/  # Labelme生成的JSON文件
    ├── image1.json
    ├── image2.json
    └── ...
```

---

## 🎨 第三部分：Labelme标注流程

### 3.1 启动和基本操作

1. **启动Labelme**
   ```bash
   labelme
   ```

2. **打开图像文件夹**
   - 点击 "Open Dir" 按钮
   - 选择包含待标注图像的文件夹

3. **加载标签文件**
   - 点击 "Tools" 菜单
   - 选择 "Load Labels"
   - 选择之前创建的 `labels.txt` 文件

### 3.2 标注步骤

#### 3.2.1 目标检测标注（可选）
1. 选择 "Create Rectangle" 工具
2. 框选目标对象
3. 输入类别名称（如：person, quadrilateral等）

#### 3.2.2 关键点标注
1. 选择 "Create Point" 工具
2. 在图像上点击关键点位置
3. 从下拉列表中选择对应的标签名称
4. 重复步骤2-3，标注所有关键点

#### 3.2.3 保存标注
1. 完成一张图像的标注后
2. 点击 "Save" 按钮或按 `Ctrl+S`
3. Labelme会生成对应的JSON文件

### 3.3 标注注意事项

1. **关键点顺序一致性**
   - 确保所有图像中关键点的标注顺序完全一致
   - 建议按照 `labels.txt` 中的顺序进行标注

2. **标注精度**
   - 关键点应标注在像素级别准确的位置
   - 对于被遮挡的关键点，使用相应的可见性标签

3. **数据质量**
   - 选择清晰、对比度高的图像
   - 避免模糊、过暗或过亮的图像

---

## 🔄 第四部分：格式转换

### 4.1 YOLO格式说明

根据Ultralytics官方文档，YOLO Pose格式有两种维度：

#### 4.1.1 Dim = 2 格式（2D关键点）
```
<class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
```

#### 4.1.2 Dim = 3 格式（3D关键点）
```
<class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> ... <pxn> <pyn> <pn-visibility>
```

**可见性标志说明：**
- `0`: 关键点不可见/被遮挡
- `1`: 关键点部分可见/被遮挡
- `2`: 关键点完全可见

### 4.2 转换脚本

#### 4.2.1 基础转换脚本

```python
# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：Labelme格式转YOLO格式转换脚本

import os
import json
import cv2
import numpy as np
from pathlib import Path

def labelme_to_yolo_pose(labelme_json_path, output_dir, class_id=0, dim=3):
    """
    将Labelme格式转换为YOLO Pose格式
    
    Args:
        labelme_json_path: Labelme JSON文件路径
        output_dir: 输出目录
        class_id: 类别ID
        dim: 关键点维度 (2 或 3)
    """
    
    # 自定义关键点顺序
    keypoint_names = [
        "top_left",      # 左上角
        "top_right",     # 右上角
        "bottom_right",  # 右下角
        "bottom_left"    # 左下角
    ]
    
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个标注对象
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            # 处理边界框
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 转换为YOLO格式 (中心点, 宽, 高) - 归一化
            bbox_center_x = (x1 + x2) / 2 / image_width
            bbox_center_y = (y1 + y2) / 2 / image_height
            bbox_width = abs(x2 - x1) / image_width
            bbox_height = abs(y2 - y1) / image_height
            
            # 查找对应的关键点
            keypoints = []
            for i, keypoint_name in enumerate(keypoint_names):
                keypoint_found = False
                for kp_shape in data['shapes']:
                    if (kp_shape['shape_type'] == 'point' and 
                        kp_shape['label'] == keypoint_name):
                        kp_x, kp_y = kp_shape['points'][0]
                        # 转换为相对坐标
                        kp_x_norm = kp_x / image_width
                        kp_y_norm = kp_y / image_height
                        
                        if dim == 2:
                            # Dim=2格式：只有x,y坐标
                            keypoints.extend([kp_x_norm, kp_y_norm])
                        elif dim == 3:
                            # Dim=3格式：x,y坐标 + 可见性
                            keypoints.extend([kp_x_norm, kp_y_norm, 2])  # 2表示完全可见
                        
                        keypoint_found = True
                        break
                
                if not keypoint_found:
                    if dim == 2:
                        keypoints.extend([0, 0])  # 不可见的关键点
                    elif dim == 3:
                        keypoints.extend([0, 0, 0])  # 不可见的关键点
            
            # 生成YOLO格式标签
            yolo_line = [str(class_id)]
            yolo_line.extend([str(bbox_center_x), str(bbox_center_y), 
                            str(bbox_width), str(bbox_height)])
            yolo_line.extend([str(kp) for kp in keypoints])
            
            # 保存标签文件
            image_name = Path(data['imagePath']).stem
            label_file = os.path.join(output_dir, f"{image_name}.txt")
            
            with open(label_file, 'w') as f:
                f.write(' '.join(yolo_line) + '\n')

def batch_convert(input_dir, output_dir, dim=3):
    """
    批量转换Labelme标注文件
    """
    json_files = list(Path(input_dir).glob('*.json'))
    
    for json_file in json_files:
        print(f"转换文件: {json_file}")
        labelme_to_yolo_pose(str(json_file), output_dir, dim=dim)

if __name__ == "__main__":
    # 使用示例
    input_directory = "path/to/your/labelme/json/files"
    output_directory = "path/to/yolo/labels"
    
    # 选择维度格式
    # dim=2: 2D关键点 (x,y)
    # dim=3: 3D关键点 (x,y,visibility)
    batch_convert(input_directory, output_directory, dim=3)
    print("转换完成！")
```

#### 4.2.2 支持可见性的转换脚本

```python
def labelme_to_yolo_pose_with_visibility(labelme_json_path, output_dir, class_id=0):
    """
    将Labelme格式转换为YOLO Pose格式，支持visibility标注
    """
    
    # 基础关键点名称
    base_keypoint_names = [
        "top_left",
        "top_right", 
        "bottom_right",
        "bottom_left"
    ]
    
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个标注对象
    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle':
            # 处理边界框
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # 转换为YOLO格式 (中心点, 宽, 高) - 归一化
            bbox_center_x = (x1 + x2) / 2 / image_width
            bbox_center_y = (y1 + y2) / 2 / image_height
            bbox_width = abs(x2 - x1) / image_width
            bbox_height = abs(y2 - y1) / image_height
            
            # 查找对应的关键点
            keypoints = []
            for i, base_name in enumerate(base_keypoint_names):
                keypoint_found = False
                
                # 查找匹配的关键点标注
                for kp_shape in data['shapes']:
                    if kp_shape['shape_type'] == 'point':
                        label = kp_shape['label']
                        
                        # 检查是否匹配基础名称
                        if label.startswith(base_name):
                            kp_x, kp_y = kp_shape['points'][0]
                            # 转换为相对坐标
                            kp_x_norm = kp_x / image_width
                            kp_y_norm = kp_y / image_height
                            
                            # 根据标签名称确定visibility
                            visibility = determine_visibility(label)
                            
                            # Dim=3格式：x,y坐标 + 可见性
                            keypoints.extend([kp_x_norm, kp_y_norm, visibility])
                            keypoint_found = True
                            break
                
                if not keypoint_found:
                    # 未找到关键点，设置为不可见
                    keypoints.extend([0, 0, 0])
            
            # 生成YOLO格式标签
            yolo_line = [str(class_id)]
            yolo_line.extend([str(bbox_center_x), str(bbox_center_y), 
                            str(bbox_width), str(bbox_height)])
            yolo_line.extend([str(kp) for kp in keypoints])
            
            # 保存标签文件
            image_name = Path(data['imagePath']).stem
            label_file = os.path.join(output_dir, f"{image_name}.txt")
            
            with open(label_file, 'w') as f:
                f.write(' '.join(yolo_line) + '\n')

def determine_visibility(label):
    """
    根据标签名称确定visibility值
    """
    if 'visible' in label.lower():
        return 2  # 完全可见
    elif 'occluded' in label.lower():
        return 1  # 部分可见/被遮挡
    elif 'invisible' in label.lower():
        return 0  # 不可见
    else:
        return 2  # 默认完全可见
```

---

## 📁 第五部分：数据集配置

### 5.1 数据集目录结构

```
custom_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── val_image1.jpg
│       ├── val_image2.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── val/
│       ├── val_image1.txt
│       ├── val_image2.txt
│       └── ...
└── dataset.yaml
```

### 5.2 数据集配置文件

#### 5.2.1 四边形关键点数据集配置

```yaml
# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：自定义四边形Pose训练数据集配置文件

# 数据集路径
path: ./custom_quadrilateral_dataset
train: images/train
val: images/val

# 关键点配置 - 根据选择的维度格式
kpt_shape: [4, 3]  # 4个关键点，3个维度 (x, y, visibility)
# 如果使用Dim=2格式，则改为: kpt_shape: [4, 2]

# 类别配置
nc: 1
names:
  0: quadrilateral

# 关键点名称
kpt_names:
  0:
    - top_left
    - top_right
    - bottom_right
    - bottom_left

# 骨架连接
skeleton:
  - [0, 1]  # 上边
  - [1, 2]  # 右边
  - [2, 3]  # 下边
  - [3, 0]  # 左边
```

#### 5.2.2 COCO 17关键点数据集配置

```yaml
# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：COCO 17关键点数据集配置文件

# 数据集路径
path: ./coco_pose_dataset
train: images/train
val: images/val

# 关键点配置
kpt_shape: [17, 3]  # 17个关键点，3个维度 (x, y, visibility)

# 类别配置
nc: 1
names:
  0: person

# 关键点名称
kpt_names:
  0:
    - nose
    - left_eye
    - right_eye
    - left_ear
    - right_ear
    - left_shoulder
    - right_shoulder
    - left_elbow
    - right_elbow
    - left_wrist
    - right_wrist
    - left_hip
    - right_hip
    - left_knee
    - right_knee
    - left_ankle
    - right_ankle

# 骨架连接
skeleton:
  - [0, 1]  # 鼻子-左眼
  - [0, 2]  # 鼻子-右眼
  - [1, 3]  # 左眼-左耳
  - [2, 4]  # 右眼-右耳
  - [5, 6]  # 左肩-右肩
  - [5, 7]  # 左肩-左肘
  - [7, 9]  # 左肘-左腕
  - [6, 8]  # 右肩-右肘
  - [8, 10] # 右肘-右腕
  - [5, 11] # 左肩-左髋
  - [6, 12] # 右肩-右髋
  - [11, 12] # 左髋-右髋
  - [11, 13] # 左髋-左膝
  - [13, 15] # 左膝-左踝
  - [12, 14] # 右髋-右膝
  - [14, 16] # 右膝-右踝
```

---

## 🚀 第六部分：模型训练

### 6.1 训练脚本

```python
# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：自定义Pose模型训练脚本

from ultralytics import YOLO
import torch

def train_custom_pose_model():
    """
    训练自定义Pose模型
    """
    # 检查GPU可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    model = YOLO('yolo11s-pose.pt')  # 或 yolo11m-pose.pt, yolo11l-pose.pt
    
    # 训练参数
    results = model.train(
        data='dataset.yaml',           # 数据集配置文件
        epochs=100,                    # 训练轮数
        imgsz=640,                     # 图像尺寸
        batch=16,                      # 批次大小
        device=device,                 # 设备
        workers=4,                     # 数据加载线程数
        project='pose_training',       # 项目目录
        name='custom_model',           # 实验名称
        save=True,                     # 保存检查点
        save_period=10,                # 每10个epoch保存一次
        val=True,                      # 验证
        plots=True,                    # 生成训练图表
        verbose=True                   # 详细输出
    )
    
    print("自定义Pose模型训练完成！")
    return results

if __name__ == "__main__":
    train_custom_pose_model()
```

### 6.2 验证脚本

```python
# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：自定义Pose模型验证脚本

from ultralytics import YOLO
import cv2
import numpy as np

def validate_custom_pose_model(model_path, test_image_path):
    """
    验证训练好的自定义Pose模型
    """
    # 加载训练好的模型
    model = YOLO(model_path)
    
    # 预测
    results = model.predict(
        source=test_image_path,
        conf=0.25,
        imgsz=640,
        save=True,
        save_txt=True
    )
    
    # 显示结果
    for result in results:
        annotated_image = result.plot(
            show_boxes=True,
            show_labels=True,
            show_conf=True,
            line_width=2
        )
        
        # 保存结果
        cv2.imwrite('pose_result.jpg', annotated_image)
        print("结果已保存为 pose_result.jpg")
        
        # 打印关键点信息
        if result.keypoints is not None:
            print(f"检测到 {len(result.keypoints)} 个对象")
            for i, kpts in enumerate(result.keypoints):
                print(f"对象 {i+1} 的关键点:")
                for j, kpt in enumerate(kpts):
                    if kpt[2] > 0:  # 如果关键点可见
                        print(f"  关键点 {j+1}: ({kpt[0]:.2f}, {kpt[1]:.2f})")

if __name__ == "__main__":
    # 使用训练好的模型
    model_path = "pose_training/custom_model/weights/best.pt"
    test_image = "path/to/test/image.jpg"
    
    validate_custom_pose_model(model_path, test_image)
```

---

## 🎯 第七部分：最佳实践

### 7.1 标注质量保证

1. **一致性检查**
   - 确保所有图像使用相同的关键点顺序
   - 定期检查标注质量
   - 使用多人标注时统一标准

2. **数据平衡**
   - 包含不同角度、光照条件的图像
   - 确保训练集和验证集的分布相似
   - 建议验证集占20%

3. **标注效率**
   - 使用快捷键提高标注速度
   - 批量处理相似图像
   - 定期保存标注进度

### 7.2 常见问题解决

1. **关键点顺序错误**
   - 检查 `labels.txt` 文件
   - 重新标注或修改转换脚本

2. **坐标归一化问题**
   - 确保所有坐标都在0-1范围内
   - 检查图像尺寸读取是否正确

3. **可见性标注问题**
   - 使用标签命名约定
   - 或使用自动可见性检测

### 7.3 性能优化

1. **数据集大小**
   - 建议至少1000张训练图像
   - 关键点数量越多，需要更多数据

2. **模型选择**
   - 小数据集：使用 `yolo11s-pose.pt`
   - 大数据集：使用 `yolo11m-pose.pt` 或 `yolo11l-pose.pt`

3. **训练参数调优**
   - 根据数据集大小调整batch size
   - 调整学习率和训练轮数

---

## 📚 第八部分：参考资料

### 8.1 官方文档
- [Ultralytics YOLO Pose数据集格式](https://docs.ultralytics.com/zh/datasets/pose/#ultralytics-yolo-format)
- [Labelme官方文档](https://github.com/wkentaro/labelme)

### 8.2 相关工具
- **Labelme**: 图像标注工具
- **Ultralytics YOLO**: 目标检测和姿态估计框架
- **OpenCV**: 图像处理库

### 8.3 示例数据集
- **COCO-Pose**: 人体姿态估计数据集
- **自定义数据集**: 根据需求创建的数据集

---

## 🔧 附录：常用命令

### A.1 Labelme相关命令
```bash
# 启动labelme
labelme

# 指定图像文件夹启动
labelme /path/to/images

# 指定标签文件启动
labelme --labels /path/to/labels.txt
```

### A.2 数据转换命令
```bash
# 运行转换脚本
python labelme_to_yolo_pose.py

# 批量转换
python batch_convert.py --input_dir /path/to/json --output_dir /path/to/labels
```

### A.3 模型训练命令
```bash
# 训练模型
python train_custom_pose.py

# 验证模型
python validate_custom_pose_model.py
```

---

## 📝 总结

本文档提供了使用Labelme进行数据标注的完整流程，包括：

1. **环境准备**: 安装和配置Labelme
2. **标注流程**: 详细的标注步骤和注意事项
3. **格式转换**: Labelme到YOLO格式的转换方法
4. **数据集配置**: 数据集组织结构和配置文件
5. **模型训练**: 训练和验证脚本
6. **最佳实践**: 质量保证和性能优化建议

通过遵循本指南，您可以成功创建高质量的自定义关键点检测数据集，并训练出满足需求的Pose模型。

---

*文档版本: v1.0*  
*最后更新: 2025年1月27日*
