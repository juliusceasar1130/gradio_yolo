# YOLO Pose 预测与角度计算系统

## 项目概述

本项目是一个基于 YOLO Pose 的姿态检测与角度计算系统，实现了从图像输入到关键点检测、角度计算、结果可视化和保存的完整流程。采用模块化架构设计，代码结构清晰，易于扩展和维护。

**主要功能：**
- 🔍 YOLO Pose 关键点检测
- 📐 三点角度自动计算
- 🎨 可视化结果标注
- 💾 JSON 结果导出
- ⚙️ 统一配置管理

**最后更新：** 2025年11月2日

---

## 目录结构

```
demo/
├── pose_predict.py              # 主入口脚本
├── angle_config.yaml            # 统一配置文件（所有参数集中管理）
├── config/                      # 配置管理模块
│   ├── __init__.py
│   └── settings.py              # Config类、配置加载和验证
├── core/                        # 核心功能模块
│   ├── __init__.py
│   ├── model_loader.py          # 模型加载器（单例模式）
│   ├── predictor.py             # YOLO预测执行器
│   └── data_extractor.py        # 结果数据提取器
├── utils/                       # 工具函数模块
│   ├── __init__.py
│   ├── angle_calculator.py      # 角度计算核心（配置文件加载、角度计算、图像标注）
│   ├── json_utils.py            # JSON序列化工具和结果保存
│   ├── output_utils.py          # 输出工具（打印检测结果和角度结果）
│   ├── image_utils.py           # 图像处理和显示工具
│   └── validators.py            # 路径和配置验证工具
├── outputs/                     # 输出目录（自动创建）
│   └── angle_results_*.json     # 角度计算结果JSON文件
└── README.md                    # 本文档
```

---

## 技术栈

### 核心依赖
- **ultralytics**: YOLO模型加载和预测
- **numpy**: 数值计算和数组操作
- **opencv-python (cv2)**: 图像处理和绘制
- **PIL/Pillow**: Unicode文本绘制（支持角度符号°）
- **PyYAML**: 配置文件解析
- **matplotlib**: 图像显示

### Python版本
- Python 3.9+

---

## 快速开始

### 1. 安装依赖

```bash
pip install ultralytics numpy opencv-python pillow pyyaml matplotlib
```

### 2. 配置文件设置

编辑 `angle_config.yaml` 文件，修改以下关键路径：

```yaml
paths:
  # 模型文件路径（.pt文件）
  model: "D:\\path\\to\\your\\model.pt"
  
  # 输入图片路径
  image: "D:\\path\\to\\your\\image.jpg"
  
  # 输出目录（自动创建）
  output: "outputs"
```

### 3. 运行脚本

```bash
cd tool_pose/demo
python pose_predict.py
```

### 4. 查看结果

- **控制台输出**：检测结果、角度计算结果
- **图像显示**：自动弹出显示标注后的图像
- **JSON文件**：保存在 `outputs/` 目录下

---

## 技术架构

### 架构设计原则

1. **模块化**：每个模块负责单一职责，便于测试和维护
2. **解耦**：模块间依赖最小化，通过接口交互
3. **可扩展**：易于添加新功能模块
4. **配置集中**：所有配置统一在 `angle_config.yaml` 管理

### 系统流程

```
输入图像
  ↓
配置加载 (config/settings.py)
  ↓
模型加载 (core/model_loader.py) → 单例缓存
  ↓
YOLO预测 (core/predictor.py)
  ↓
结果提取 (core/data_extractor.py)
  ↓
角度计算 (utils/angle_calculator.py)
  ↓
图像标注 (utils/angle_calculator.py)
  ↓
结果保存 (utils/json_utils.py)
  ↓
可视化显示 (utils/image_utils.py)
```

### 模块依赖关系

```
pose_predict.py (主入口)
├── config/settings.py (配置管理)
├── core/
│   ├── model_loader.py (模型加载)
│   ├── predictor.py (预测执行)
│   └── data_extractor.py (数据提取)
└── utils/
    ├── angle_calculator.py (角度计算核心)
    ├── json_utils.py (JSON工具和结果保存)
    ├── output_utils.py (输出工具)
    ├── image_utils.py (图像工具)
    └── validators.py (验证工具)
```

---

## 核心功能

### 1. 关键点检测
- 基于 YOLO Pose 模型进行实时关键点检测
- 支持多类别、多对象同时检测
- 自动提取关键点坐标和可见性信息

### 2. 角度计算
- **三点角度计算**：通过三个关键点计算角度（中间点为顶点）
- **批量处理**：自动为每个对象计算所有定义的角度
- **错误处理**：处理关键点缺失、不可见等情况

### 3. 可视化标注
- YOLO 默认检测框和关键点标注
- 角度值文本标注（支持Unicode符号°）
- 角度连线绘制（p1→p2→p3）

### 4. 结果导出
- JSON格式保存完整检测和角度信息
- 包含对象信息、关键点坐标、角度值等
- 时间戳命名，避免覆盖

---

## 配置文件说明

### angle_config.yaml 结构

```yaml
# 路径配置（必填）
paths:
  model: "模型路径.pt"
  image: "图片路径.jpg"
  output: "outputs"

# 预测参数（可选，有默认值）
predict:
  conf: 0.25        # 置信度阈值 [0-1]
  iou: 0.45         # IoU阈值 [0-1]
  imgsz: 640        # 图像尺寸（像素）
  device: ""        # 设备：""=自动, "cuda"=GPU, "cpu"=CPU
  half: false       # 半精度推理
  max_det: 1000     # 最大检测数量

# 输出选项（可选）
output:
  show_image: true  # 是否显示结果图片
  save_image: false # 是否保存标注后的图片
  save_json: true   # 是否保存JSON结果文件

# 类别和关键点配置
names:
  0: tool1
  1: tool2

kpt_names:
  0: [t1_1, t1_2, ...]  # tool1的关键点名称列表
  1: [t2_1, t2_2, ...]  # tool2的关键点名称列表

# 角度定义
angles:
  tool1:
    angle1: [t1_1, t1_2, t1_3]  # [起点, 顶点, 终点]
    angle2: [t1_2, t1_3, t1_4]
    ...
  tool2:
    angle1: [t2_1, t2_2, t2_3]
    ...
```

详细配置说明请参考 `CONFIG_GUIDE.md`

---

## 函数功能清单

### 主入口模块 (pose_predict.py)

#### `main()`
- **功能**：主函数，执行完整的预测和角度计算流程
- **流程**：
  1. 加载配置
  2. 验证路径
  3. 加载角度配置文件
  4. 加载模型
  5. 执行预测
  6. 计算角度
  7. 标注图像
  8. 保存结果
  9. 显示结果

---

### 配置管理模块 (config/settings.py)

#### `Config` 类
- **功能**：配置数据类，统一管理所有配置项
- **属性**：
  - `model_path`: 模型文件路径
  - `image_path`: 输入图片路径
  - `config_path`: 配置文件路径
  - `output_dir`: 输出目录
  - `conf`: 置信度阈值
  - `iou`: IoU阈值
  - `imgsz`: 图像尺寸
  - `device`: 设备（cuda/cpu/自动）
  - `half`: 半精度推理
  - `max_det`: 最大检测数量
  - `show_image`: 是否显示图像
  - `save_image`: 是否保存图像
  - `save_json`: 是否保存JSON

#### `Config.validate()`
- **功能**：验证配置有效性
- **返回**：`(是否有效: bool, 错误信息: Optional[str])`

#### `load_config(config_path: Optional[Path] = None) -> Config`
- **功能**：从YAML配置文件加载配置
- **参数**：
  - `config_path`: 配置文件路径，None则使用默认路径
- **返回**：`Config` 对象

#### `get_default_config() -> Config`
- **功能**：获取默认配置（从YAML文件加载）
- **返回**：`Config` 对象

#### `get_config_dict(config: Config) -> Dict[str, Any]`
- **功能**：获取配置字典（包含完整YAML配置）
- **参数**：
  - `config`: 配置对象
- **返回**：配置字典

---

### 模型加载模块 (core/model_loader.py)

#### `ModelLoader` 类
- **功能**：模型加载器（单例模式，支持缓存）

#### `ModelLoader.get_model(model_path: str, device: str = "") -> YOLO`
- **功能**：获取模型实例（支持缓存，避免重复加载）
- **参数**：
  - `model_path`: 模型文件路径
  - `device`: 设备（"cuda" 或 "cpu"，空字符串表示自动选择）
- **返回**：YOLO模型实例

#### `ModelLoader.clear_cache()`
- **功能**：清除模型缓存

#### `ModelLoader.reload_model(model_path: str, device: str = "") -> YOLO`
- **功能**：重新加载模型（清除缓存后加载）
- **参数**：
  - `model_path`: 模型文件路径
  - `device`: 设备
- **返回**：YOLO模型实例

---

### 预测执行模块 (core/predictor.py)

#### `Predictor` 类
- **功能**：预测执行器

#### `Predictor.predict(model: YOLO, image_path: str, config: Config) -> List`
- **功能**：执行YOLO预测
- **参数**：
  - `model`: YOLO模型实例
  - `image_path`: 输入图片路径
  - `config`: 配置对象
- **返回**：预测结果列表

#### `Predictor.validate_results(results: List) -> bool`
- **功能**：验证预测结果有效性
- **参数**：
  - `results`: 预测结果列表
- **返回**：是否有效

---

### 数据提取模块 (core/data_extractor.py)

#### `DataExtractor` 类
- **功能**：数据提取器

#### `DataExtractor.extract(results: List[Results]) -> Dict[str, Any]`
- **功能**：从预测结果中提取结构化数据
- **参数**：
  - `results`: YOLO预测结果列表
- **返回**：提取的结构化数据字典
- **包含**：
  - `image_shape`: 图像尺寸
  - `boxes`: 边界框列表
  - `keypoints`: 关键点列表
  - `classes`: 类别列表

#### `DataExtractor.get_keypoints_for_object(result: Results, obj_idx: int) -> Optional[np.ndarray]`
- **功能**：获取指定对象的关键点数据
- **参数**：
  - `result`: YOLO预测结果
  - `obj_idx`: 对象索引
- **返回**：关键点数据 `[num_keypoints, 3]`，不存在返回None

#### `DataExtractor.get_class_for_object(result: Results, obj_idx: int) -> Optional[int]`
- **功能**：获取指定对象的类别ID
- **参数**：
  - `result`: YOLO预测结果
  - `obj_idx`: 对象索引
- **返回**：类别ID，不存在返回None

---

### 角度计算模块 (utils/angle_calculator.py)

#### `load_angle_config(config_path: str) -> Dict[str, Any]`
- **功能**：加载角度配置文件（YAML格式）
- **参数**：
  - `config_path`: 角度配置文件路径
- **返回**：角度配置字典

#### `get_system_font_path() -> Optional[str]`
- **功能**：获取系统字体路径（支持Unicode）
- **返回**：字体文件路径，找不到返回None
- **支持平台**：Windows、macOS、Linux

#### `draw_text_with_pil(image, text, position, font_size=20, color=(0,255,0), bg_color=None, bg_alpha=0.6) -> np.ndarray`
- **功能**：使用PIL绘制文本（支持Unicode字符如°符号）
- **参数**：
  - `image`: OpenCV图像（BGR格式）
  - `text`: 要绘制的文本
  - `position`: 文本位置 (x, y)
  - `font_size`: 字体大小
  - `color`: 文本颜色 (B, G, R)
  - `bg_color`: 背景颜色 (B, G, R)
  - `bg_alpha`: 背景透明度 (0-1)
- **返回**：绘制后的图像

#### `get_keypoint_index_by_name(kpt_name: str, kpt_names_list: List[str]) -> Optional[int]`
- **功能**：通过关键点名称字符串匹配获取关键点索引
- **参数**：
  - `kpt_name`: 关键点名称（如 't1_1'）
  - `kpt_names_list`: 关键点名称列表
- **返回**：关键点索引，未找到返回None

#### `calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float`
- **功能**：计算三点角度（p2为顶点）
- **参数**：
  - `p1`: 起点坐标 [x, y]
  - `p2`: 顶点坐标 [x, y]
  - `p3`: 终点坐标 [x, y]
- **返回**：角度值（度），范围 [0, 180]

#### `calculate_angles_for_object(class_name, kpt_data, kpt_names, angle_config, visibility_threshold=0.5) -> Dict[str, Any]`
- **功能**：为单个对象计算所有角度
- **参数**：
  - `class_name`: 类别名称（如 'tool1', 'tool2'）
  - `kpt_data`: 关键点数据 `[num_keypoints, 3]`，每行为 [x, y, visibility]
  - `kpt_names`: 关键点名称列表
  - `angle_config`: 角度配置字典
  - `visibility_threshold`: 可见性阈值
- **返回**：角度计算结果字典
- **结果格式**：
  ```python
  {
    'angle1': {
      'value': 123.45,  # 角度值（度）
      'keypoints': ['t1_1', 't1_2', 't1_3'],
      'keypoint_indices': [0, 1, 2],
      'keypoint_positions': {...},
      'valid': True
    },
    ...
  }
  ```

#### `annotate_angles_on_image(image, angles_result, keypoints_xy, keypoint_indices_map, font_scale=0.6, font_thickness=2, arc_radius=25) -> np.ndarray`
- **功能**：在图像上标注角度值（显示文本和连线）
- **参数**：
  - `image`: 图像数组（BGR格式）
  - `angles_result`: 角度计算结果字典
  - `keypoints_xy`: 关键点坐标数组 `[num_keypoints, 2]`
  - `keypoint_indices_map`: 关键点名称到索引的映射
  - `font_scale`: 字体大小
  - `font_thickness`: 字体粗细
  - `arc_radius`: 角度弧线半径（已禁用，保留用于日后开发）
- **返回**：标注后的图像
- **标注内容**：
  - 关键点连线（p1→p2→p3）
  - 角度文本（格式：`angle_name: value°`）

---

### JSON工具模块 (utils/json_utils.py)

#### `convert_to_json_serializable(obj: Any) -> Any`
- **功能**：将对象转换为JSON可序列化的格式
- **处理类型**：
  - numpy整数 → Python int
  - numpy浮点数 → Python float
  - numpy布尔值 → Python bool
  - numpy数组 → Python list
  - 字典和列表（递归转换）
- **参数**：
  - `obj`: 待转换的对象
- **返回**：JSON可序列化的对象

#### `save_results_to_json(result, angles_results, image_path, output_dir, class_names, kpt_names_dict) -> Path`
- **功能**：保存检测和角度结果到JSON文件
- **参数**：
  - `result`: YOLO预测结果对象
  - `angles_results`: 角度计算结果列表
  - `image_path`: 图片路径（Path对象）
  - `output_dir`: 输出目录（Path对象）
  - `class_names`: 类别ID到类别名称的映射字典
  - `kpt_names_dict`: 类别ID到关键点名称列表的映射字典
- **返回**：保存的JSON文件路径（Path对象）
- **输出**：`outputs/angle_results_YYYYMMDD_HHMMSS.json`
- **包含内容**：
  - 图像路径和尺寸信息
  - 每个对象的边界框坐标
  - 每个对象的关键点坐标和可见性
  - 每个对象的角度计算结果
  - 时间戳信息

---

### 输出工具模块 (utils/output_utils.py)

#### `print_detection_results(result, class_names: Dict[int, str])`
- **功能**：打印检测结果到控制台
- **参数**：
  - `result`: YOLO预测结果对象
  - `class_names`: 类别ID到类别名称的映射字典
- **输出内容**：
  - 检测到的对象数量和每个对象的类别、ID、置信度
  - 关键点数量和可见性统计

#### `print_angle_results(angles_results: List[Dict[str, Any]])`
- **功能**：打印角度计算结果到控制台
- **参数**：
  - `angles_results`: 角度计算结果列表，每个元素包含：
    - `object_id`: 对象索引
    - `class_name`: 类别名称
    - `angles`: 角度计算结果字典
- **输出内容**：
  - 每个对象的角度值（度）
  - 哪些角度计算成功（✓）或失败（✗）及原因
  - 成功计算的角度数量统计

---

### 图像工具模块 (utils/image_utils.py)

#### `convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray`
- **功能**：将BGR格式图像转换为RGB格式
- **参数**：
  - `image`: 输入图像（BGR格式）
- **返回**：RGB格式图像

#### `display_image(image, title="图像显示", figsize=(12,8), show=True) -> None`
- **功能**：使用matplotlib显示图像
- **参数**：
  - `image`: 图像数组（BGR或RGB格式）
  - `title`: 图像标题
  - `figsize`: 图像显示尺寸
  - `show`: 是否立即显示

#### `save_image(image, output_path, is_bgr=True) -> bool`
- **功能**：保存图像到文件
- **参数**：
  - `image`: 图像数组
  - `output_path`: 输出文件路径
  - `is_bgr`: 是否为BGR格式（cv2.imwrite需要BGR格式）
- **返回**：是否保存成功

---

### 验证工具模块 (utils/validators.py)

#### `validate_path(path, path_type="文件") -> tuple[bool, Optional[str]]`
- **功能**：验证路径是否存在
- **参数**：
  - `path`: 路径（str或Path对象）
  - `path_type`: 路径类型（用于错误提示）
- **返回**：`(是否有效, 错误信息)`

#### `validate_paths(*paths: tuple[Union[str, Path], str]) -> tuple[bool, Optional[str]]`
- **功能**：批量验证路径
- **参数**：
  - `*paths`: 路径元组列表，每个元组为 `(path, path_type)`
- **返回**：`(是否全部有效, 错误信息)`

---

## 使用示例

### 基本使用

1. **修改配置文件** `angle_config.yaml`：
   ```yaml
   paths:
     model: "path/to/model.pt"
     image: "path/to/image.jpg"
   ```

2. **运行脚本**：
   ```bash
   python pose_predict.py
   ```

3. **查看结果**：
   - 控制台会显示检测结果和角度值
   - 自动弹出图像显示窗口
   - JSON文件保存在 `outputs/` 目录

### 自定义角度定义

在 `angle_config.yaml` 中添加角度定义：

```yaml
angles:
  tool1:
    angle1: [t1_1, t1_2, t1_3]  # [起点, 顶点, 终点]
    angle2: [t1_2, t1_3, t1_4]
    angle3: [t1_3, t1_4, t1_5]
```

### 调整预测参数

```yaml
predict:
  conf: 0.3      # 降低置信度阈值，检测更多对象
  iou: 0.5       # 提高IoU阈值，减少重复检测
  imgsz: 1280    # 增大图像尺寸，提高精度（但速度变慢）
```

---

## 输出格式

### JSON输出结构

```json
{
  "image_path": "path/to/image.jpg",
  "image_shape": [1080, 1920, 3],
  "timestamp": "20250131_123456",
  "predictions": [
    {
      "object_id": 0,
      "class_id": 0,
      "class_name": "tool1",
      "confidence": 0.95,
      "bbox": {
        "x1": 100.0,
        "y1": 200.0,
        "x2": 300.0,
        "y2": 400.0
      },
      "keypoints": [
        {
          "index": 0,
          "name": "t1_1",
          "x": 150.0,
          "y": 250.0,
          "visibility": 0.98,
          "visible": true
        },
        ...
      ],
      "angles": {
        "angle1": {
          "value": 123.45,
          "keypoints": ["t1_1", "t1_2", "t1_3"],
          "keypoint_indices": [0, 1, 2],
          "keypoint_positions": {
            "t1_1": [150.0, 250.0],
            "t1_2": [200.0, 300.0],
            "t1_3": [250.0, 350.0]
          },
          "valid": true
        },
        ...
      }
    }
  ]
}
```

---

## 常见问题

### Q1: 模型加载失败
**A:** 检查 `angle_config.yaml` 中的 `paths.model` 路径是否正确，确保 `.pt` 文件存在。

### Q2: 角度计算返回 None
**A:** 可能原因：
- 关键点未检测到或不可见（可见性 < 阈值）
- 关键点坐标为 (0,0)
- 角度定义中的关键点名称不匹配

### Q3: 图像显示失败
**A:** 确保安装了 `matplotlib` 且支持图形界面显示。

### Q4: JSON序列化错误
**A:** 使用了 `convert_to_json_serializable()` 函数，正常情况下应该能处理所有类型。

### Q5: 配置文件加载失败
**A:** 确保 `angle_config.yaml` 文件存在且格式正确（YAML语法）。

---

## 开发历史

### 2025年11月2日 - 函数模块化迁移
- 将 `print_detection_results()` 和 `print_angle_results()` 迁移至 `utils/output_utils.py`
- 将 `save_results_to_json()` 迁移至 `utils/json_utils.py`
- 主入口文件精简约37%（从365行减至230行）
- 提高代码复用性和可维护性
- 更新 README.md 文档，反映新的模块结构

### 2025年11月2日 - 文档更新
- 更新所有文件的时间注释
- 合并 README.md 和 REFACTORING.md 为综合文档

### 2025年1月31日 - 结构简化
- 移除冗余的 `processing/` 模块
- `angle_calculator.py` 迁移至 `utils/` 目录
- 主入口直接使用工具函数，简化调用链

### 2025年1月31日 - 模块化重构
- 将单一脚本重构为模块化架构
- 分离配置管理、核心功能、工具函数
- 统一配置文件管理

### 2025年1月28日 - 初始版本
- 基础 YOLO Pose 预测功能
- 三点角度计算
- 图像标注功能

---

## 扩展指南

### 添加新的角度定义

1. 在 `angle_config.yaml` 的 `angles` 部分添加新角度：
   ```yaml
   angles:
     tool1:
       new_angle: [t1_5, t1_6, t1_7]
   ```

2. 运行脚本即可自动计算新角度。

### 添加新的处理步骤

在 `pose_predict.py` 的 `main()` 函数中添加：

```python
# 添加新的处理步骤
new_result = process_new_feature(result, config)
```

### 自定义可视化样式

修改 `utils/angle_calculator.py` 中的：
- `annotate_angles_on_image()` 函数中的颜色定义
- `draw_text_with_pil()` 函数中的字体大小和位置

---

## 许可证

本项目为内部工具，使用需遵循项目相关规定。

---

## 联系方式

如有问题或建议，请联系项目维护者。
