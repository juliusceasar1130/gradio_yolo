# 配置文件使用指南

## 配置文件位置
`angle_config.yaml` - 统一配置文件，包含所有可配置参数

## 配置结构

配置文件分为以下几个部分，每个部分都有清晰的分隔线和注释：

### 1. 路径配置（必填）
```yaml
paths:
  model: "路径/to/模型.pt"      # 模型文件路径
  image: "路径/to/图片.jpg"     # 输入图片路径
  output: "outputs"              # 输出目录（相对路径或绝对路径）
```

### 2. 预测参数（可选）
```yaml
predict:
  conf: 0.25          # 置信度阈值 [0.0-1.0]
  iou: 0.45           # IoU阈值 [0.0-1.0]
  imgsz: 640          # 图像尺寸（像素）
  device: ""          # 设备：""=自动, "cuda"=GPU, "cpu"=CPU
  half: false         # 半精度推理
  max_det: 1000       # 最大检测数量
```

### 3. 输出选项（可选）
```yaml
output:
  show_image: true    # 是否显示结果图片
  save_image: false   # 是否保存标注后的图片
  save_json: true     # 是否保存JSON结果文件
```

### 4. 类别和关键点配置（必填）
```yaml
names:
  0: tool1
  1: tool2

kpt_names:
  0: [t1_1, t1_2, ...]
  1: [t2_1, t2_2, ...]
```

### 5. 角度定义（必填）
```yaml
angles:
  tool1:
    angle1: [起点, 顶点, 终点]
    angle2: [起点, 顶点, 终点]
  tool2:
    angle1: [起点, 顶点, 终点]
```

## 快速修改指南

### 修改模型和图片路径
直接在配置文件顶部修改：
```yaml
paths:
  model: "你的模型路径.pt"
  image: "你的图片路径.jpg"
```

### 调整预测参数
在 `predict` 部分修改：
```yaml
predict:
  conf: 0.3      # 提高置信度阈值，减少误检
  imgsz: 1280    # 增大图像尺寸，提高精度
```

### 修改输出选项
在 `output` 部分修改：
```yaml
output:
  show_image: false  # 不显示图片（后台运行）
  save_image: true   # 保存标注后的图片
```

## 配置优势

1. **统一管理** - 所有配置在一个文件中，不需要修改代码
2. **清晰醒目** - 使用分隔线和注释，参数一目了然
3. **易于修改** - 只需修改YAML文件，无需修改Python代码
4. **支持注释** - YAML格式支持详细注释说明

## 注意事项

1. **路径格式**
   - Windows路径使用双反斜杠：`D:\\path\\to\\file.pt`
   - 或使用单正斜杠：`D:/path/to/file.pt`

2. **布尔值**
   - 使用小写：`true` 或 `false`
   - 不要使用引号：`"true"` ❌

3. **数字类型**
   - 整数直接写：`640`
   - 浮点数直接写：`0.25`

4. **列表格式**
   - YAML列表使用短横线：`- item1`
   - 或方括号：`[item1, item2, item3]`

## 配置验证

程序启动时会自动验证配置：
- 检查文件路径是否存在
- 验证参数范围是否合理
- 如果配置有误，会显示具体错误信息

## 示例配置

完整配置示例请参考 `angle_config.yaml` 文件。

