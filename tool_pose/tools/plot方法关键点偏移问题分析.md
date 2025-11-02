# plot()方法关键点偏移问题分析

**创建时间**：2025年1月27日  
**参考文档**：[Ultralytics YOLO Predict Mode](https://docs.ultralytics.com/zh/modes/predict/#key-features-of-predict-mode)

---

## 📖 官方文档分析

根据 Ultralytics YOLO 官方文档，`plot()` 方法的关键参数：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `img` | `np.ndarray` | **用于绘制图像的替代图像。如果未提供，则使用原始图像。** | `None` |
| `kpt_radius` | `int` | 绘制的关键点半径 | `5` |
| `kpt_line` | `bool` | 用线条连接关键点 | `True` |
| `labels` | `bool` | 在注释中包含类别标签 | `True` |
| `boxes` | `bool` | 在图像上叠加边界框 | `True` |
| `masks` | `bool` | 在图像上叠加掩码 | `True` |
| `line_width` | `int` | 边界框线条宽度 | `3` |

---

## 🔍 当前代码检查

### 当前实现

```python
# src/yolo_detector/core/detector.py (第614行)
def _create_visualization(self) -> Optional[np.ndarray]:
    """创建可视化图像"""
    try:
        if hasattr(self.raw_result, 'plot'):
            # 使用可视化参数调用plot方法
            return self.raw_result.plot(**self.visualization_params)
```

**分析**：
- ✅ 代码看起来是正确的
- ✅ YOLO的 `Results` 对象内部已经保存了原始图像信息
- ✅ `plot()` 方法会自动处理坐标映射

---

## ⚠️ 可能的问题

### 问题1：参数名称不匹配

**配置中的参数**：
```yaml
visualization:
  kpt_radius: 5
  kpt_line: true
  labels: true
  boxes: true
  masks: true
  probs: true
  line_width: 3
  font_size: 12      # ⚠️ 这个参数可能不被plot()识别
  color_mode: "class"
```

**plot()支持的参数**（根据文档）：
- ✅ `kpt_radius` - 支持
- ✅ `kpt_line` - 支持
- ✅ `labels` - 支持
- ✅ `boxes` - 支持
- ✅ `masks` - 支持
- ✅ `probs` - 支持
- ✅ `line_width` - 支持
- ❓ `font_size` - **可能不支持**（文档中没有列出）
- ❓ `color_mode` - **可能不支持**（文档中列出的格式不同）

**影响**：传入不支持的参数可能导致 plot() 方法行为异常，但通常不会导致坐标偏移。

---

### 问题2：img参数未传入

根据文档，`plot()` 方法有一个 `img` 参数：

> **img** (`np.ndarray`): 用于绘制图像的替代图像。如果未提供，则使用原始图像。

**关键点**：
- ✅ 如果未提供 `img` 参数，plot() 会使用 `Results` 对象内部保存的原始图像
- ✅ 这应该是正确的做法，因为 YOLO 会自动处理坐标映射
- ⚠️ **但如果 `Results` 对象中的图像被修改过，可能导致坐标映射错误**

---

### 问题3：图像预处理影响

如果传入 `plot()` 的图像与推理时使用的图像不一致，会导致坐标映射错误。

**检查点**：
1. ✅ 推理时直接传入原始图像路径或图像对象
2. ✅ 没有对图像进行额外的resize或裁剪
3. ✅ plot() 使用 Results 对象内部保存的图像

---

## 🛠️ 解决方案

### 方案1：确保plot()使用正确的参数 ✅

**修改代码**，只传入 plot() 支持的参数：

```python
def _create_visualization(self) -> Optional[np.ndarray]:
    """创建可视化图像"""
    try:
        if hasattr(self.raw_result, 'plot'):
            # 只传入plot()支持的参数
            plot_params = {
                'kpt_radius': self.visualization_params.get('kpt_radius', 5),
                'kpt_line': self.visualization_params.get('kpt_line', True),
                'labels': self.visualization_params.get('labels', True),
                'boxes': self.visualization_params.get('boxes', True),
                'masks': self.visualization_params.get('masks', True),
                'probs': self.visualization_params.get('probs', True),
                'line_width': self.visualization_params.get('line_width', 3),
                # 不传入可能不支持的参数（font_size, color_mode）
            }
            
            # 确保不传入None值
            plot_params = {k: v for k, v in plot_params.items() if v is not None}
            
            return self.raw_result.plot(**plot_params)
        else:
            logger.warning("检测结果不支持可视化")
            return None
    except Exception as e:
        logger.error(f"创建可视化失败: {e}")
        return None
```

### 方案2：显式传入原始图像（如果需要）

如果发现坐标映射有问题，可以尝试显式传入原始图像：

```python
def _create_visualization(self, original_image: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """创建可视化图像"""
    try:
        if hasattr(self.raw_result, 'plot'):
            plot_params = {
                'kpt_radius': self.visualization_params.get('kpt_radius', 5),
                'kpt_line': self.visualization_params.get('kpt_line', True),
                'labels': self.visualization_params.get('labels', True),
                'boxes': self.visualization_params.get('boxes', True),
                'line_width': self.visualization_params.get('line_width', 3),
            }
            
            # 如果提供了原始图像，显式传入
            if original_image is not None:
                plot_params['img'] = original_image
            
            return self.raw_result.plot(**plot_params)
```

**注意**：通常不需要这样做，因为 YOLO 会自动处理。

---

## 🧪 验证方法

### 测试1：检查plot()返回的图像尺寸

```python
result = detector.detect(image)
visualization = result.get_visualization()

# 检查可视化图像的尺寸是否与原始图像一致
print(f"原始图像尺寸: {image.size}")
print(f"可视化图像尺寸: {visualization.shape[:2]}")
# 应该一致，如果不一致说明有问题
```

### 测试2：直接使用YOLO的plot()验证

```python
from ultralytics import YOLO

model = YOLO('your_model.pt')
image = 'test_image.jpg'

# 直接使用YOLO的plot()
results = model.predict(image, imgsz=640)
annotated = results[0].plot()  # 不使用任何额外参数

# 检查关键点位置是否正确
```

### 测试3：比较不同参数的结果

```python
# 测试1：使用所有参数
result1 = results[0].plot(
    kpt_radius=5,
    kpt_line=True,
    labels=True,
    boxes=True,
    line_width=3
)

# 测试2：只使用默认参数
result2 = results[0].plot()

# 比较结果是否一致
```

---

## 🔍 可能的原因总结

根据官方文档和代码分析，关节点位置偏移的**最可能原因**：

### 1. 训练与推理时的imgsz不一致 ⚠️ **最可能**

- **症状**：所有关键点按比例偏移
- **解决**：确保训练和推理时使用相同的 `imgsz`

### 2. plot()参数问题 ⚠️

- **症状**：可视化显示偏移，但实际坐标可能正确
- **解决**：只传入plot()支持的参数

### 3. YOLO内部坐标映射问题（罕见）⚠️

- **症状**：坐标本身就有问题
- **解决**：更新ultralytics库到最新版本

---

## ✅ 推荐的检查步骤

1. ✅ **首先确认**：训练和推理时的 `imgsz` 是否一致
2. ✅ **检查参数**：确认传入plot()的参数都支持
3. ✅ **测试验证**：运行测试脚本检查坐标是否在图像范围内
4. ✅ **对比测试**：直接使用YOLO的plot()方法，不使用额外参数

---

## 📝 下一步行动

1. ✅ 清理plot()参数，只传入支持的参数
2. ✅ 添加调试代码，检查坐标映射是否正确
3. ✅ 运行测试脚本，验证关键点坐标

---

**参考文档**：[Ultralytics YOLO Predict Mode](https://docs.ultralytics.com/zh/modes/predict/#key-features-of-predict-mode)
