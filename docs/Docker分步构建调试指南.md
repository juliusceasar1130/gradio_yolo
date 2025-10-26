# Docker分步构建调试指南

**创建者/修改者**: chenliang  
**修改时间**: 2025年8月2日  
**主要修改内容**: 创建Docker构建问题分步调试完整指南

## 📋 问题背景

在Docker构建过程中遇到了Bus error错误，主要表现为：
- `gradio==5.31.0`版本不存在导致安装失败
- 大型依赖包安装时出现内存不足
- `/tmp`目录I/O错误和Bus error (core dumped)

## 🧹 环境清理指令

### 🎯 针对此案例的推荐清理命令

基于Bus error和I/O错误的具体情况，建议按以下顺序执行：

#### 第1步：基础清理（推荐先执行）
```bash
# 清理Docker构建缓存和系统缓存
docker builder prune -a -f
docker system prune -a -f

# 检查清理效果
docker system df
```

#### 第2步：如果问题仍存在，执行深度清理
```bash
# 停止所有运行的容器（Windows CMD兼容）
for /f "tokens=*" %i in ('docker ps -aq') do docker stop %i 2>nul

# 删除所有容器
for /f "tokens=*" %i in ('docker ps -aq') do docker rm %i 2>nul

# 删除所有镜像
for /f "tokens=*" %i in ('docker images -q') do docker rmi %i -f 2>nul

# 再次清理系统缓存
docker system prune -a -f
```

#### 第3步：针对Bus Error的特殊清理
```bash
# 清理可能损坏的构建缓存
docker builder prune -a -f

# 清理可能有问题的镜像层
docker system prune -a -f

# 清理卷（可能包含损坏的临时文件）
docker volume prune -f

# 检查清理结果
docker system df
```

### 完整清理命令（如果以上方法都无效）
```bash
# 停止所有运行的容器
docker stop $(docker ps -aq)

# 删除所有容器
docker rm $(docker ps -aq)

# 删除所有镜像
docker rmi $(docker images -q) -f

# 清理所有Docker缓存和数据
docker system prune -a -f

# 清理构建缓存
docker builder prune -a -f

# 清理网络
docker network prune -f

# 清理卷
docker volume prune -f

# 检查清理结果
docker system df
```

### 分步清理命令
```bash
# 1. 清理构建缓存
docker builder prune -a -f

# 2. 清理系统缓存
docker system prune -a -f

# 3. 清理未使用的镜像
docker image prune -a -f

# 4. 清理未使用的容器
docker container prune -f

# 5. 检查磁盘使用情况
docker system df
```

## 🔧 分步构建调试文件

### 1. 基础依赖文件 (requirements-base.txt)
```txt
# 基础依赖 - 分步安装调试用（严格按照指定版本）
gradio==5.31.0
numpy==2.2.6
Pillow==11.3.0
PyYAML==6.0.2
```

### 2. 机器学习依赖文件 (requirements-ml.txt)
```txt
# 机器学习依赖 - 分步安装调试用（严格按照指定版本）
torch==2.7.1
torchvision==0.22.1
ultralytics==8.3.161
opencv-python==4.11.0.86
pandas==2.3.0
```

### 3. 分步调试Dockerfile (Dockerfile.debug)
```dockerfile
# 分步构建调试用Dockerfile
FROM python:3.10-slim as base

WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN pip install --upgrade pip

# 阶段1: 基础依赖
FROM base as base-deps
COPY requirements-base.txt .
RUN echo "=== 安装基础依赖 ===" && \
    pip install --no-cache-dir -r requirements-base.txt && \
    echo "=== 基础依赖安装完成 ==="

# 阶段2: 机器学习依赖
FROM base-deps as ml-deps
COPY requirements-ml.txt .
RUN echo "=== 安装机器学习依赖 ===" && \
    pip install --no-cache-dir -r requirements-ml.txt && \
    echo "=== 机器学习依赖安装完成 ==="

# 阶段3: 测试依赖
FROM ml-deps as test-deps
RUN echo "=== 安装测试依赖 ===" && \
    pip install pytest==8.4.1 pytest-cov==6.2.1 && \
    echo "=== 测试依赖安装完成 ==="

# 最终阶段
FROM test-deps as final
COPY . .
RUN mkdir -p /app/logs /app/outputs /app/models /app/uploads /app/test_images
EXPOSE 7860
CMD ["python", "main.py", "web"]
```

## 🚀 分步测试详细步骤

### 第1步：环境清理

#### 🎯 你应该执行的命令（推荐）：
```bash
# 基础清理（推荐先执行这个）
docker builder prune -a -f
docker system prune -a -f

# 检查清理效果
docker system df

# 确认当前目录
pwd
# 应该在 D:\00deeplearn\yolo11\gradi_yolo
```

#### 如果基础清理无效，执行深度清理：
```bash
# Windows CMD版本（如果你使用CMD）
for /f "tokens=*" %i in ('docker ps -aq') do docker stop %i 2>nul
for /f "tokens=*" %i in ('docker ps -aq') do docker rm %i 2>nul
for /f "tokens=*" %i in ('docker images -q') do docker rmi %i -f 2>nul
docker system prune -a -f

# PowerShell版本（如果你使用PowerShell）
docker stop $(docker ps -aq) 2>$null
docker rm $(docker ps -aq) 2>$null
docker rmi $(docker images -q) -f 2>$null
docker system prune -a -f
```

### 第2步：测试基础环境（不安装Python包）
```bash
# 构建基础环境，只安装系统依赖
docker build -f Dockerfile.debug --target base -t yolo-base . --no-cache

# 如果成功，验证基础环境
docker run --rm yolo-base python --version
docker run --rm yolo-base pip --version
```

### 第3步：测试基础Python依赖
```bash
# 构建基础依赖阶段
docker build -f Dockerfile.debug --target base-deps -t yolo-base-deps . --no-cache

# 如果失败，可以单独测试每个基础包
echo "gradio==5.31.0" > test-gradio.txt
echo "numpy==2.2.6" > test-numpy.txt  
echo "Pillow==11.3.0" > test-pillow.txt
echo "PyYAML==6.0.2" > test-yaml.txt

# 单独测试每个包（Windows PowerShell）
docker run --rm -v ${PWD}:/app python:3.10-slim bash -c "cd /app && pip install gradio==5.31.0"
docker run --rm -v ${PWD}:/app python:3.10-slim bash -c "cd /app && pip install numpy==2.2.6"
docker run --rm -v ${PWD}:/app python:3.10-slim bash -c "cd /app && pip install Pillow==11.3.0"
docker run --rm -v ${PWD}:/app python:3.10-slim bash -c "cd /app && pip install PyYAML==6.0.2"
```

### 第4步：测试机器学习依赖
```bash
# 构建机器学习依赖阶段
docker build -f Dockerfile.debug --target ml-deps -t yolo-ml-deps . --no-cache

# 如果失败，单独测试每个ML包
docker run --rm -v ${PWD}:/app python:3.10-slim bash -c "cd /app && pip install torch==2.7.1"
docker run --rm -v ${PWD}:/app python:3.10-slim bash -c "cd /app && pip install torchvision==0.22.1"  
docker run --rm -v ${PWD}:/app python:3.10-slim bash -c "cd /app && pip install ultralytics==8.3.161"
docker run --rm -v ${PWD}:/app python:3.10-slim bash -c "cd /app && pip install opencv-python==4.11.0.86"
docker run --rm -v ${PWD}:/app python:3.10-slim bash -c "cd /app && pip install pandas==2.3.0"
```

### 第5步：测试测试依赖
```bash
# 构建测试依赖阶段
docker build -f Dockerfile.debug --target test-deps -t yolo-test-deps . --no-cache

# 单独测试
docker run --rm python:3.10-slim pip install pytest==8.4.1
docker run --rm python:3.10-slim pip install pytest-cov==6.2.1
```

### 第6步：完整构建测试
```bash
# 完整构建
docker build -f Dockerfile.debug --target final -t yolo-debug . --no-cache

# 如果成功，测试运行
docker run --rm -p 7860:7860 yolo-debug
```

## 🔍 问题排查指南

### 如果第3步失败（基础依赖）

#### gradio==5.31.0 失败
```bash
# 检查可用版本
docker run --rm python:3.10-slim pip index versions gradio

# 如果版本不存在，尝试最新稳定版本
docker run --rm python:3.10-slim pip install gradio --dry-run

# 解决方案：使用存在的版本
# 将 gradio==5.31.0 改为 gradio==4.44.1
```

#### numpy==2.2.6 失败
```bash
# 检查可用版本
docker run --rm python:3.10-slim pip index versions numpy

# 测试内存限制安装
docker run --rm -m 2g python:3.10-slim pip install numpy==2.2.6

# 如果内存不足，尝试更小的版本
docker run --rm python:3.10-slim pip install numpy==1.24.3
```

### 如果第4步失败（ML依赖）

#### torch/torchvision 失败
```bash
# 检查PyTorch版本兼容性
docker run --rm python:3.10-slim pip index versions torch
docker run --rm python:3.10-slim pip index versions torchvision

# 尝试CPU版本
docker run --rm python:3.10-slim pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 如果版本不存在，使用兼容版本
docker run --rm python:3.10-slim pip install torch==2.0.1 torchvision==0.15.2
```

#### ultralytics 失败
```bash
# 检查版本
docker run --rm python:3.10-slim pip index versions ultralytics

# 先安装依赖再安装ultralytics
docker run --rm python:3.10-slim bash -c "pip install torch torchvision && pip install ultralytics==8.0.196"
```

### Bus Error 专项排查

#### 内存问题排查
```bash
# 限制内存测试
docker run --rm -m 1g python:3.10-slim pip install torch==2.7.1
docker run --rm -m 2g python:3.10-slim pip install torch==2.7.1
docker run --rm -m 4g python:3.10-slim pip install torch==2.7.1

# 检查系统资源（Windows）
docker system df
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory

# 检查系统资源（Linux）
free -h
df -h
```

#### 磁盘空间问题排查
```bash
# 检查Docker磁盘使用
docker system df

# 检查临时目录空间
docker run --rm python:3.10-slim df -h /tmp

# 清理并重试
docker system prune -a -f
```

## 📊 日志收集命令

### 详细构建日志
```bash
# 保存构建日志
docker build -f Dockerfile.debug --target base-deps -t yolo-base-deps . --no-cache --progress=plain > build-base.log 2>&1

docker build -f Dockerfile.debug --target ml-deps -t yolo-ml-deps . --no-cache --progress=plain > build-ml.log 2>&1
```

### 系统信息收集
```bash
# Docker信息
docker version > system-info.txt
docker system info >> system-info.txt

# 系统资源（Windows）
echo "=== Disk Space ===" >> system-info.txt
wmic logicaldisk get size,freespace,caption >> system-info.txt
echo "=== Memory Info ===" >> system-info.txt
wmic computersystem get TotalPhysicalMemory >> system-info.txt
```

## ⚡ 快速解决方案

### 方案1：使用兼容版本
创建 `requirements-compatible.txt`：
```txt
# 兼容版本依赖
gradio==4.44.1
ultralytics==8.0.196
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.1.78
Pillow==10.0.1
numpy==1.24.3
pandas==2.0.3
PyYAML==6.0.2
pytest==8.4.1
pytest-cov==6.2.1
```

### 方案2：分阶段安装
```dockerfile
# 分阶段安装Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 基础环境
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 升级pip
RUN pip install --upgrade pip

# 第一阶段：基础包
RUN pip install --no-cache-dir PyYAML==6.0.2 Pillow==10.0.1

# 第二阶段：数值计算
RUN pip install --no-cache-dir numpy==1.24.3 pandas==2.0.3

# 第三阶段：Web框架
RUN pip install --no-cache-dir gradio==4.44.1

# 第四阶段：深度学习
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2

# 第五阶段：YOLO
RUN pip install --no-cache-dir ultralytics==8.0.196

# 第六阶段：图像处理
RUN pip install --no-cache-dir opencv-python==4.8.1.78

# 复制项目文件
COPY . .
RUN mkdir -p /app/logs /app/outputs /app/models /app/uploads /app/test_images

EXPOSE 7860
CMD ["python", "main.py", "web"]
```

## 🎯 执行建议

### 🚀 立即行动指南（针对你的案例）

1. **首先执行基础清理**：
   ```bash
   docker builder prune -a -f
   docker system prune -a -f
   docker system df
   ```

2. **如果清理后空间仍不足，执行深度清理**：
   ```bash
   # 选择适合你的Shell版本执行
   # CMD版本或PowerShell版本（见上文）
   ```

3. **清理完成后，开始分步测试**：
   - 从第2步开始：测试基础环境
   - 逐步进行到第6步：完整构建测试

4. **如果遇到gradio==5.31.0版本问题**：
   - 立即使用兼容版本：gradio==4.44.1

### 详细执行建议

### 1. 按顺序执行
严格按照步骤1-6的顺序执行，每一步成功后再进行下一步。

### 2. 记录失败点
详细记录在哪一步失败，保存完整的错误信息和日志。

### 3. 保存日志
使用提供的日志收集命令保存详细的构建信息。

### 4. 单包测试
如果某个阶段失败，使用单包测试命令定位具体问题包。

### 5. 资源监控
执行过程中监控系统内存和磁盘使用情况。

### 6. 版本兼容性
如果指定版本不存在或有问题，使用兼容版本进行测试。

## 📝 常见问题解决

### 问题1：gradio版本不存在
**解决方案**：使用 `gradio==4.44.1` 替代 `gradio==5.31.0`

### 问题2：内存不足导致Bus error
**解决方案**：
- 增加Docker内存限制
- 使用更小的依赖版本
- 分阶段安装大型包

### 问题3：磁盘空间不足
**解决方案**：
- 清理Docker缓存
- 清理系统临时文件
- 使用更小的基础镜像

### 问题4：网络超时
**解决方案**：
- 使用国内镜像源
- 增加pip超时时间
- 使用 `--timeout 300` 参数

## 🔚 总结

通过分步构建调试，可以精确定位Docker构建失败的具体原因，然后针对性地解决问题。关键是要有耐心，按步骤执行，详细记录每一步的结果，这样才能快速找到问题的根源并解决。