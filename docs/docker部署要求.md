# Docker 部署指南

**创建者/修改者**: chenliang  
**修改时间**: 2025年8月2日11点32分  
**主要修改内容**: 完善Docker部署文档，增加常用命令和故障排除

## 📋 部署要求
- ✅ 生成dockerfile文件
- ✅ 程序入口为 main.py
- ✅ 端口配置：7860
- ✅ Docker专用配置文件：configs/docker.yaml

## 🚀 快速开始

### 方式一：使用 Docker Compose（强烈推荐）

```bash
# 构建并启动服务
docker-compose up -d

# 查看启动日志
docker-compose logs -f yolo-detector

# 停止服务
docker-compose down
```

### 方式二：使用 Docker 命令

```bash
# 构建镜像
docker build -t yolo-detector .

# 运行容器
docker run -d \
  --name yolo-detector-app \
  -p 7860:7860 \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  yolo-detector
```

## 🌐 访问应用

启动成功后，访问：http://localhost:7860

## 📁 目录挂载说明

- `./configs` → `/app/configs` - 配置文件目录
- `./outputs` → `/app/outputs` - 检测结果输出目录  
- `./logs` → `/app/logs` - 日志文件目录
- `./models` → `/app/models` - 模型文件目录
- `./uploads` → `/app/uploads` - 上传文件目录（自动创建）

## 🎯 模型文件配置

### 1. 创建模型目录
```bash
mkdir models
```

### 2. 复制模型文件到项目目录
将您的YOLO模型文件复制到 `models/` 目录下：
```
models/
├── detection_best.pt      # 目标检测模型
├── segmentation_best.pt   # 图像分割模型
└── classification_best.pt # 图像分类模型（可选）
```

### 3. 使用Docker专用配置
Docker环境会自动使用 `configs/docker.yaml` 配置文件，该文件包含容器内的正确路径：
- 检测模型：`/app/models/detection_best.pt`
- 分割模型：`/app/models/segmentation_best.pt`
- 分类模型：`/app/models/classification_best.pt`

### 4. 模型文件来源
将以下文件复制到对应位置：
- 检测模型：`D:/00deeplearn/yolo11/【2】训练模型/cls/train_new/weights/best.pt` → `models/detection_best.pt`
- 分割模型：`D:/00deeplearn/yolo11/【2】训练模型/seg/train2_new/weights/best.pt` → `models/segmentation_best.pt`  
- 分类模型：`E:/00learning/00test/cls_train/train12/weights/best.pt` → `models/classification_best.pt`

## ⚙️ GPU 支持（可选）

如需GPU支持，请：
1. 安装 nvidia-docker2
2. 取消注释 docker-compose.yml 中的 GPU 配置
3. 使用支持CUDA的基础镜像

## 🔧 常用管理命令

### 服务管理
```bash
# 查看服务状态
docker-compose ps

# 启动服务
docker-compose up -d

# 停止服务
docker-compose stop

# 重启服务
docker-compose restart

# 停止并删除容器
docker-compose down

# 强制重建并启动
docker-compose up -d --build
```

### 日志查看
```bash
# 查看实时日志
docker-compose logs -f

# 查看指定服务日志
docker-compose logs -f yolo-detector

# 查看最近100行日志
docker-compose logs --tail=100 yolo-detector

# 查看特定时间的日志
docker-compose logs --since="2025-08-02T10:00:00" yolo-detector
```

### 容器操作
```bash
# 进入容器
docker-compose exec yolo-detector bash

# 在容器中执行命令
docker-compose exec yolo-detector python main.py info

# 查看容器资源使用
docker stats yolo-detector-app

# 查看容器详细信息
docker inspect yolo-detector-app
```

### 镜像管理
```bash
# 查看镜像
docker images

# 删除旧镜像
docker image prune

# 强制重建镜像
docker-compose build --no-cache

# 删除指定镜像
docker rmi yolo-detector
```

### 数据管理
```bash
# 备份输出目录
tar -czf outputs_backup_$(date +%Y%m%d).tar.gz outputs/

# 清理输出目录
rm -rf outputs/*

# 查看挂载卷
docker volume ls

# 清理未使用的卷
docker volume prune
```

## 🔍 故障排除

### 常见问题

1. **端口被占用**
```bash
# 查看端口占用
netstat -tlnp | grep 7860
# 或使用
lsof -i :7860

# 修改端口（在docker-compose.yml中）
ports:
  - "7861:7860"  # 改为其他端口
```

2. **模型文件找不到**
```bash
# 检查模型文件是否存在
ls -la models/

# 检查容器内模型文件
docker-compose exec yolo-detector ls -la /app/models/
```

3. **内存不足**
```bash
# 查看系统资源
free -h
df -h

# 限制容器内存使用（在docker-compose.yml中）
deploy:
  resources:
    limits:
      memory: 2G
```

4. **权限问题**
```bash
# 修复目录权限
sudo chown -R $USER:$USER outputs/ logs/ models/
chmod -R 755 outputs/ logs/ models/
```

### 调试命令
```bash
# 查看容器启动过程
docker-compose up

# 进入容器调试
docker-compose exec yolo-detector bash

# 检查Python环境
docker-compose exec yolo-detector python --version
docker-compose exec yolo-detector pip list

# 测试模型加载
docker-compose exec yolo-detector python -c "
from src.yolo_detector import Config, ModelLoader
config = Config()
loader = ModelLoader(config)
print(loader.list_available_models())
"
```

## 📊 性能监控

```bash
# 实时监控容器资源
docker stats yolo-detector-app

# 查看容器进程
docker-compose exec yolo-detector ps aux

# 监控日志大小
du -sh logs/

# 监控输出目录大小
du -sh outputs/
```