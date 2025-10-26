# Docker环境清理指令快速参考

**创建者/修改者**: chenliang  
**修改时间**: 2025年8月2日  
**主要修改内容**: Docker环境清理指令快速参考手册

## 🧹 完整清理指令（一键清理）

```bash
# 停止所有运行的容器
docker stop $(docker ps -aq) 2>/dev/null || true

# 删除所有容器
docker rm $(docker ps -aq) 2>/dev/null || true

# 删除所有镜像
docker rmi $(docker images -q) -f 2>/dev/null || true

# 清理所有系统缓存
docker system prune -a -f

# 清理构建缓存
docker builder prune -a -f

# 清理网络
docker network prune -f

# 清理卷
docker volume prune -f

# 检查清理结果
echo "=== 清理完成，检查剩余资源 ==="
docker system df
```

## 🔧 分步清理指令

### 1. 基础清理
```bash
# 清理构建缓存
docker builder prune -a -f

# 清理系统缓存（包括未使用的镜像、容器、网络）
docker system prune -a -f
```

### 2. 深度清理
```bash
# 停止所有容器
docker stop $(docker ps -aq)

# 删除所有容器
docker rm $(docker ps -aq)

# 删除所有镜像
docker rmi $(docker images -q) -f

# 清理所有卷
docker volume prune -f

# 清理所有网络
docker network prune -f
```

### 3. 检查清理结果
```bash
# 查看Docker磁盘使用情况
docker system df

# 查看剩余镜像
docker images

# 查看剩余容器
docker ps -a

# 查看剩余卷
docker volume ls

# 查看剩余网络
docker network ls
```

## ⚠️ 安全清理指令（保留重要资源）

### 只清理构建缓存
```bash
# 只清理构建缓存，不删除镜像和容器
docker builder prune -f
```

### 只清理未使用的资源
```bash
# 只清理未使用的镜像、容器、网络，保留正在使用的
docker system prune -f
```

### 选择性清理
```bash
# 只删除停止的容器
docker container prune -f

# 只删除未使用的镜像
docker image prune -a -f

# 只删除未使用的卷
docker volume prune -f

# 只删除未使用的网络
docker network prune -f
```

## 🎯 针对特定问题的清理

### Bus Error 问题清理
```bash
# 清理可能损坏的构建缓存
docker builder prune -a -f

# 清理可能有问题的镜像层
docker system prune -a -f

# 重启Docker服务（Windows Docker Desktop）
# 在Docker Desktop中点击 "Restart Docker"

# 或使用命令行重启（Linux）
# sudo systemctl restart docker
```

### 磁盘空间不足清理
```bash
# 清理所有未使用的资源
docker system prune -a -f --volumes

# 检查清理效果
docker system df

# 如果还不够，删除所有镜像重新下载
docker rmi $(docker images -q) -f
```

### 内存问题清理
```bash
# 停止所有容器释放内存
docker stop $(docker ps -aq)

# 清理系统缓存
docker system prune -a -f

# 检查系统内存使用（Windows）
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory

# 检查系统内存使用（Linux）
free -h
```

## 📊 清理前后对比检查

### 清理前检查
```bash
echo "=== 清理前状态 ==="
docker system df
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Size}}"
```

### 清理后检查
```bash
echo "=== 清理后状态 ==="
docker system df
docker images
docker ps -a
docker volume ls
docker network ls
```

## 🚨 Windows 特殊清理指令

### Docker Desktop 完全重置
```bash
# 在 Docker Desktop 设置中选择 "Reset to factory defaults"
# 或者删除 Docker Desktop 数据目录
# %APPDATA%\Docker Desktop
```

### 清理 Docker Desktop 缓存
```bash
# 清理 Docker Desktop 的缓存目录
# %LOCALAPPDATA%\Docker\wsl\data\ext4.vhdx
```

## 🔄 自动化清理脚本

### Windows PowerShell 脚本
```powershell
# clean-docker.ps1
Write-Host "开始清理Docker环境..." -ForegroundColor Green

# 停止所有容器
docker stop $(docker ps -aq) 2>$null

# 删除所有容器
docker rm $(docker ps -aq) 2>$null

# 清理系统资源
docker system prune -a -f

# 清理构建缓存
docker builder prune -a -f

Write-Host "清理完成！" -ForegroundColor Green
docker system df
```

### Linux/Mac Bash 脚本
```bash
#!/bin/bash
# clean-docker.sh

echo "开始清理Docker环境..."

# 停止所有容器
docker stop $(docker ps -aq) 2>/dev/null || true

# 删除所有容器
docker rm $(docker ps -aq) 2>/dev/null || true

# 清理系统资源
docker system prune -a -f

# 清理构建缓存
docker builder prune -a -f

echo "清理完成！"
docker system df
```

## ⏰ 定期清理建议

### 每日清理
```bash
# 清理未使用的资源
docker system prune -f
```

### 每周清理
```bash
# 深度清理
docker system prune -a -f
docker builder prune -a -f
```

### 每月清理
```bash
# 完全清理重建
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker rmi $(docker images -q) -f
docker system prune -a -f --volumes
```

## 🎯 使用建议

1. **构建失败后立即清理**：每次遇到构建问题后，先执行基础清理
2. **定期维护**：建议每周执行一次深度清理
3. **监控磁盘空间**：经常检查 `docker system df` 的输出
4. **保留重要镜像**：清理前确认不会删除重要的生产镜像
5. **备份重要数据**：清理卷之前确保重要数据已备份

## ⚠️ 注意事项

- 完整清理会删除所有Docker资源，包括正在使用的
- 清理后需要重新下载基础镜像，可能需要较长时间
- 在生产环境中谨慎使用完整清理指令
- 清理前确保没有重要的容器正在运行
- 某些清理操作不可逆，请谨慎执行