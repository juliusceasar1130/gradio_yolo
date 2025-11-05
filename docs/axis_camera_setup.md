# Axis 摄像头配置指南

## 概述

本指南帮助您正确配置 Axis 网络摄像头，以便与 YOLO 检测系统配合使用。

## 前置要求

1. **硬件要求**
   - Axis 网络摄像头（支持 MJPEG 流）
   - 确保摄像头和计算机在同一个网络中

2. **软件要求**
   - 摄像头固件版本支持 HTTP/HTTPS 流
   - 网络端口 80 可用

## 步骤 1: 初始设置

### 1.1 查找摄像头 IP 地址

```bash
# 方法 1: 使用网络扫描工具
nmap -sn 192.168.1.0/24 | grep Axis

# 方法 2: 使用 Axis 官方工具
# 下载并安装 "Axis IP Utility" 或 "AXIS Camera Management"

# 方法 3: 登录路由器查看 DHCP 客户端列表
```

### 1.2 首次访问摄像头

1. 打开浏览器，访问：`http://YOUR_CAMERA_IP`
   - 例如：`http://192.168.39.253`

2. 使用默认凭据登录：
   - **管理员**：root / root（较新固件）
   - **管理员**：admin / pass（较老固件）

3. 如果无法登录，尝试恢复出厂设置：
   - 长按摄像头上的重置按钮 10-15 秒

## 步骤 2: 网络配置

### 2.1 设置固定 IP 地址

1. 登录摄像头管理界面
2. 进入 **Settings > Network > TCP/IP**
3. 设置静态 IP：
   ```
   IP Address: 192.168.39.253
   Subnet Mask: 255.255.255.0
   Gateway: 192.168.39.1
   DNS: 192.168.39.1
   ```

### 2.2 验证网络连通性

```bash
# 从计算机测试连接
ping 192.168.39.253

# 测试 HTTP 端口
telnet 192.168.39.253 80
```

## 步骤 3: 配置用户账户

### 3.1 创建专用用户（推荐）

1. 进入 **Settings > System > Users**
2. 点击 **Add User**
3. 设置用户信息：
   ```
   User Name: camera_user
   Password: your_secure_password
   Role: Operator（足够权限）
   ```

### 3.2 禁用匿名访问（安全）

1. 进入 **Settings > System > Security**
2. 禁用 **Anonymous Login**

## 步骤 4: 配置视频流

### 4.1 启用 MJPEG 流

1. 进入 **Settings > Video > Stream Profiles**
2. 创建一个新的配置文件：
   ```
   Profile Name: MJPEG_Stream
   Image Settings:
   - Resolution: 640x480 或 1280x720
   - Frame Rate: 25 fps
   ```

3. 启用 MJPEG：
   - **Settings > Video > Video Streams**
   - 选择 **MJPEG** 编码

### 4.2 配置 CGI 访问

1. 进入 **Settings > System > Options**
2. 启用 **Basic CGI**（如果需要 Basic 认证）
3. 启用 **CGI Applications**

### 4.3 测试 MJPEG 流

在浏览器中访问：
```
http://192.168.39.253/axis-cgi/mjpg/video.cgi
http://192.168.39.253/axis-cgi/mjpg/video.cgi?resolution=640x480&fps=25
```

如果需要认证，系统会弹出登录框。

## 步骤 5: 常见 URL 格式

根据摄像头配置，可以使用以下 URL 格式：

### 5.1 无认证（匿名访问）
```http
http://192.168.39.253/axis-cgi/mjpg/video.cgi
http://192.168.39.253/mjpg/video.mjpg
```

### 5.2 Basic 认证
```http
http://root:password@192.168.39.253/axis-cgi/mjpg/video.cgi
http://admin:password@192.168.39.253/axis-cgi/mjpg/video.cgi
```

### 5.3 带参数
```http
http://192.168.39.253/axis-cgi/mjpg/video.cgi?resolution=640x480&fps=25
http://192.168.39.253/axis-cgi/mjpg/video.cgi?resolution=1280x720&fps=15
```

### 5.4 HTTPS（安全）
```http
https://192.168.39.253/axis-cgi/mjpg/video.cgi
```

## 步骤 6: 测试连接

### 6.1 使用浏览器测试

1. 打开新标签页
2. 访问摄像头 MJPEG URL
3. 应该能看到实时视频流

### 6.2 使用 curl 测试

```bash
# 测试无认证
curl -I http://192.168.39.253/axis-cgi/mjpg/video.cgi

# 测试 Basic 认证
curl -u root:password http://192.168.39.253/axis-cgi/mjpg/video.cgi
```

### 6.3 使用我们的诊断工具

在 Python 中使用：

```python
from src.yolo_detector.core.camera_capture import CameraCapture

# 创建摄像头对象
camera = CameraCapture(
    camera_type="axis",
    axis_ip="192.168.39.253",
    axis_username="root",
    axis_password="your_password"
)

# 运行连接测试
test_results = camera.test_connection()
print(json.dumps(test_results, indent=2, ensure_ascii=False))
```

## 常见问题解决

### 问题 1: "Error number -138" - 连接超时

**原因**：
- 摄像头 IP 地址错误
- 网络不通
- 防火墙阻止

**解决方案**：
1. 验证 IP 地址：`ping 192.168.39.253`
2. 检查网络连接
3. 临时禁用防火墙测试

### 问题 2: "401 Unauthorized" - 认证失败

**原因**：
- 用户名或密码错误
- 禁用匿名访问

**解决方案**：
1. 确认用户名和密码正确
2. 检查摄像头用户权限
3. 尝试恢复出厂设置

### 问题 3: "404 Not Found" - URL 路径错误

**原因**：
- 摄像头固件不支持该路径
- CGI 服务未启用

**解决方案**：
1. 检查摄像头固件版本
2. 启用 CGI 应用（设置 > 系统 > 选项）
3. 尝试其他 URL 格式

### 问题 4: "Connection refused" - 端口关闭

**原因**：
- HTTP 服务未启动
- 端口配置错误

**解决方案**：
1. 重启摄像头
2. 检查网络设置
3. 确认端口 80 可用

## 性能优化建议

### 1. 调整分辨率和帧率

根据需要平衡质量和性能：
- **高清检测**：1280x720 @ 15fps
- **实时预览**：640x480 @ 25fps
- **低延迟**：480x360 @ 30fps

### 2. 使用有线连接

相比 WiFi，有线以太网连接更稳定，延迟更低。

### 3. 配置 QoS

在路由器上设置摄像头流量为高优先级。

## 安全注意事项

1. **更改默认密码**：务必修改默认凭据
2. **使用 HTTPS**：生产环境中启用 SSL/TLS
3. **限制访问**：配置防火墙，只允许必要端口
4. **定期更新**：保持固件版本最新
5. **监控日志**：定期检查摄像头访问日志

## 故障排除清单

- [ ] IP 地址可 ping 通
- [ ] HTTP 端口 80 可访问
- [ ] 用户名和密码正确
- [ ] MJPEG 流 URL 可在浏览器中访问
- [ ] 防火墙未阻止连接
- [ ] 摄像头固件是最新版本
- [ ] CGI 服务已启用

## 支持资源

- [Axis 官方文档](https://www.axis.com/documentation)
- [Axis 开发者指南](https://www.axis.com/developer-community)
- [摄像头型号特定手册](https://www.axis.com/products)

## 修改历史

- 2025-11-04：初始版本，添加多 URL 格式支持
