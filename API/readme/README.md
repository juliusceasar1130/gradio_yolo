# YOLO 图像检测与MinIO存储系统

这个项目使用YOLOv11进行图像检测，并将检测结果存储在MinIO对象存储系统中。

## 功能特点

- 使用YOLO进行图像目标检测
- 将检测结果保存到MySQL数据库
- 使用MinIO对象存储系统保存检测图像
- 按年/月/日的目录结构组织图像文件
- 提供RESTful API接口

## 环境要求

- Python 3.8+
- MinIO服务器
- MySQL数据库

## 安装步骤

1. 安装依赖包：

```bash
pip install -r requirements.txt
```

2. 设置MinIO服务器：

- 下载并安装MinIO：https://min.io/download
- 启动MinIO服务器：

```bash
# Windows
minio.exe server D:\minio-data

# Linux/Mac
minio server /data
```

- 默认访问凭证：
  - 访问密钥: minioadmin
  - 秘密密钥: minioadmin
  - 端口: 9000

3. 配置数据库：

- 确保MySQL服务已启动
- 创建名为`skid`的数据库
- 在数据库中创建`skid_cls`表：

```sql
CREATE TABLE skid_cls (
  id INT AUTO_INCREMENT PRIMARY KEY,
  skid_nr VARCHAR(50) NOT NULL,
  dateTime DATETIME NOT NULL,
  result_cls VARCHAR(50) NOT NULL
);
```

## 配置说明

在`API/db_utils.py`文件中，可以修改以下配置：

1. 数据库配置：

```python
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "skid",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}
```

2. MinIO配置：

```python
MINIO_CONFIG = {
    "endpoint": "127.0.0.1:9000",  # MinIO服务器地址
    "access_key": "minioadmin",    # 访问密钥
    "secret_key": "minioadmin",    # 秘密密钥
    "secure": False                # 是否使用HTTPS
}
```

## 运行应用

启动Flask API服务器：

```bash
python API/service.py --port 5000
```

## API使用说明

发送POST请求到`/test`端点，带有以下参数：

- 请求体：图像二进制数据
- 请求头：
  - `skid-number`: 雪橇号码（可选）
  - `date`: 日期（可选）

响应包含以下信息：

- 响应体：处理后的图像数据
- 响应头：
  - `X-Status-Code`: 状态码
  - `X-Status-Message`: 状态消息
  - `X-Detected-Class`: 检测到的类别

## 存储结构

图像按照以下结构存储在MinIO中：

```
skidcls/
  ├── 年份/
  │   ├── 月份/
  │   │   ├── 日期/
  │   │   │   ├── 图像文件1.jpg
  │   │   │   ├── 图像文件2.jpg
  │   │   │   └── ...
``` 