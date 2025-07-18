"""
数据库工具模块 - 用于处理数据库和minio操作
"""

import pymysql
import os
import datetime
from logger_config import logger

# MinIO配置
import io
from minio import Minio
import uuid
import cv2
##########################################################
# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "skid",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}
host=DB_CONFIG.get("host")
db=DB_CONFIG.get("database")


# MinIO客户端配置
MINIO_CONFIG = {
    "endpoint": "127.0.0.1:9000",  # MinIO服务器地址
    "access_key": "minioadmin",    # 访问密钥
    "secret_key": "minioadmin",    # 秘密密钥
    "secure": False                # 是否使用HTTPS
}
# 桶名称
BUCKET_NAME = "skidcls"

##########################################################

def get_connection():
    """
    获取数据库连接
    
    返回:
        pymysql.connections.Connection: 数据库连接对象
    """
 
    try:
        conn = pymysql.connect(host=DB_CONFIG["host"],user=DB_CONFIG["user"],passwd=DB_CONFIG["password"],db=DB_CONFIG["database"])
        logger.info(f"数据库{host}的{db}连接成功")        
        return conn
    except Exception as e:       
        logger.info(f"数据库{host}的{db}连接失败")
        return None

def save_detection_result(skid_number, cls_name,img_url):
    """
    保存检测结果到数据库
    
    参数:
        skid_number: 雪橇号码
        cls_name: 检测的类别名称
    
    返回:
        bool: 操作是否成功
    """
    conn = None
    try:
        # 获取当前时间
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 连接数据库
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor() 
               
        # 插入数据到skid_cls表
        cursor.execute(
            "INSERT INTO skid_cls (skid_nr, dateTime, result_cls, pic_path) VALUES (%s, %s, %s, %s)",
            (skid_number, current_time, cls_name, img_url)
        )
        
        # 提交事务
        conn.commit()
        logger.info(f"检测结果已保存到数据库{host}的{db} - 雪橇号: {skid_number}, 类别: {cls_name}, 时间: {current_time}, 图片地址: {img_url}")
        return True
    except Exception as e:
        logger.error(f"保存检测结果到数据库失败: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def get_detection_history(skid_number=None, limit=10):
    """
    获取检测历史记录
    
    参数:
        skid_number: 可选，指定雪橇号码
        limit: 返回记录的最大数量
        
    返回:
        list: 检测记录列表
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return []
            
        cursor = conn.cursor()
        
        if skid_number:
            cursor.execute(
                "SELECT * FROM skid_cls WHERE skid_nr = %s ORDER BY dateTime DESC LIMIT %s",
                (skid_number, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM skid_cls ORDER BY dateTime DESC LIMIT %s",
                (limit,)
            )
        
        # 获取结果
        result = cursor.fetchall()
        
        return result
    except Exception as e:
        logger.error(f"获取检测历史记录失败: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

def update_detection_result(record_id, new_cls_name):
    """
    更新检测结果
    
    参数:
        record_id: 记录ID
        new_cls_name: 新的检测类别名称
        
    返回:
        bool: 操作是否成功
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE skid_cls SET result_cls = %s WHERE id = %s",
            (new_cls_name, record_id)
        )
        
        if cursor.rowcount == 0:
            logger.warning(f"未找到ID为{record_id}的记录，更新失败")
            return False
            
        conn.commit()
        logger.info(f"检测结果已更新 - ID: {record_id}, 新类别: {new_cls_name}")
        return True
    except Exception as e:
        logger.error(f"更新检测结果失败: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def delete_detection_record(record_id):
    """
    删除检测记录
    
    参数:
        record_id: 记录ID
        
    返回:
        bool: 操作是否成功
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM skid_cls WHERE id = %s",
            (record_id,)
        )
        
        if cursor.rowcount == 0:
            logger.warning(f"未找到ID为{record_id}的记录，删除失败")
            return False
            
        conn.commit()
        logger.info(f"检测记录已删除 - ID: {record_id}")
        return True
    except Exception as e:
        logger.error(f"删除检测记录失败: {str(e)}")
        return False
    finally:
        if conn:
            conn.close() 

#++++++++++++++++++++++++minio功能+++++++++++++++++++++++++++++++++++++
def get_minio_client():
    """
    获取MinIO客户端
    
    返回:
        Minio: MinIO客户端对象
    """
    try:
        client = Minio(
            MINIO_CONFIG["endpoint"],
            access_key=MINIO_CONFIG["access_key"],
            secret_key=MINIO_CONFIG["secret_key"],
            secure=MINIO_CONFIG["secure"]
        )
        logger.info(f"MinIO客户端连接成功 - 端点: {MINIO_CONFIG['endpoint']}")
        return client
    except Exception as e:
        logger.error(f"MinIO客户端连接失败: {str(e)}")
        return None

def ensure_bucket_exists(client, bucket_name):
    """
    确保存储桶存在，如果不存在则创建
    
    参数:
        client: MinIO客户端
        bucket_name: 存储桶名称
        
    返回:
        bool: 操作是否成功
    """
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logger.info(f"创建存储桶: {bucket_name}")
        return True
    except Exception as e:
        logger.error(f"确保存储桶存在失败: {str(e)}")
        return False

def save_image_to_minio(image_data,prefix,server_ip_port="127.0.0.1:9000"):
    """
    将图片保存到MinIO对象存储
    
    参数:
        image_data: 图片数据，可以是OpenCV图像对象或图像字节数据
        
    返回:
        str: 保存后的图片URL，失败则返回None
    """
    try:
        # 获取MinIO客户端
        client = get_minio_client()
        if not client:
            return None
        
        # 确保存储桶存在
        if not ensure_bucket_exists(client, BUCKET_NAME):
            return None
        
        # 获取当前日期，用于创建目录结构
        now = datetime.datetime.now()
        year = str(now.year)
        month = f"{now.month:02d}"
        day = f"{now.day:02d}"
        
        # 生成唯一文件名
        file_name = f"{prefix}_{uuid.uuid4()}.jpg"
        
        # 构建对象名称，格式为：年/月/日/文件名
        object_name = f"{year}/{month}/{day}/{file_name}"
        
        # 将图像数据转换为字节流
        if isinstance(image_data, bytes):
            # 如果已经是字节数据，直接使用
            image_bytes = image_data
        else:
            # 如果是OpenCV图像对象，转换为JPEG字节流
            _, image_bytes = cv2.imencode(".jpg", image_data)
            image_bytes = image_bytes.tobytes()
        
        # 创建内存文件对象
        image_stream = io.BytesIO(image_bytes)
        
        # 上传到MinIO
        client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=image_stream,
            length=len(image_bytes),
            content_type="image/jpeg"
        )
        
        # 构建并返回图片URL
        if MINIO_CONFIG["secure"]:
            protocol = "https"
        else:
            protocol = "http"
            
        # image_url = f"{protocol}://{MINIO_CONFIG['endpoint']}/{BUCKET_NAME}/{object_name}"
        image_url = f"{protocol}://{server_ip_port}/{BUCKET_NAME}/{object_name}"
        logger.info(f"图片已保存到MinIO - URL: {image_url}")
        
        return image_url
    except Exception as e:
        logger.error(f"保存图片到MinIO失败: {str(e)}")
        return None 