# YOLO 🚀 by Ultralytics, GPL-3.0 license
"""
1.0版本
生产版本
"""

import argparse
import numpy as np
from flask import Flask, request, Response
from ultralytics import YOLO
from logger_config import setup_logger
# 重新导入db_utils模块，确保使用最新版本
from db_utils import *

# 初始化日志系统
setup_logger()

app = Flask(__name__)
models = {}

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# DETECTION_URL = '/v1/object-detection/<model>'
DETECTION_URL = '/skid_detection'
model_path = r"E:\yolo11\yolo11\ultralytics-8.3.148\ultralytics-8.3.148\runs\detect\train\weights\best.pt"
#数据库为本地，不用配置，或者参见db_utils.py
minio_server_ip_port = "172.22.44.99:9000"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

logger.info(f"模型位置: {model_path}")  
# 封装返回结果的函数
def create_response(image_bytes, status_code=200, class_name=None, message="success"):
    """
    创建包含图像数据、状态码和检测类别的响应
    
    参数:
        image_bytes: 图像字节数据
        status_code: 状态码
        class_name: 检测的类别名称
        message: 状态消息
        
    返回:
        Response对象，包含图像数据和JSON格式的元数据
    """
    # 确保所有值都是ASCII兼容的
    if class_name is None:
        class_name = "unknown"
    
    # 创建包含元数据的响应头 - 确保所有值都是ASCII兼容的
    headers = {
        'X-Status-Code': str(status_code),
        'X-Status-Message': message,
        'X-Detected-Class': class_name,
        'Content-Type': 'image/jpeg'
    }
    
    logger.info(f"创建响应 - 状态码: {status_code}, 消息: {message}, 检测类别: {class_name}")
    
    # 返回带有自定义头部的响应
    return Response(image_bytes, headers=headers)


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    if request.method != 'POST':
        return create_response(b'', 405, None, "method_not_allowed")
    
    # 从请求头中提取雪橇号码和日期
    skid_number = request.headers.get('skid-number', '')
    date = request.headers.get('date', '')
    time = request.headers.get('time', '')
    
    # 记录接收到的雪橇号码和日期信息
    logger.info(f"接收到请求 - 雪橇号码: {skid_number or '未知'}, 日期: {date or '未知'}, 时间: {time or '未知'}")
    
    # 图片方式
    if request.data:       
        logger.info("接收到图片数据")
        try:
            # 将图片转换为numpy数组
            img = cv2.imdecode(np.frombuffer(request.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logger.error("无法解码图像数据")
                return create_response(b'', 400, None, "invalid_image")
                
            # 执行模型推理
            model = YOLO(model_path, task="detect")
            results = model(source=img)
            
            # 获取类别名称
            names = results[0].names
            
            # 检查是否有检测框
            if len(results[0].boxes) == 0:
                logger.warning("未检测到任何目标")
                result_img = results[0].plot()
                img_bytes = cv2.imencode(".jpg", result_img)[1].tobytes()
                return create_response(img_bytes, 200, "no_detection", "no_target_detected")
            
            # 获取第一个框的类别名称
            cls_name = names[results[0].boxes[0].cls.item()]        
            logger.info('.....................')
            logger.debug(f"模型所有类别: {names}")        
            logger.info(f"检测类别结果: {cls_name}")  # 根据索引获取类别名称
            logger.info('----------------------')
            
            # 在图像上添加雪橇号码和日期信息
            result_img = results[0].plot()
            if skid_number or date:
                # 获取图像尺寸
                h, w = result_img.shape[:2]
                
                # 添加文本到图像右下角
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                
                # 计算文本尺寸
                if skid_number:
                    skid_text = f"skid: {skid_number}"
                    (skid_width, skid_height), _ = cv2.getTextSize(skid_text, font, font_scale, thickness)
                    # 放在右下角，留出一定边距
                    skid_x = w - skid_width - 10
                    skid_y = h - 20
                    # 添加半透明背景
                    cv2.rectangle(result_img, (skid_x - 5, skid_y - skid_height - 5), (skid_x + skid_width + 5, skid_y + 5), (0, 0, 0, 128), -1)
                    # 添加文本
                    cv2.putText(result_img, skid_text, (skid_x, skid_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                if date:
                    date_text = f"date: {date}"
                    (date_width, date_height), _ = cv2.getTextSize(date_text, font, font_scale, thickness)
                    # 放在雪橇号上方
                    date_x = w - date_width - 10
                    date_y = h - 20 - (skid_height + 15 if skid_number else 0)
                    # 添加半透明背景
                    cv2.rectangle(result_img, (date_x - 5, date_y - date_height - 5), (date_x + date_width + 5, date_y + 5), (0, 0, 0, 128), -1)
                    # 添加文本
                    cv2.putText(result_img, date_text, (date_x, date_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                if time:
                    time_text = f"time: {time}"
                    (time_width, time_height), _ = cv2.getTextSize(time_text, font, font_scale, thickness)
                    # 放在日期上方
                    time_x = w - time_width - 10
                    time_y = date_y - (date_height + 15 if date else 0) - (0 if date else (skid_height + 15 if skid_number else 0))
                    # 添加半透明背景
                    cv2.rectangle(result_img, (time_x - 5, time_y - time_height - 5), (time_x + time_width + 5, time_y + 5), (0, 0, 0, 128), -1)
                    # 添加文本
                    cv2.putText(result_img, time_text, (time_x, time_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # logger.info("图像处理完成，准备显示结果")
            # cv2.imshow('yolo_result', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # 返回预测结果(这是图片)
            img_bytes = cv2.imencode(".jpg", result_img)[1].tobytes()

            # 保存图片到minio并返回图片地址
            prefix = f"{skid_number}_{date}_{time}"
            img_url = save_image_to_minio(result_img,prefix,minio_server_ip_port)            
            
            # 保存至MS数据库(本地)
            if skid_number:
                save_detection_result(skid_number, cls_name,img_url)
            else:
                logger.warning("未提供雪橇号码，跳过数据库保存")
            
            return create_response(img_bytes, 200, cls_name, "detection_success")
            
        except Exception as e:
            logger.error(f"处理图像时出错: {str(e)}")
            return create_response(b'', 500, None, f"server_error")     

    
    logger.warning("请求中没有图像数据")
    return create_response(b'', 400, None, "no_image_data")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv11 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    # parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    # for m in opt.model:
    #     models[m] = torch.hub.load('./', m, source="local")
    
    logger.info(f"服务启动在端口: {opt.port}")
    app.run(host='0.0.0.0', port=opt.port)  # debug=True causes Restarting with stat
