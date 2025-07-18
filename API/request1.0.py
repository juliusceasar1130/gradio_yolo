# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
1.0版本
"""

import pprint
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from file_utils import extract_skid_number, extract_date, extract_time

#请求地址不能有中文
def send_request(save_path):
    DETECTION_URL = 'http://localhost:5000/skid_detection'
    # IMAGE = r'D:\Python\00雪橇打标\20250521\1780_20250521_023122.jpg'
    # 提取雪橇号码和日期
    skid_number = extract_skid_number(save_path)
    date = extract_date(save_path)
    time = extract_time(save_path)
    print('.....................')
    print(f"本地图片路径: {save_path}")
    print(f"提取的雪橇号: {skid_number}, 日期: {date}, 时间：{time}")
    print('.....................')


    # Read image
    # with open(IMAGE, 'rb') as f:
    #     image_data = f.read()

    img = cv2.imread(save_path)
    print("正在发送请求...")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #将numpy数组格式的buf转换为原始字节流（bytes类型），便于网络传输、文件保存或其他二进制操作。
    img = cv2.imencode(".jpg", img)[1].tobytes()

    # 添加自定义请求头，包含雪橇号码和日期信息
    headers = {
        'skid-number': str(skid_number) if skid_number else '',
        'date': str(date) if date else '',
        'time': str(time) if time else ''
    }

    print(f"发送请求到: {DETECTION_URL}")
    response = requests.post(DETECTION_URL, data=img, headers=headers, timeout=5000)
    print(f"接收到响应，状态码: {response.status_code}")

    # 从响应头中获取检测结果信息
    status_code = response.headers.get('X-Status-Code', '未知')
    status_message = response.headers.get('X-Status-Message', '未知')
    detected_class = response.headers.get('X-Detected-Class', '未知')

    # 状态消息映射表（将英文状态消息映射为中文）
    status_message_map = {
        'success': '成功',
        'detection_success': '检测成功',
        'method_not_allowed': '方法不允许',
        'invalid_image': '无效的图像数据',
        'no_target_detected': '未检测到目标',
        'server_error': '服务器错误',
        'no_image_data': '请求中没有图像数据'
    }

    # 类别映射表（如果需要将英文类别映射为中文）
    class_map = {
        'C': '清洁',
        'M': '中度污染',
        'S': '严重污染',
        'no_detection': '无检测结果',
        'unknown': '未知'
    }

    # 获取中文显示
    status_message_cn = status_message_map.get(status_message, status_message)
    detected_class_cn = class_map.get(detected_class, detected_class)

    print("检测结果信息:")
    print(f"状态码: {status_code}")
    print(f"状态消息: {status_message} ({status_message_cn})")
    print(f"检测类别: {detected_class} ({detected_class_cn})")

    # # 处理图像数据
    # img = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    # # 显示
    # print("显示结果图像")
    # plt.imshow(img)
    # plt.show()

    # pprint.pprint(response)

# 测试
if __name__ == "__main__":
    send_request(r'D:\1005_20250528_185917.jpg')