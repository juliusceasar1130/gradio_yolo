# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Perform test request
"""

import pprint
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import requests

DETECTION_URL = 'http://localhost:5000/test'
IMAGE = r'D:\Python\00雪橇打标\20250521\10_20250521_103553.jpg'

# Read image
# with open(IMAGE, 'rb') as f:
#     image_data = f.read()

img = cv2.imread(IMAGE)
print("requesting....")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#将numpy数组格式的buf转换为原始字节流（bytes类型），便于网络传输、文件保存或其他二进制操作。
img = cv2.imencode(".jpg", img)[1].tobytes()
response = requests.post(DETECTION_URL, data=img,timeout=5000)
img = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
# 显示
plt.imshow(img)
plt.show()

# pprint.pprint(response)
