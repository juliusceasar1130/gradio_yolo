# YOLO 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLO models
"""

import argparse
import io
import numpy as np
import cv2

import torch
from flask import Flask, request
from PIL import Image

from ultralytics import YOLO
import pprint


app = Flask(__name__)
models = {}

# DETECTION_URL = '/v1/object-detection/<model>'
DETECTION_URL = '/test'
model_path = r"D:\00deeplearn\yolo11\【2】训练模型\cls\train_new\weights\best.pt"


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    if request.method != 'POST':
        return
    # 图片方式
    if request.data:       
        pprint.pprint("接受到请求")
        # 将图片转换为numpy数组
        img = cv2.imdecode(np.frombuffer(request.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        # if model in models:
        #     results = models[model](img)  # reduce size=320 for faster inference
        #     results = results.render()[0]
        #     return cv2.imencode(".jpg", results)[1].tobytes()
        model = YOLO(model_path, task="detect")
        results = model(source=img)
        cv2.imshow('yolo_result', cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 返回预测结果(这是图片)
        return cv2.imencode(".jpg", results[0].plot())[1].tobytes()
    
    # 文件方式（不用）
    if request.files.get('image'):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = models[model](im, size=640)  # reduce size=320 for faster inference
            return results.pandas().xyxy[0].to_json(orient='records')      


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv11 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    # parser.add_argument('--model', nargs='+', default=['yolov5s'], help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()

    # for m in opt.model:
    #     models[m] = torch.hub.load('./', m, source="local")

    app.run(host='0.0.0.0', port=opt.port)  # debug=True causes Restarting with stat
