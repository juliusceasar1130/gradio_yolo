import requests
import os
from datetime import datetime
from requests.auth import HTTPDigestAuth

username, password = 'root', 'root'  # 网络摄像机的用户名和密码
class Camera:
    def __init__(self, url, name):
        # 确保 URL 有 http:// 前缀
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        self.url = url  # 网络摄像机的完整URL（带http://）
        self.image_data = None
        self.name = name
    #抓取照片
    def capture_photo(self,cmd='/axis-cgi/jpg/image.cgi?resolution=1280x720&compression=25&camera=1'):
        # 拼接命令
        url = self.url+cmd
        print("........")
        print(url)
        print("........")
        """
        从网络摄像机捕获照片
        """
        try:
            # 发送HTTP请求获取图片
            response = requests.get(url, timeout=5,auth=HTTPDigestAuth(username, password))
            response.raise_for_status()
            # 获取图片的二进制数据
            self.image_data = response.content
            # print("图片获取成功,开始保存程序...")
            # self.save_photo()
            return self.image_data
        except requests.exceptions.RequestException as e:
            print(f"图片获取失败: {e}")
            return None

    def gotoPosition(self, positon):
        # 拼接命令
        url = self.url + '/axis-cgi/com/ptz.cgi?gotoserverpresetname='+positon
        print("........")
        print(url)
        print("........")
        """
        到相机到预定为止
        """
        try:
            # 发送HTTP请求获取图片
            response = requests.get(url, timeout=5,auth=HTTPDigestAuth(username, password))
            response.raise_for_status()
            print(f'预置位response的反馈:{response}')
            return response
        except requests.exceptions.RequestException as e:
            print(f"相机到预定位置失败: {e}")
            return None

    def save_photo(self,name):
         # 获取当前时间戳
        date = datetime.now().strftime("%Y%m%d")
        time = datetime.now().strftime("%H%M%S")
         # 获取当前时间戳
        datestamp = datetime.now().strftime("%Y%m%d")
        save_dir = f"D:/img2/{datestamp}"
         # 创建保存目录
        save_path = f"{save_dir}/{name}_{date}_{time}.jpg"
         # 创建保存目录
        if not os.path.exists(save_dir):
             os.makedirs(save_dir)
        try:
            # 写入文件
            with open(save_path, "wb") as f:
                f.write(self.image_data)
            print(f"Photo saved to: {save_path}")
        except Exception as e:
            print(f"Error saving photo: {e}")
        return save_path

