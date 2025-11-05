# 创建者/修改者: chenliang
# 修改时间：2025年11月02日 10:00
# 主要修改内容：测试 Gradio 升级兼容性

"""
测试 Gradio 升级后的 every 参数支持
"""

import gradio as gr
import numpy as np
from datetime import datetime

def test_every_parameter():
    """测试 every 参数是否支持"""
    print(f"Gradio 版本: {gr.__version__}")
    
    # 测试 every 参数
    try:
        with gr.Blocks() as demo:
            image = gr.Image()
            
            def update_image():
                # 生成测试图像
                img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
                return img
            
            # 尝试使用 every 参数
            try:
                image.load(
                    fn=update_image,
                    every=0.1  # 每100ms更新一次
                )
                print("✅ every 参数支持！")
                return True
            except TypeError as e:
                print(f"❌ every 参数不支持: {e}")
                return False
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Gradio 升级兼容性测试")
    print("=" * 50)
    result = test_every_parameter()
    print("\n测试结果:", "支持" if result else "不支持")
    print("=" * 50)

