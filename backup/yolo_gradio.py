import gradio as gr
from ultralytics import YOLO
import os
import torch
from PIL import Image
import numpy as np
import cv2

# 加载YOLO模型###############################################
model_path = r"D:\00deeplearn\yolo11\【2】训练模型\cls\train_new\weights\best.pt"
yolo_model = YOLO(model_path)
 # 指定图片所在文件夹
img_folder = r"D:\Python\00雪橇打标\20250524"
############################################################

# 图像检测函数
def detect_objects(image, conf_threshold=0.25):
    # 如果输入为None，返回None
    if image is None:
        return None
    
    # 使用YOLO模型检测对象
    results = yolo_model.predict(source=image, conf=conf_threshold)
    return results[0].plot()

# 获取模型信息
def get_model_info():
    try:
        model_type = os.path.basename(model_path).split('.')[0]
        task_type = "分割任务" if "seg" in model_path else "检测任务"
        return f"**模型信息**\n- 名称: {os.path.basename(model_path)}\n- 类型: {model_type}\n- 任务: {task_type}"
    except:
        return f"**模型信息**: {os.path.basename(model_path)}"

# 获取图片信息（文件名\大小\尺寸）
def get_image_info(file_path):
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path) / 1024  # 转换为KB
    img=cv2.imread(file_path)
    size = img.shape[0:2]
    return f"{file_name}-{size}-({file_size:.1f} KB)"

# 检查图片是否存在，并返回完整路径
def check_example_image(filename):
    # 检查当前目录
    if os.path.exists(filename):
        return filename
    
    # 检查其他可能的目录
    possible_dirs = ['.', './images', '../images']
    for dir_path in possible_dirs:
        path = os.path.join(dir_path, filename)
        if os.path.exists(path):
            return path
    
    return None

# 获取可用示例图像，返回图片路径file_path
def get_example_images():
    example_files = []
    
   
    
    # 检查文件夹是否存在
    if not os.path.exists(img_folder):
        print(f"图片文件夹 {img_folder} 不存在，尝试创建")
        try:
            os.makedirs(img_folder, exist_ok=True)
        except:
            print(f"无法创建文件夹 {img_folder}")
    
    # 从zzimg文件夹加载JPG和PNG文件
    if os.path.exists(img_folder):
        for file in os.listdir(img_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.abspath(os.path.join(img_folder, file))            
                if os.path.exists(file_path):
                    example_files.append(file_path)
                    print(f"找到图片文件: {file_path}")
    
    # 如果没有找到图片，使用默认图片
    if not example_files:
        print("未找到可用图片，尝试使用默认图片")
        default_image = os.path.join(img_folder, "sample.jpg")
        if os.path.exists(default_image):
            example_files.append(os.path.abspath(default_image))
            print(f"使用默认图片: {default_image}")
        else:
            print(f"默认图片不存在: {default_image}")
    
    print(f"总共找到 {len(example_files)} 个图片文件")
    return example_files

# 测试加载一个示例图片，确认读取格式正确
def test_image_loading():
    example_files = get_example_images()
    if not example_files:
        print("没有可用的示例图片文件")
        return None
    
    first_image = example_files[0]
    print(f"测试加载图片: {first_image}")
    
    try:
        # 使用PIL加载
        pil_img = Image.open(first_image)
        print(f"PIL成功加载图片: 大小={pil_img.size}, 模式={pil_img.mode}")
        
        # 使用OpenCV加载
        cv_img = cv2.imread(first_image)
        if cv_img is not None:
            print(f"OpenCV成功加载图片: 形状={cv_img.shape}")
        else:
            print("OpenCV加载图片失败")
        
        return first_image
    except Exception as e:
        print(f"测试加载图片失败: {e}")
        return None

# 检测并显示结果统计
def detect_with_stats(image, conf_threshold=0.25):
    if image is None:
        return None, "未检测到图像"
    
    # 使用YOLO模型检测对象
    results = yolo_model.predict(source=image, conf=conf_threshold)
    result = results[0]
    
    # 提取检测统计信息
    detected_classes = {}
    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = result.names[cls_id]
            if cls_name in detected_classes:
                detected_classes[cls_name] += 1
            else:
                detected_classes[cls_name] = 1
    
    # 创建统计信息文本
    stats_text = "**检测结果统计**\n"
    if len(detected_classes) > 0:
        for cls_name, count in detected_classes.items():
            stats_text += f"- {cls_name}: {count}个\n"
    else:
        stats_text += "未检测到任何对象"
    
    # 转换为RGB格式
    result_img = result.plot()
    if isinstance(result_img, np.ndarray):
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_img, stats_text

# 创建主界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# YOLO 目标检测与分割系统")
    gr.Markdown("### 上传图片或使用示例图片进行物体检测")
    
    # 测试图片加载
    test_image = test_image_loading()
    if test_image:
        # gr.Markdown(f"示例图片测试加载成功: {os.path.basename(test_image)}")
        pass
    else:
        gr.Markdown("⚠️ 警告: 无法加载测试图片，示例功能可能不可用")
    
    with gr.Row():
        #左侧
        with gr.Column(scale=1):
            # 输入选项
            input_image = gr.Image(label="输入图片", type="pil")
            image_info = gr.Markdown(label="图片信息")
            
            # 定义一个全局变量用于存储当前文件名
            current_filename = {"value": ""}
            
            # 添加文件名设置函数
            def set_filename(filename):
                current_filename["value"] = filename
                return filename
            
            # 在Gradio界面添加一个隐藏的文本框，用于存储上传的文件名
            upload_filename = gr.Textbox(visible=False, elem_id="upload_filename")
            
            # 监听文件名变化
            upload_filename.change(
                fn=set_filename,
                inputs=[upload_filename],
                outputs=[upload_filename]
            )
            
            # 添加自定义JavaScript
            demo.load(fn=lambda: None)
            
            gr.HTML("""
                <script>
                function captureFileName() {
                    const uploadButton = document.querySelector('#input_image');
                    if (!uploadButton) return;
                    
                    // 监听文件选择
                    uploadButton.addEventListener('change', function(e) {
                        if (e.target.files && e.target.files.length > 0) {
                            const filename = e.target.files[0].name;
                            document.querySelector('#upload_filename').value = filename;
                            document.querySelector('#upload_filename').dispatchEvent(new Event('input'));
                        }
                    });
                }
                
                // 在页面加载完成后执行
                window.addEventListener('load', captureFileName);
                </script>
            """)
            
            # 定义图片上传后的回调函数
            def update_image_info(image):
                if image is None:
                    return "未上传图片"
                
                # 获取图片尺寸
                width, height = image.size
                # 计算图片大小(内存中)
                img_array = np.array(image)
                size_kb = img_array.nbytes / 1024
                
                # 使用全局变量中保存的文件名
                filename = current_filename.get("value", "")
                
                # 显示文件名(如果有)和尺寸信息
                if filename:
                    return f"**图片信息**\n- 文件名: {filename}\n- 尺寸: {width}×{height} 像素\n"
                else:
                    return f"**图片信息**\n- 尺寸: {width}×{height} 像素\n"
            
            # 设置图片变化事件，更新信息
            input_image.change(
                fn=update_image_info,
                inputs=[input_image],
                outputs=[image_info]
            )
            
            # 置信度滑块
            conf_slider = gr.Slider(
                minimum=0.1, 
                maximum=0.9, 
                value=0.25, 
                step=0.05, 
                label="置信度阈值", 
                info="调整检测阈值（值越低检测越多）"
            )
            
            # 操作按钮
            with gr.Row():
                detect_btn = gr.Button("开始检测", variant="primary")
                clear_btn = gr.Button("清除", variant="secondary")
            
            # 模型信息
            model_info = gr.Markdown(get_model_info())
            

        #右侧    
        with gr.Column(scale=1):
            # 输出区域
            output_image = gr.Image(label="")
            stats_output = gr.Markdown(label="统计信息")
            
            with gr.Accordion("YOLO 模型信息", open=False):
                # gr.Markdown("""
                # ### YOLO模型说明
                
                # - YOLO (You Only Look Once) 是一种流行的实时对象检测系统
                # - 该模型能够检测图像中的多个对象并提供边界框和类别预测
                # - 当前加载的模型支持分割任务，可以生成对象的轮廓
                
                # 更多信息请参考 [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
                # """)
                pass

         # 示例图片
            gr.Markdown(f"### 示例图片（来自{img_folder}文件夹）")            
            # 获取示例图片
            example_files = get_example_images()
            
            if example_files:
                
                with gr.Column():              
                                                  
                    # 准备图片及其标签
                    image_labels = []
                    example_images = []
                    for img_path in example_files:
                        img_info = get_image_info(img_path)
                        image_labels.append(img_info)
                        example_images.append((img_path, img_info))
                    
                    # 设置Gallery的值为图片列表
                    examples_gallery = gr.Gallery(
                        columns=3,
                        rows=4,
                        object_fit="scale-down", # 可选值: "contain", "cover", "fill", "none", "scale-down"
                        height="300px",  # 可选值: "auto", "200px", "300px", "400px" 等具体像素值
                        value=example_images,                        
                    )
                    
                    # 点击图片时触发的事件
                    def use_example(evt: gr.SelectData):
                        selected_index = evt.index
                        if 0 <= selected_index < len(example_files):
                            selected_img_path = example_files[selected_index]
                            img = Image.open(selected_img_path)
                            filename = os.path.basename(selected_img_path)
                            
                            # 更新全局文件名变量
                            current_filename["value"] = filename
                            
                            # 返回图片和自定义信息（包含文件名）
                            width, height = img.size
                            img_array = np.array(img)
                            size_kb = img_array.nbytes / 1024
                            custom_info = f"**图片信息**\n- 文件名: {filename}\n- 尺寸: {width}×{height} 像素\n"
                            return img, custom_info
                        return None, "未选择图片"
                    
                    # 连接选择示例图片事件到图片展示和信息更新
                    examples_gallery.select(use_example, outputs=[input_image, image_info])
            else:
                gr.Markdown("⚠️ 未找到可用的示例图片")
    
    # 处理按钮事件
    # 检测按钮和图片变化时的函数，需要保持图片信息
    def detect_and_keep_info(image, conf):
        result_img, stats = detect_with_stats(image, conf)
        
        # 使用全局变量中保存的文件名
        filename = current_filename.get("value", "")
        
        # 使用当前图片信息
        if image is not None:
            width, height = image.size
            img_array = np.array(image)
            size_kb = img_array.nbytes / 1024
            
            if filename:
                img_info = f"**图片信息**\n- 文件名: {filename}\n- 尺寸: {width}×{height} 像素\n"
            else:
                img_info = f"**图片信息**\n- 尺寸: {width}×{height} 像素\n"
        else:
            img_info = "未上传图片"
            
        return result_img, stats, img_info
    
    detect_btn.click(
        fn=detect_and_keep_info,
        inputs=[input_image, conf_slider],
        outputs=[output_image, stats_output, image_info]
    )
    
    # 定义清除功能
    def clear_all():
        # 清除全局文件名变量
        current_filename["value"] = ""
        return None, "", "未上传图片"
    
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[input_image, stats_output, image_info]
    )
    
    # 图片变化时自动检测
    input_image.change(
        fn=detect_and_keep_info,
        inputs=[input_image, conf_slider],
        outputs=[output_image, stats_output, image_info]
    )

# 启动应用
demo.launch(share=True, debug=True)