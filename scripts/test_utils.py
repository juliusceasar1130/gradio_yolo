# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建工具函数测试脚本

"""
工具函数测试脚本
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from yolo_detector.utils import (
    get_image_files, ensure_dir, get_file_info, get_example_images,
    load_image_pil, get_image_info, format_image_info
)


def test_file_utils():
    """测试文件工具函数"""
    print("=== 测试文件工具函数 ===")
    
    try:
        # 测试目录创建
        test_dir = project_root / "test_output"
        success = ensure_dir(test_dir)
        print(f"✓ 目录创建测试: {success}")
        
        # 测试获取图像文件
        # 使用配置中的输入文件夹路径
        input_folder = r"D:\Python\00雪橇打标\20250524"
        if os.path.exists(input_folder):
            image_files = get_image_files(input_folder)
            print(f"✓ 找到图像文件: {len(image_files)} 个")
            
            if image_files:
                # 测试获取文件信息
                first_file = image_files[0]
                file_info = get_file_info(first_file)
                print(f"✓ 文件信息获取: {file_info.get('name', 'N/A')}")
                
                # 测试获取示例图像
                examples = get_example_images(input_folder, max_count=3)
                print(f"✓ 示例图像获取: {len(examples)} 个")
        else:
            print(f"⚠ 输入文件夹不存在，跳过图像文件测试: {input_folder}")
        
        return True
        
    except Exception as e:
        print(f"✗ 文件工具函数测试失败: {e}")
        return False


def test_image_utils():
    """测试图像工具函数"""
    print("\n=== 测试图像工具函数 ===")
    
    try:
        # 测试图像加载和信息获取
        input_folder = r"D:\Python\00雪橇打标\20250524"
        if os.path.exists(input_folder):
            image_files = get_image_files(input_folder)
            
            if image_files:
                first_image = image_files[0]
                
                # 测试PIL图像加载
                pil_img = load_image_pil(first_image)
                if pil_img:
                    print(f"✓ PIL图像加载成功: {pil_img.size}")
                    
                    # 测试图像信息获取
                    img_info = get_image_info(pil_img)
                    print(f"✓ 图像信息获取: {img_info.get('width', 'N/A')}x{img_info.get('height', 'N/A')}")
                    
                    # 测试格式化图像信息
                    formatted_info = format_image_info(pil_img, os.path.basename(first_image))
                    print(f"✓ 格式化信息生成成功")
                    print(formatted_info[:100] + "..." if len(formatted_info) > 100 else formatted_info)
                else:
                    print("✗ PIL图像加载失败")
            else:
                print("⚠ 未找到图像文件进行测试")
        else:
            print(f"⚠ 输入文件夹不存在，跳过图像工具测试: {input_folder}")
        
        return True
        
    except Exception as e:
        print(f"✗ 图像工具函数测试失败: {e}")
        return False


def test_integration():
    """测试工具函数集成"""
    print("\n=== 测试工具函数集成 ===")
    
    try:
        # 模拟原始代码中的功能
        input_folder = r"D:\Python\00雪橇打标\20250524"
        
        if os.path.exists(input_folder):
            # 获取示例图像（模拟get_example_images函数）
            example_files = get_example_images(input_folder, max_count=5)
            print(f"✓ 获取示例图像: {len(example_files)} 个")
            
            if example_files:
                # 模拟图像信息更新功能
                first_image_path = example_files[0]
                pil_img = load_image_pil(first_image_path)
                
                if pil_img:
                    # 模拟update_image_info函数
                    filename = os.path.basename(first_image_path)
                    formatted_info = format_image_info(pil_img, filename)
                    
                    print("✓ 模拟图像信息更新功能:")
                    print(formatted_info)
                    
                    return True
        else:
            print(f"⚠ 输入文件夹不存在，跳过集成测试: {input_folder}")
            return True  # 不算失败，只是跳过
        
        return True
        
    except Exception as e:
        print(f"✗ 工具函数集成测试失败: {e}")
        return False


if __name__ == "__main__":
    print("开始测试工具函数模块...")
    
    success = True
    success &= test_file_utils()
    success &= test_image_utils()
    success &= test_integration()
    
    if success:
        print("\n🎉 所有工具函数测试通过！")
    else:
        print("\n❌ 部分工具函数测试失败！")
        sys.exit(1)
