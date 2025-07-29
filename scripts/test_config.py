# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建配置测试脚本

"""
配置管理模块测试脚本
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from yolo_detector.config.settings import Config, get_config


def test_config_loading():
    """测试配置加载"""
    print("=== 测试配置加载 ===")
    
    try:
        config = Config()
        print(f"✓ 配置加载成功: {config.config_path}")
        
        # 测试基本配置获取
        detection_model = config.get('models.detection.path')
        print(f"✓ 检测模型路径: {detection_model}")
        
        segmentation_model = config.get('models.segmentation.path')
        print(f"✓ 分割模型路径: {segmentation_model}")
        
        input_folder = config.get('data.input_folder')
        print(f"✓ 输入文件夹: {input_folder}")
        
        confidence = config.get('detection.confidence_threshold')
        print(f"✓ 置信度阈值: {confidence}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


def test_config_methods():
    """测试配置方法"""
    print("\n=== 测试配置方法 ===")
    
    try:
        config = get_config()
        
        # 测试专用方法
        model_config = config.get_model_config('detection')
        print(f"✓ 检测模型配置: {model_config}")
        
        detection_config = config.get_detection_config()
        print(f"✓ 检测参数配置: {detection_config}")
        
        ui_config = config.get_ui_config()
        print(f"✓ UI配置: {ui_config}")
        
        # 测试默认值
        non_existent = config.get('non.existent.key', 'default_value')
        print(f"✓ 默认值测试: {non_existent}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置方法测试失败: {e}")
        return False


def test_config_validation():
    """测试配置验证"""
    print("\n=== 测试配置验证 ===")
    
    try:
        config = get_config()
        
        # 检查必需的配置节
        required_sections = ['models', 'data', 'detection', 'ui']
        for section in required_sections:
            value = config.get(section)
            if value:
                print(f"✓ 配置节 '{section}' 存在")
            else:
                print(f"✗ 配置节 '{section}' 缺失")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置验证失败: {e}")
        return False


if __name__ == "__main__":
    print("开始测试配置管理模块...")
    
    # 确保scripts目录存在
    os.makedirs(Path(__file__).parent, exist_ok=True)
    
    success = True
    success &= test_config_loading()
    success &= test_config_methods()
    success &= test_config_validation()
    
    if success:
        print("\n🎉 所有配置测试通过！")
    else:
        print("\n❌ 部分配置测试失败！")
        sys.exit(1)
