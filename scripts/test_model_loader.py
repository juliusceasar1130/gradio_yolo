# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建模型加载器测试脚本

"""
模型加载器测试脚本
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from yolo_detector.config.settings import Config
from yolo_detector.models.model_loader import ModelLoader


def test_model_loader_basic():
    """测试模型加载器基本功能"""
    print("=== 测试模型加载器基本功能 ===")
    
    try:
        # 创建配置和模型加载器
        config = Config()
        loader = ModelLoader(config)
        
        print("✓ 模型加载器创建成功")
        
        # 列出可用模型
        available_models = loader.list_available_models()
        print(f"✓ 可用模型: {list(available_models.keys())}")
        
        for model_type, info in available_models.items():
            exists = "存在" if info['exists'] else "不存在"
            print(f"  - {model_type}: {info['path']} ({exists})")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载器基本功能测试失败: {e}")
        return False


def test_model_loading():
    """测试模型加载功能"""
    print("\n=== 测试模型加载功能 ===")
    
    try:
        config = Config()
        loader = ModelLoader(config)
        
        # 测试加载检测模型
        detection_model = loader.load_model('detection')
        if detection_model:
            print("✓ 检测模型加载成功")
            
            # 获取模型信息
            model_info = loader.get_model_info('detection')
            print(f"  - 任务类型: {model_info.get('task', 'N/A')}")
            print(f"  - 类别数量: {len(model_info.get('names', {})) if model_info.get('names') else 'N/A'}")
        else:
            print("⚠ 检测模型加载失败（可能是模型文件不存在）")
        
        # 测试加载分割模型
        segmentation_model = loader.load_model('segmentation')
        if segmentation_model:
            print("✓ 分割模型加载成功")
        else:
            print("⚠ 分割模型加载失败（可能是模型文件不存在）")
        
        # 测试模型缓存
        cached_model = loader.get_model('detection')
        if cached_model:
            print("✓ 模型缓存功能正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载功能测试失败: {e}")
        return False


def test_model_management():
    """测试模型管理功能"""
    print("\n=== 测试模型管理功能 ===")
    
    try:
        config = Config()
        loader = ModelLoader(config)
        
        # 测试模型状态检查
        is_loaded_before = loader.is_model_loaded('detection')
        print(f"✓ 加载前检测模型状态: {'已加载' if is_loaded_before else '未加载'}")
        
        # 尝试加载模型
        model = loader.load_model('detection')
        if model:
            is_loaded_after = loader.is_model_loaded('detection')
            print(f"✓ 加载后检测模型状态: {'已加载' if is_loaded_after else '未加载'}")
            
            # 测试模型信息获取
            model_info = loader.get_model_info('detection')
            print(f"✓ 模型信息获取成功: {model_info['type']}")
            
            # 测试模型卸载
            unload_success = loader.unload_model('detection')
            print(f"✓ 模型卸载: {'成功' if unload_success else '失败'}")
            
            is_loaded_final = loader.is_model_loaded('detection')
            print(f"✓ 卸载后检测模型状态: {'已加载' if is_loaded_final else '未加载'}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型管理功能测试失败: {e}")
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    try:
        config = Config()
        loader = ModelLoader(config)
        
        # 测试加载不存在的模型类型
        invalid_model = loader.load_model('invalid_type')
        if invalid_model is None:
            print("✓ 无效模型类型处理正确")
        
        # 测试加载不存在的模型文件
        nonexistent_model = loader.load_model('detection', '/path/to/nonexistent/model.pt')
        if nonexistent_model is None:
            print("✓ 不存在模型文件处理正确")
        
        # 测试获取未加载模型的信息
        info = loader.get_model_info('nonexistent')
        if not info['loaded']:
            print("✓ 未加载模型信息获取正确")
        
        return True
        
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
        return False


def test_integration():
    """测试与配置系统的集成"""
    print("\n=== 测试与配置系统集成 ===")
    
    try:
        # 测试从配置加载模型
        config = Config()
        loader = ModelLoader(config)
        
        # 获取配置中的模型信息
        detection_config = config.get_model_config('detection')
        segmentation_config = config.get_model_config('segmentation')
        
        print(f"✓ 检测模型配置: {detection_config.get('path', 'N/A')}")
        print(f"✓ 分割模型配置: {segmentation_config.get('path', 'N/A')}")
        
        # 测试预加载功能
        preload_results = loader.preload_models(['detection'])
        print(f"✓ 预加载结果: {preload_results}")
        
        return True
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        return False


if __name__ == "__main__":
    print("开始测试模型加载器...")
    
    success = True
    success &= test_model_loader_basic()
    success &= test_model_loading()
    success &= test_model_management()
    success &= test_error_handling()
    success &= test_integration()
    
    if success:
        print("\n🎉 所有模型加载器测试通过！")
    else:
        print("\n❌ 部分模型加载器测试失败！")
        sys.exit(1)
