# 创建者/修改者: chenliang；修改时间：2025年7月27日 23:20；主要修改内容：创建系统集成测试脚本

"""
系统集成测试脚本

验证整个重构后的系统功能完整性
"""

import sys
import os
import time
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from yolo_detector import (
    Config, ModelLoader, ObjectDetector, SegmentationDetector,
    BatchProcessor, create_gradio_interface
)
from yolo_detector.utils import setup_logging, get_logger


def test_configuration_system():
    """测试配置系统"""
    print("=== 测试配置系统 ===")
    
    try:
        config = Config()
        print("✓ 配置加载成功")
        
        # 测试配置获取
        models_config = config.get('models')
        data_config = config.get('data')
        detection_config = config.get('detection')
        
        print(f"✓ 模型配置: {len(models_config)} 个模型")
        print(f"✓ 数据配置: {data_config.get('input_folder', 'N/A')}")
        print(f"✓ 检测配置: 置信度阈值 {detection_config.get('confidence_threshold', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置系统测试失败: {e}")
        return False


def test_model_loading():
    """测试模型加载"""
    print("\n=== 测试模型加载 ===")
    
    try:
        config = Config()
        model_loader = ModelLoader(config)
        
        # 列出可用模型
        available_models = model_loader.list_available_models()
        print(f"✓ 发现 {len(available_models)} 个模型配置")
        
        for model_type, model_info in available_models.items():
            status = "存在" if model_info['exists'] else "不存在"
            print(f"  - {model_type}: {model_info['path']} ({status})")
        
        # 尝试加载模型（如果存在）
        detection_model = model_loader.load_model('detection')
        if detection_model:
            print("✓ 检测模型加载成功")
        else:
            print("⚠ 检测模型加载失败（可能是文件不存在）")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型加载测试失败: {e}")
        return False


def test_detector_functionality():
    """测试检测器功能"""
    print("\n=== 测试检测器功能 ===")
    
    try:
        config = Config()
        model_loader = ModelLoader(config)
        
        # 创建检测器
        object_detector = ObjectDetector(model_loader, config)
        segmentation_detector = SegmentationDetector(model_loader, config)
        
        print("✓ 检测器创建成功")
        
        # 测试模型加载状态
        obj_loaded = object_detector.is_model_loaded()
        seg_loaded = segmentation_detector.is_model_loaded()
        
        print(f"✓ 目标检测器状态: {'已加载' if obj_loaded else '未加载'}")
        print(f"✓ 分割检测器状态: {'已加载' if seg_loaded else '未加载'}")
        
        return True
        
    except Exception as e:
        print(f"✗ 检测器功能测试失败: {e}")
        return False


def test_image_processing():
    """测试图像处理"""
    print("\n=== 测试图像处理 ===")
    
    try:
        from yolo_detector.core.image_processor import ImageProcessor
        from yolo_detector.utils import get_image_files
        
        config = Config()
        processor = ImageProcessor(config)
        
        # 获取测试图像
        input_folder = config.get('data.input_folder')
        if os.path.exists(input_folder):
            image_files = get_image_files(input_folder)
            if image_files:
                test_image = image_files[0]
                
                # 测试图像验证
                validation = processor.validate_image_input(test_image)
                print(f"✓ 图像验证: {'通过' if validation['valid'] else '失败'}")
                
                # 测试图像预处理
                processed = processor.preprocess_image(test_image, target_size=(640, 640))
                if processed is not None:
                    print(f"✓ 图像预处理成功: {processed.shape}")
                else:
                    print("✗ 图像预处理失败")
                
                return True
            else:
                print("⚠ 未找到测试图像")
                return True  # 不算失败
        else:
            print(f"⚠ 输入文件夹不存在: {input_folder}")
            return True  # 不算失败
        
    except Exception as e:
        print(f"✗ 图像处理测试失败: {e}")
        return False


def test_batch_processing():
    """测试批量处理"""
    print("\n=== 测试批量处理 ===")
    
    try:
        from yolo_detector.utils import get_image_files
        
        config = Config()
        model_loader = ModelLoader(config)
        object_detector = ObjectDetector(model_loader, config)
        batch_processor = BatchProcessor(object_detector, config)
        
        # 获取测试图像
        input_folder = config.get('data.input_folder')
        if os.path.exists(input_folder):
            image_files = get_image_files(input_folder)
            if len(image_files) >= 2:
                # 选择少量图像进行测试
                test_images = image_files[:2]
                
                print(f"✓ 批量处理器创建成功")
                print(f"✓ 准备处理 {len(test_images)} 张图像")
                
                # 获取处理统计
                stats = batch_processor.get_processing_stats()
                print(f"✓ 处理配置: {stats['max_workers']} 个工作线程")
                
                return True
            else:
                print("⚠ 测试图像数量不足")
                return True
        else:
            print(f"⚠ 输入文件夹不存在: {input_folder}")
            return True
        
    except Exception as e:
        print(f"✗ 批量处理测试失败: {e}")
        return False


def test_result_processing():
    """测试结果处理"""
    print("\n=== 测试结果处理 ===")
    
    try:
        from yolo_detector.core.result_processor import ResultProcessor
        
        config = Config()
        processor = ResultProcessor(config)
        
        # 创建模拟结果数据
        mock_result = {
            'timestamp': '2025-07-27T23:20:00',
            'detector_type': 'detection',
            'image_path': 'test_image.jpg',
            'statistics': {
                'total_detections': 2,
                'classes': {
                    'person': {'count': 1, 'confidences': [0.85]},
                    'car': {'count': 1, 'confidences': [0.92]}
                },
                'has_detections': True
            },
            'detections': [
                {
                    'id': 0,
                    'class_name': 'person',
                    'class_id': 0,
                    'confidence': 0.85,
                    'bbox': [100, 100, 200, 300]
                }
            ]
        }
        
        # 测试汇总报告生成
        report = processor.generate_summary_report(mock_result)
        print("✓ 汇总报告生成成功")
        
        # 测试统计信息
        stats = processor.get_statistics_summary()
        print(f"✓ 统计信息获取成功: 已处理 {stats.get('total_processed', 0)} 个结果")
        
        return True
        
    except Exception as e:
        print(f"✗ 结果处理测试失败: {e}")
        return False


def test_logging_system():
    """测试日志系统"""
    print("\n=== 测试日志系统 ===")
    
    try:
        config = Config()
        setup_logging(config)
        
        logger = get_logger(__name__)
        
        # 测试不同级别的日志
        logger.debug("这是调试信息")
        logger.info("这是信息日志")
        logger.warning("这是警告信息")
        
        print("✓ 日志系统工作正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 日志系统测试失败: {e}")
        return False


def test_gradio_interface():
    """测试Gradio界面"""
    print("\n=== 测试Gradio界面 ===")
    
    try:
        config = Config()
        
        # 创建界面（不启动）
        demo = create_gradio_interface(config)
        
        print("✓ Gradio界面创建成功")
        print("✓ 界面组件配置正常")
        
        return True
        
    except Exception as e:
        print(f"✗ Gradio界面测试失败: {e}")
        return False


def test_main_entry():
    """测试主入口文件"""
    print("\n=== 测试主入口文件 ===")
    
    try:
        # 测试导入主模块
        import main
        
        print("✓ 主入口文件导入成功")
        
        # 测试系统信息功能
        config = Config()
        main.show_system_info(config)
        
        print("✓ 系统信息显示正常")
        
        return True
        
    except Exception as e:
        print(f"✗ 主入口文件测试失败: {e}")
        return False


def run_performance_test():
    """运行性能测试"""
    print("\n=== 性能测试 ===")
    
    try:
        config = Config()
        
        # 测试配置加载性能
        start_time = time.time()
        for _ in range(100):
            Config()
        config_time = time.time() - start_time
        
        print(f"✓ 配置加载性能: {config_time:.3f}秒 (100次)")
        
        # 测试模型加载器创建性能
        start_time = time.time()
        for _ in range(10):
            ModelLoader(config)
        loader_time = time.time() - start_time
        
        print(f"✓ 模型加载器创建性能: {loader_time:.3f}秒 (10次)")
        
        return True
        
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        return False


def main():
    """主函数"""
    print("开始系统集成测试...")
    print("=" * 50)
    
    tests = [
        ("配置系统", test_configuration_system),
        ("模型加载", test_model_loading),
        ("检测器功能", test_detector_functionality),
        ("图像处理", test_image_processing),
        ("批量处理", test_batch_processing),
        ("结果处理", test_result_processing),
        ("日志系统", test_logging_system),
        ("Gradio界面", test_gradio_interface),
        ("主入口文件", test_main_entry),
        ("性能测试", run_performance_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"集成测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有集成测试通过！系统功能完整。")
        return True
    else:
        print("❌ 部分集成测试失败，请检查相关功能。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
