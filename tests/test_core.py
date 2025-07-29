# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:45；主要修改内容：创建核心模块单元测试

"""
核心模块单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from yolo_detector.core.detector import DetectionResult, BaseDetector, ObjectDetector, SegmentationDetector
from yolo_detector.core.image_processor import ImageProcessor
from yolo_detector.core.result_processor import ResultProcessor
from yolo_detector.core.batch_processor import BatchProcessor


class TestDetectionResult:
    """检测结果测试"""
    
    def test_detection_result_initialization(self, mock_detection_result):
        """测试检测结果初始化"""
        result = DetectionResult(mock_detection_result, "detection")
        
        assert result.detector_type == "detection"
        assert result.raw_result == mock_detection_result
        assert result.boxes is not None
        assert result.names == {0: 'person', 1: 'car'}
    
    def test_detection_result_statistics(self, mock_detection_result):
        """测试检测结果统计"""
        result = DetectionResult(mock_detection_result, "detection")
        stats = result.get_statistics()
        
        assert stats['total_detections'] == 2
        assert stats['has_detections'] == True
        assert 'person' in stats['classes']
        assert 'car' in stats['classes']
        assert stats['classes']['person']['count'] == 1
        assert stats['classes']['car']['count'] == 1
    
    def test_detection_result_format_statistics(self, mock_detection_result):
        """测试统计信息格式化"""
        result = DetectionResult(mock_detection_result, "detection")
        formatted = result.format_statistics()
        
        assert "**检测结果统计**" in formatted
        assert "总计: 2 个对象" in formatted
        assert "person" in formatted
        assert "car" in formatted
    
    def test_detection_result_visualization(self, mock_detection_result):
        """测试可视化结果"""
        result = DetectionResult(mock_detection_result, "detection")
        vis_image = result.get_visualization()
        
        assert vis_image is not None
        assert isinstance(vis_image, np.ndarray)
        assert len(vis_image.shape) == 3  # RGB图像
    
    def test_empty_detection_result(self):
        """测试空检测结果"""
        class EmptyResult:
            def __init__(self):
                self.boxes = []
                self.masks = None
                self.names = {}
                self.path = None
        
        result = DetectionResult(EmptyResult(), "detection")
        stats = result.get_statistics()
        
        assert stats['total_detections'] == 0
        assert stats['has_detections'] == False
        assert stats['classes'] == {}


class TestMaskStatistics:
    """掩码统计功能测试"""

    @pytest.fixture
    def mock_mask_data(self):
        """创建模拟掩码数据"""
        # 创建20x20的测试掩码
        mask1 = np.zeros((20, 20))
        mask1[5:15, 5:15] = 1.0  # 10x10 = 100像素

        mask2 = np.zeros((20, 20))
        mask2[10:18, 10:18] = 1.0  # 8x8 = 64像素，与mask1重叠5x5=25像素

        mask3 = np.zeros((20, 20))
        mask3[0:6, 0:6] = 1.0  # 6x6 = 36像素，无重叠

        return [mask1, mask2, mask3]

    @pytest.fixture
    def mock_segmentation_result(self, mock_mask_data):
        """创建模拟分割结果"""
        class MockMask:
            def __init__(self, data):
                self.data = MockTensor(data)

        class MockTensor:
            def __init__(self, data):
                self._data = data

            def cpu(self):
                return self

            def numpy(self):
                return self._data

            def item(self):
                return self._data.item()

            @property
            def shape(self):
                return self._data.shape

        class MockBox:
            def __init__(self, cls_id, conf):
                self.cls = MockTensor(np.array([cls_id]))
                self.conf = MockTensor(np.array([conf]))

        class MockResult:
            def __init__(self):
                self.masks = [MockMask(data) for data in mock_mask_data]
                self.boxes = [
                    MockBox(0, 0.9),
                    MockBox(1, 0.8),
                    MockBox(0, 0.85)
                ]
                self.names = {0: 'person', 1: 'car'}
                self.path = 'test.jpg'

            def plot(self):
                return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        return MockResult()

    def test_mask_statistics_calculation(self, mock_segmentation_result):
        """测试掩码统计计算"""
        result = DetectionResult(mock_segmentation_result, "segmentation")
        stats = result.get_statistics()

        # 验证基本统计
        assert stats['total_detections'] == 3
        assert stats['has_detections'] == True

        # 验证掩码统计
        assert stats['total_masks'] == 3
        assert len(stats['individual_areas']) == 3
        assert stats['individual_areas'] == [100.0, 64.0, 36.0]
        assert stats['individual_areas_sum'] == 200.0
        assert stats['largest_mask_area'] == 100.0
        assert stats['smallest_mask_area'] == 36.0

        # 验证覆盖率和重叠
        assert stats['actual_coverage_area'] > 0
        assert stats['overlap_info'] >= 0
        assert 0 <= stats['mask_coverage_ratio'] <= 1

    def test_mask_statistics_formatting(self, mock_segmentation_result):
        """测试掩码统计格式化"""
        result = DetectionResult(mock_segmentation_result, "segmentation")
        formatted = result.format_statistics()

        # 验证包含掩码统计信息
        assert "**掩码统计:**" in formatted
        assert "实例数量: 3" in formatted
        assert "个体面积:" in formatted
        assert "面积总和:" in formatted
        assert "最大实例:" in formatted
        assert "最小实例:" in formatted
        assert "覆盖率:" in formatted

        # 验证实际覆盖面积的HTML样式突出显示
        assert "<span style=" in formatted
        assert "background-color: #ffeb3b" in formatted
        assert "font-weight: bold" in formatted
        assert "实际覆盖:" in formatted

    def test_no_masks_case(self):
        """测试无掩码情况"""
        class NoMaskResult:
            def __init__(self):
                self.masks = None
                self.boxes = []
                self.names = {}
                self.path = None

        result = DetectionResult(NoMaskResult(), "detection")
        stats = result.get_statistics()

        # 验证无掩码时的默认值
        assert stats.get('total_masks', 0) == 0
        assert stats.get('individual_areas', []) == []
        assert stats.get('individual_areas_sum', 0) == 0

    def test_empty_masks_case(self):
        """测试空掩码列表情况"""
        class EmptyMaskResult:
            def __init__(self):
                self.masks = []
                self.boxes = []
                self.names = {}
                self.path = None

        result = DetectionResult(EmptyMaskResult(), "detection")
        stats = result.get_statistics()

        # 验证空掩码列表时的默认值
        assert stats.get('total_masks', 0) == 0
        assert stats.get('individual_areas', []) == []


class TestImageProcessor:
    """图像处理器测试"""
    
    def test_image_processor_initialization(self, test_config):
        """测试图像处理器初始化"""
        processor = ImageProcessor(test_config)
        assert processor.config == test_config
        assert isinstance(processor.supported_formats, list)
    
    def test_preprocess_image_numpy(self, sample_image_array):
        """测试numpy图像预处理"""
        processor = ImageProcessor()
        
        # 基本预处理
        processed = processor.preprocess_image(sample_image_array)
        assert processed is not None
        assert isinstance(processed, np.ndarray)
        
        # 调整尺寸
        processed = processor.preprocess_image(sample_image_array, target_size=(320, 240))
        assert processed.shape[:2] == (240, 320)
        
        # 归一化
        processed = processor.preprocess_image(sample_image_array, normalize=True)
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0
    
    def test_preprocess_image_file(self, sample_image_file):
        """测试文件图像预处理"""
        processor = ImageProcessor()
        processed = processor.preprocess_image(sample_image_file)
        
        assert processed is not None
        assert isinstance(processed, np.ndarray)
        assert len(processed.shape) == 3
    
    def test_validate_image_input(self, sample_image_file, sample_image_array):
        """测试图像输入验证"""
        processor = ImageProcessor()
        
        # 验证文件
        result = processor.validate_image_input(sample_image_file)
        assert result['valid'] == True
        assert 'info' in result
        
        # 验证numpy数组
        result = processor.validate_image_input(sample_image_array)
        assert result['valid'] == True
        
        # 验证无效输入
        result = processor.validate_image_input(None)
        assert result['valid'] == False
        assert result['error'] is not None
    
    def test_postprocess_result(self, sample_image_array):
        """测试结果后处理"""
        processor = ImageProcessor()
        
        # 基本后处理
        processed = processor.postprocess_result(sample_image_array)
        assert processed is not None
        assert processed.dtype == np.uint8
        
        # 格式转换
        processed = processor.postprocess_result(sample_image_array, output_format='BGR')
        assert processed is not None
    
    def test_save_image(self, sample_image_array, temp_dir):
        """测试图像保存"""
        processor = ImageProcessor()
        output_path = temp_dir / "saved_image.jpg"
        
        success = processor.save_image(sample_image_array, str(output_path))
        assert success == True
        assert output_path.exists()


class TestResultProcessor:
    """结果处理器测试"""
    
    def test_result_processor_initialization(self, test_config):
        """测试结果处理器初始化"""
        processor = ResultProcessor(test_config)
        assert processor.config == test_config
        assert isinstance(processor.results_history, list)
    
    def test_process_single_result(self, mock_detection_result):
        """测试单个结果处理"""
        processor = ResultProcessor()
        detection_result = DetectionResult(mock_detection_result, "detection")
        
        processed = processor.process_single_result(detection_result, "test_image.jpg")
        
        assert processed['detector_type'] == "detection"
        assert processed['image_path'] == "test_image.jpg"
        assert processed['has_detections'] == True
        assert 'statistics' in processed
        assert 'formatted_stats' in processed
    
    def test_process_batch_results(self, mock_detection_result):
        """测试批量结果处理"""
        processor = ResultProcessor()
        
        # 创建多个检测结果
        results = [
            DetectionResult(mock_detection_result, "detection"),
            DetectionResult(mock_detection_result, "detection")
        ]
        image_paths = ["image1.jpg", "image2.jpg"]
        
        batch_result = processor.process_batch_results(results, image_paths)
        
        assert batch_result['total_images'] == 2
        assert batch_result['successful_detections'] == 2
        assert batch_result['total_objects'] == 4  # 2 objects per image
        assert 'class_summary' in batch_result
    
    def test_export_results_to_csv(self, mock_detection_result, temp_dir):
        """测试CSV导出"""
        processor = ResultProcessor()
        detection_result = DetectionResult(mock_detection_result, "detection")
        processed = processor.process_single_result(detection_result, "test_image.jpg")
        
        csv_path = temp_dir / "test_results.csv"
        success = processor.export_results_to_csv(processed, str(csv_path))
        
        assert success == True
        assert csv_path.exists()
    
    def test_export_results_to_json(self, mock_detection_result, temp_dir):
        """测试JSON导出"""
        processor = ResultProcessor()
        detection_result = DetectionResult(mock_detection_result, "detection")
        processed = processor.process_single_result(detection_result, "test_image.jpg")
        
        json_path = temp_dir / "test_results.json"
        success = processor.export_results_to_json(processed, str(json_path))
        
        assert success == True
        assert json_path.exists()
    
    def test_generate_summary_report(self, mock_detection_result):
        """测试汇总报告生成"""
        processor = ResultProcessor()
        detection_result = DetectionResult(mock_detection_result, "detection")
        processed = processor.process_single_result(detection_result, "test_image.jpg")
        
        report = processor.generate_summary_report(processed)
        
        assert "# 检测结果汇总报告" in report
        assert "总图像数" in report
        assert "检测对象总数" in report


class TestBaseDetector:
    """基础检测器测试"""

    def test_object_detector_initialization(self, model_loader, test_config):
        """测试目标检测器初始化（使用具体类）"""
        detector = ObjectDetector(model_loader, test_config)

        assert detector.model_loader == model_loader
        assert detector.config == test_config
        assert detector.model is None
        assert detector.detector_type == "detection"

    def test_validate_input(self, model_loader, sample_image_file, sample_image_array):
        """测试输入验证"""
        detector = ObjectDetector(model_loader)

        # 有效输入
        assert detector._validate_input(sample_image_file) == True
        assert detector._validate_input(sample_image_array) == True

        # 无效输入
        assert detector._validate_input(None) == False
        assert detector._validate_input("/nonexistent/file.jpg") == False

    def test_get_detection_params(self, model_loader, test_config):
        """测试检测参数获取"""
        detector = ObjectDetector(model_loader, test_config)

        params = detector._get_detection_params(conf=0.5, custom_param="test")

        assert 'conf' in params
        assert params['conf'] == 0.5
        assert params['custom_param'] == "test"


@pytest.mark.integration
class TestDetectorIntegration:
    """检测器集成测试"""
    
    @pytest.mark.slow
    def test_object_detector_with_mock_model(self, model_loader, test_config):
        """测试目标检测器（使用模拟模型）"""
        detector = ObjectDetector(model_loader, test_config)
        
        # 模拟模型加载
        with patch.object(model_loader, 'load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            success = detector.load_model()
            assert success == True
            assert detector.model == mock_model
    
    @pytest.mark.slow
    def test_segmentation_detector_with_mock_model(self, model_loader, test_config):
        """测试分割检测器（使用模拟模型）"""
        detector = SegmentationDetector(model_loader, test_config)
        
        # 模拟模型加载
        with patch.object(model_loader, 'load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            success = detector.load_model()
            assert success == True
            assert detector.model == mock_model


class TestBatchProcessor:
    """批量处理器测试"""
    
    def test_batch_processor_initialization(self, model_loader, test_config):
        """测试批量处理器初始化"""
        detector = ObjectDetector(model_loader, test_config)
        processor = BatchProcessor(detector, test_config)
        
        assert processor.detector == detector
        assert processor.config == test_config
        assert isinstance(processor.max_workers, int)
        assert isinstance(processor.chunk_size, int)
    
    def test_estimate_processing_time(self, model_loader, test_config):
        """测试处理时间估算"""
        detector = ObjectDetector(model_loader, test_config)
        processor = BatchProcessor(detector, test_config)
        
        estimate = processor.estimate_processing_time(10, 0.5)
        
        assert estimate['total_images'] == 10
        assert estimate['estimated_seconds'] == 5.0
        assert estimate['estimated_minutes'] == 5.0 / 60
        assert estimate['avg_time_per_image'] == 0.5
    
    def test_get_processing_stats(self, model_loader, test_config):
        """测试获取处理统计"""
        detector = ObjectDetector(model_loader, test_config)
        processor = BatchProcessor(detector, test_config)
        
        stats = processor.get_processing_stats()
        
        assert 'max_workers' in stats
        assert 'chunk_size' in stats
        assert 'export_format' in stats
        assert 'detector_type' in stats
        assert 'model_loaded' in stats
