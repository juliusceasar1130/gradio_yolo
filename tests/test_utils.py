# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:45；主要修改内容：创建工具函数单元测试

"""
工具函数单元测试
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

from yolo_detector.utils.image_utils import (
    load_image, load_image_pil, convert_color_space, resize_image,
    get_image_info, format_image_info, validate_image_format, create_thumbnail
)
from yolo_detector.utils.file_utils import (
    get_image_files, ensure_dir, get_file_info, get_example_images,
    safe_filename, copy_file, move_file, delete_file, get_unique_filename
)
from yolo_detector.utils.exceptions import (
    YOLODetectorError, ConfigurationError, ModelLoadError,
    ErrorHandler, handle_exceptions, safe_execute
)


class TestImageUtils:
    """图像工具函数测试"""
    
    def test_load_image_pil(self, sample_image_file):
        """测试PIL图像加载"""
        image = load_image_pil(sample_image_file)
        assert image is not None
        assert isinstance(image, Image.Image)
        assert image.size == (640, 480)  # width, height
    
    def test_load_image_numpy(self, sample_image_file):
        """测试numpy图像加载"""
        image = load_image(sample_image_file)
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.shape == (480, 640, 3)  # height, width, channels
    
    def test_load_nonexistent_image(self):
        """测试加载不存在的图像"""
        image = load_image_pil('/nonexistent/image.jpg')
        assert image is None
        
        image = load_image('/nonexistent/image.jpg')
        assert image is None
    
    def test_convert_color_space(self, sample_image_array):
        """测试颜色空间转换"""
        # RGB to BGR
        bgr_image = convert_color_space(sample_image_array, 'RGB', 'BGR')
        assert bgr_image.shape == sample_image_array.shape
        
        # BGR to RGB (应该恢复原始)
        rgb_image = convert_color_space(bgr_image, 'BGR', 'RGB')
        np.testing.assert_array_equal(rgb_image, sample_image_array)
        
        # 相同格式转换
        same_image = convert_color_space(sample_image_array, 'RGB', 'RGB')
        np.testing.assert_array_equal(same_image, sample_image_array)
    
    def test_resize_image(self, sample_image_array):
        """测试图像尺寸调整"""
        # 保持宽高比
        resized = resize_image(sample_image_array, (320, 240), keep_aspect_ratio=True)
        assert resized.shape[:2] == (240, 320)  # height, width
        
        # 不保持宽高比
        resized = resize_image(sample_image_array, (320, 240), keep_aspect_ratio=False)
        assert resized.shape[:2] == (240, 320)
    
    def test_get_image_info_numpy(self, sample_image_array):
        """测试获取numpy图像信息"""
        info = get_image_info(sample_image_array)
        assert info['height'] == 480
        assert info['width'] == 640
        assert info['channels'] == 3
        assert 'size_bytes' in info
    
    def test_get_image_info_pil(self, sample_pil_image):
        """测试获取PIL图像信息"""
        info = get_image_info(sample_pil_image)
        assert info['width'] == 640
        assert info['height'] == 480
        assert info['mode'] == 'RGB'
    
    def test_get_image_info_file(self, sample_image_file):
        """测试获取文件图像信息"""
        info = get_image_info(sample_image_file)
        assert info['width'] == 640
        assert info['height'] == 480
        assert 'path' in info
    
    def test_format_image_info(self, sample_pil_image):
        """测试格式化图像信息"""
        formatted = format_image_info(sample_pil_image, "test.jpg")
        assert "**图片信息**" in formatted
        assert "test.jpg" in formatted
        assert "640×480" in formatted
    
    def test_validate_image_format(self, sample_image_file):
        """测试图像格式验证"""
        # 有效格式
        assert validate_image_format(sample_image_file) == True
        
        # 无效格式
        assert validate_image_format('/path/to/file.txt') == False
        
        # 不存在的文件
        assert validate_image_format('/nonexistent/file.jpg') == False
    
    def test_create_thumbnail(self, sample_pil_image):
        """测试创建缩略图"""
        thumbnail = create_thumbnail(sample_pil_image, (128, 128))
        assert thumbnail is not None
        assert isinstance(thumbnail, Image.Image)
        assert max(thumbnail.size) <= 128


class TestFileUtils:
    """文件工具函数测试"""
    
    def test_ensure_dir(self, temp_dir):
        """测试确保目录存在"""
        new_dir = temp_dir / "new_directory"
        assert not new_dir.exists()
        
        success = ensure_dir(new_dir)
        assert success == True
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_get_image_files(self, sample_image_files):
        """测试获取图像文件"""
        folder = Path(sample_image_files[0]).parent
        image_files = get_image_files(str(folder))

        # 至少应该包含我们创建的3个文件
        assert len(image_files) >= 3
        for file_path in image_files:
            assert Path(file_path).suffix.lower() == '.jpg'

        # 验证我们创建的文件都在列表中
        for sample_file in sample_image_files:
            assert sample_file in image_files
    
    def test_get_image_files_empty_folder(self, temp_dir):
        """测试空文件夹"""
        empty_folder = temp_dir / "empty"
        empty_folder.mkdir()
        
        image_files = get_image_files(str(empty_folder))
        assert len(image_files) == 0
    
    def test_get_file_info(self, sample_image_file):
        """测试获取文件信息"""
        info = get_file_info(sample_image_file)
        
        assert 'name' in info
        assert 'size_bytes' in info
        assert 'absolute_path' in info
        assert info['is_file'] == True
        assert info['is_dir'] == False
    
    def test_get_example_images(self, sample_image_files):
        """测试获取示例图像"""
        folder = Path(sample_image_files[0]).parent
        examples = get_example_images(str(folder), max_count=2)
        
        assert len(examples) <= 2
        assert len(examples) > 0
    
    def test_safe_filename(self):
        """测试安全文件名生成"""
        # 包含非法字符的文件名
        unsafe_name = 'file<>:"/\\|?*.txt'
        safe_name = safe_filename(unsafe_name)
        
        assert '<' not in safe_name
        assert '>' not in safe_name
        assert ':' not in safe_name
        assert '"' not in safe_name
        assert '/' not in safe_name
        assert '\\' not in safe_name
        assert '|' not in safe_name
        assert '?' not in safe_name
        assert '*' not in safe_name
    
    def test_copy_file(self, sample_image_file, temp_dir):
        """测试文件复制"""
        dest_path = temp_dir / "copied_image.jpg"
        success = copy_file(sample_image_file, dest_path)
        
        assert success == True
        assert dest_path.exists()
        assert dest_path.is_file()
    
    def test_get_unique_filename(self, temp_dir):
        """测试获取唯一文件名"""
        # 创建一个文件
        original_file = temp_dir / "test.txt"
        original_file.touch()
        
        # 获取唯一文件名
        unique_path = get_unique_filename(original_file)
        
        assert unique_path != original_file
        assert not unique_path.exists()
        assert unique_path.stem.startswith("test_")


class TestExceptions:
    """异常处理测试"""
    
    def test_yolo_detector_error(self):
        """测试基础异常类"""
        error = YOLODetectorError("Test error", "TEST_ERROR", {"key": "value"})
        
        assert str(error) == "Test error (Code: TEST_ERROR, Details: {'key': 'value'})"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}
        
        error_dict = error.to_dict()
        assert error_dict['error_type'] == 'YOLODetectorError'
        assert error_dict['error_code'] == 'TEST_ERROR'
        assert error_dict['message'] == 'Test error'
    
    def test_specific_exceptions(self):
        """测试特定异常类"""
        # 配置错误
        config_error = ConfigurationError("Config error", config_key="test.key")
        assert config_error.error_code == "CONFIG_ERROR"
        assert config_error.details['config_key'] == "test.key"
        
        # 模型加载错误
        model_error = ModelLoadError("Model error", model_path="/test/model.pt", model_type="detection")
        assert model_error.error_code == "MODEL_LOAD_ERROR"
        assert model_error.details['model_path'] == "/test/model.pt"
        assert model_error.details['model_type'] == "detection"
    
    def test_error_handler(self):
        """测试错误处理器"""
        handler = ErrorHandler()
        
        # 处理异常
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_record = handler.handle_error(e, {"context": "test"})
        
        assert error_record['error_type'] == 'ValueError'
        assert error_record['message'] == 'Test error'
        assert error_record['context']['context'] == 'test'
        
        # 检查统计
        stats = handler.get_error_stats()
        assert stats['total_errors'] == 1
        assert 'ValueError' in stats['error_types']
    
    def test_handle_exceptions_decorator(self):
        """测试异常处理装饰器"""
        handler = ErrorHandler()
        
        @handle_exceptions(error_handler=handler, reraise=False, default_return="error")
        def failing_function():
            raise ValueError("Function failed")
        
        result = failing_function()
        assert result == "error"
        
        stats = handler.get_error_stats()
        assert stats['total_errors'] == 1
    
    def test_safe_execute(self):
        """测试安全执行函数"""
        def success_function(x, y):
            return x + y
        
        def failing_function():
            raise ValueError("Function failed")
        
        # 成功执行
        success, result = safe_execute(success_function, 1, 2)
        assert success == True
        assert result == 3
        
        # 失败执行
        success, result = safe_execute(failing_function)
        assert success == False
        assert 'error' in result


@pytest.mark.unit
class TestUtilsIntegration:
    """工具函数集成测试"""
    
    def test_image_processing_pipeline(self, sample_image_file, temp_dir):
        """测试图像处理流水线"""
        # 加载图像
        pil_image = load_image_pil(sample_image_file)
        assert pil_image is not None
        
        # 获取信息
        info = get_image_info(pil_image)
        assert info['width'] > 0
        assert info['height'] > 0
        
        # 创建缩略图
        thumbnail = create_thumbnail(pil_image, (100, 100))
        assert thumbnail is not None
        assert max(thumbnail.size) <= 100
        
        # 保存缩略图
        thumb_path = temp_dir / "thumbnail.jpg"
        thumbnail.save(thumb_path)
        assert thumb_path.exists()
    
    def test_file_operations_pipeline(self, sample_image_file, temp_dir):
        """测试文件操作流水线"""
        # 获取文件信息
        info = get_file_info(sample_image_file)
        assert info['is_file'] == True
        
        # 复制文件
        dest_path = temp_dir / "copied.jpg"
        success = copy_file(sample_image_file, dest_path)
        assert success == True
        
        # 获取唯一文件名
        unique_path = get_unique_filename(dest_path)
        assert unique_path != dest_path
        
        # 验证格式
        assert validate_image_format(str(dest_path)) == True
