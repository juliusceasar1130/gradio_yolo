# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:45；主要修改内容：创建pytest配置和测试夹具

"""
pytest配置文件

定义测试夹具和配置
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import sys
import os

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from yolo_detector.config.settings import Config
from yolo_detector.models.model_loader import ModelLoader
from yolo_detector.utils import setup_logging, setup_error_handling


@pytest.fixture(scope="session")
def project_root_path():
    """项目根目录路径"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_config(project_root_path):
    """测试配置"""
    config_path = project_root_path / "configs" / "default.yaml"
    if config_path.exists():
        return Config(str(config_path))
    else:
        # 创建最小配置用于测试
        return Config()


@pytest.fixture(scope="session")
def temp_dir():
    """临时目录"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image_array():
    """示例图像数组"""
    # 创建一个简单的RGB图像
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_pil_image():
    """示例PIL图像"""
    # 创建一个简单的RGB图像
    image_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(image_array, 'RGB')


@pytest.fixture
def sample_image_file(temp_dir, sample_pil_image):
    """示例图像文件"""
    image_path = temp_dir / "test_image.jpg"
    sample_pil_image.save(image_path, "JPEG")
    return str(image_path)


@pytest.fixture
def sample_image_files(temp_dir):
    """多个示例图像文件"""
    image_files = []
    for i in range(3):
        # 创建不同的图像
        image_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pil_image = Image.fromarray(image_array, 'RGB')
        
        image_path = temp_dir / f"test_image_{i}.jpg"
        pil_image.save(image_path, "JPEG")
        image_files.append(str(image_path))
    
    return image_files


@pytest.fixture
def mock_detection_result():
    """模拟检测结果"""
    class MockBox:
        def __init__(self, cls_id, conf, bbox):
            self.cls = MockTensor(cls_id)
            self.conf = MockTensor(conf)
            self.xyxy = MockTensor(bbox)
    
    class MockTensor:
        def __init__(self, value):
            self.value = value
        
        def item(self):
            return self.value
        
        def tolist(self):
            if isinstance(self.value, list):
                return [self.value]
            return [self.value]
    
    class MockResult:
        def __init__(self):
            self.boxes = [
                MockBox(0, 0.85, [100, 100, 200, 300]),
                MockBox(1, 0.92, [300, 150, 500, 350])
            ]
            self.masks = None
            self.names = {0: 'person', 1: 'car'}
            self.path = 'test_image.jpg'
        
        def plot(self):
            # 返回一个模拟的可视化图像
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    return MockResult()


@pytest.fixture
def model_loader(test_config):
    """模型加载器"""
    return ModelLoader(test_config)


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """设置测试日志"""
    setup_logging()
    setup_error_handling()


@pytest.fixture
def output_dir(temp_dir):
    """输出目录"""
    output_path = temp_dir / "outputs"
    output_path.mkdir(exist_ok=True)
    return output_path


# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# 跳过条件
def pytest_collection_modifyitems(config, items):
    """修改测试项目"""
    # 检查是否有模型文件
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "default.yaml"
    
    skip_model_tests = True
    if config_path.exists():
        try:
            test_config = Config(str(config_path))
            detection_model = test_config.get('models.detection.path')
            if detection_model and os.path.exists(detection_model):
                skip_model_tests = False
        except:
            pass
    
    # 为需要模型的测试添加跳过标记
    skip_marker = pytest.mark.skip(reason="Model files not available")
    for item in items:
        if "model" in item.nodeid.lower() and skip_model_tests:
            item.add_marker(skip_marker)
