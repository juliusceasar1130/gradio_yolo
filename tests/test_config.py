# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:45；主要修改内容：创建配置管理模块单元测试

"""
配置管理模块单元测试
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from yolo_detector.config.settings import Config, get_config, reload_config


class TestConfig:
    """配置类测试"""
    
    def test_config_initialization(self, test_config):
        """测试配置初始化"""
        assert test_config is not None
        assert hasattr(test_config, '_config')
        assert isinstance(test_config._config, dict)
    
    def test_get_basic_config(self, test_config):
        """测试基本配置获取"""
        # 测试存在的配置
        models_config = test_config.get('models')
        assert models_config is not None
        assert isinstance(models_config, dict)
        
        # 测试不存在的配置
        non_existent = test_config.get('non_existent_key', 'default_value')
        assert non_existent == 'default_value'
    
    def test_get_nested_config(self, test_config):
        """测试嵌套配置获取"""
        # 测试点号分隔的键
        detection_model = test_config.get('models.detection.path')
        assert detection_model is not None
        
        # 测试不存在的嵌套键
        non_existent = test_config.get('models.non_existent.key', 'default')
        assert non_existent == 'default'
    
    def test_specialized_config_methods(self, test_config):
        """测试专用配置方法"""
        # 测试模型配置
        detection_config = test_config.get_model_config('detection')
        assert isinstance(detection_config, dict)
        
        # 测试检测配置
        detection_params = test_config.get_detection_config()
        assert isinstance(detection_params, dict)
        
        # 测试UI配置
        ui_config = test_config.get_ui_config()
        assert isinstance(ui_config, dict)
        
        # 测试数据配置
        data_config = test_config.get_data_config()
        assert isinstance(data_config, dict)
    
    def test_config_set(self, test_config):
        """测试配置设置"""
        # 设置简单值
        test_config.set('test.key', 'test_value')
        assert test_config.get('test.key') == 'test_value'
        
        # 设置嵌套值
        test_config.set('test.nested.key', 42)
        assert test_config.get('test.nested.key') == 42
    
    def test_config_with_custom_file(self, temp_dir):
        """测试自定义配置文件"""
        # 创建临时配置文件
        config_data = {
            'models': {
                'detection': {
                    'path': '/test/model.pt',
                    'type': 'detection'
                }
            },
            'data': {
                'input_folder': '/test/input',
                'output_folder': '/test/output'
            }
        }
        
        config_file = temp_dir / "test_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)
        
        # 加载自定义配置
        config = Config(str(config_file))
        
        assert config.get('models.detection.path') == '/test/model.pt'
        assert config.get('data.input_folder') == '/test/input'
    
    def test_config_validation(self, temp_dir):
        """测试配置验证"""
        # 创建不完整的配置文件
        incomplete_config = {
            'models': {
                'detection': {
                    'path': '/nonexistent/model.pt'
                }
            }
        }
        
        config_file = temp_dir / "incomplete_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(incomplete_config, f)
        
        # 配置应该能够加载，但会有警告
        config = Config(str(config_file))
        assert config is not None
    
    def test_config_save(self, test_config, temp_dir):
        """测试配置保存"""
        # 修改配置
        test_config.set('test.save_key', 'save_value')
        
        # 保存到临时文件
        save_path = temp_dir / "saved_config.yaml"
        test_config.save(str(save_path))
        
        # 验证文件存在
        assert save_path.exists()
        
        # 验证内容
        with open(save_path, 'r', encoding='utf-8') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['test']['save_key'] == 'save_value'
    
    def test_config_reload(self, test_config):
        """测试配置重新加载"""
        # 修改配置
        original_value = test_config.get('models.detection.path')
        test_config.set('models.detection.path', '/modified/path')
        
        # 重新加载
        test_config.reload()
        
        # 验证配置已恢复
        reloaded_value = test_config.get('models.detection.path')
        assert reloaded_value == original_value


class TestGlobalConfig:
    """全局配置测试"""
    
    def test_get_global_config(self):
        """测试获取全局配置"""
        config1 = get_config()
        config2 = get_config()
        
        # 应该返回同一个实例
        assert config1 is config2
    
    def test_reload_global_config(self):
        """测试重新加载全局配置"""
        config_before = get_config()
        reload_config()
        config_after = get_config()
        
        # 配置对象应该是同一个，但内容可能已更新
        assert config_before is config_after


class TestConfigErrorHandling:
    """配置错误处理测试"""
    
    def test_nonexistent_config_file(self):
        """测试不存在的配置文件"""
        with pytest.raises(FileNotFoundError):
            Config('/nonexistent/config.yaml')
    
    def test_invalid_yaml_file(self, temp_dir):
        """测试无效的YAML文件"""
        invalid_yaml = temp_dir / "invalid.yaml"
        with open(invalid_yaml, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            Config(str(invalid_yaml))
    
    def test_empty_config_file(self, temp_dir):
        """测试空配置文件"""
        empty_config = temp_dir / "empty.yaml"
        empty_config.touch()

        # 空配置文件应该能够处理，会自动添加默认节
        config = Config(str(empty_config))
        assert config is not None
        assert isinstance(config._config, dict)
        # 验证自动添加的默认节
        assert 'models' in config._config
        assert 'data' in config._config
        assert 'detection' in config._config
        assert 'ui' in config._config


@pytest.mark.unit
class TestConfigPerformance:
    """配置性能测试"""
    
    def test_config_access_performance(self, test_config):
        """测试配置访问性能"""
        import time
        
        # 测试大量配置访问
        start_time = time.time()
        for _ in range(1000):
            test_config.get('models.detection.path')
        end_time = time.time()
        
        # 1000次访问应该在合理时间内完成
        assert (end_time - start_time) < 1.0  # 1秒内
    
    def test_nested_config_access(self, test_config):
        """测试嵌套配置访问"""
        # 测试深层嵌套访问
        test_config.set('level1.level2.level3.level4.key', 'deep_value')
        
        value = test_config.get('level1.level2.level3.level4.key')
        assert value == 'deep_value'
