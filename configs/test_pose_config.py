#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试pose配置读取

验证从default.yaml读取工具关键点配置是否正确
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.yolo_detector.config.settings import get_config
    from src.yolo_detector.core.detector import get_pose_config
    
    def test_config_loading():
        """测试配置加载"""
        print("=" * 60)
        print("测试: 从default.yaml读取工具关键点配置")
        print("=" * 60)
        
        try:
            # 测试配置对象
            config = get_config()
            print("✓ 配置对象创建成功")
            
            # 测试读取pose配置
            pose_config = config.get_pose_config()
            print(f"✓ pose配置读取成功: {len(pose_config)} 个配置项")
            
            # 测试读取工具关键点配置
            tool_config = config.get_tool_pose_config()
            print(f"✓ 工具关键点配置读取成功")
            print(f"  - kpt_shape: {tool_config['kpt_shape']}")
            print(f"  - num_classes: {tool_config['num_classes']}")
            print(f"  - class_names: {tool_config['class_names']}")
            print(f"  - tool1关键点数: {len(tool_config['kpt_names'][0])}")
            print(f"  - tool2关键点数: {len(tool_config['kpt_names'][1])}")
            print(f"  - flip_idx: {tool_config['flip_idx']}")
            print(f"  - pose_weights: {tool_config['pose_weights']}")
            
            # 测试get_pose_config函数
            pose_func_result = get_pose_config()
            print(f"✓ get_pose_config()函数调用成功")
            print(f"  - default_kpt_names: {pose_func_result['default_kpt_names']}")
            print(f"  - skeleton连接数: {len(pose_func_result['skeleton'])}")
            
            print("\n" + "=" * 60)
            print("所有测试通过！配置读取正常。")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == '__main__':
        success = test_config_loading()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

