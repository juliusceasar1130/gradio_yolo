#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块（从YAML文件加载）

主要修改内容：
1. 统一从 angle_config.yaml 加载所有配置
2. 配置结构清晰醒目，方便修改
3. 支持默认值和配置验证
修改时间：2025年1月31日
最后更新：2025年11月2日
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml


@dataclass
class Config:
    """
    配置类：统一管理所有配置项（从YAML文件加载）
    """
    # 路径配置
    model_path: str
    image_path: str
    config_path: Path
    output_dir: Path
    
    # 预测参数
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = ""
    half: bool = False
    max_det: int = 1000
    
    # 输出配置
    show_image: bool = True
    save_image: bool = False
    save_json: bool = True
    
    def validate(self) -> tuple:
        """
        验证配置
        
        Returns:
            (是否有效: bool, 错误信息: Optional[str])
        """
        if not Path(self.model_path).exists():
            return False, f"模型文件不存在: {self.model_path}"
        
        if not Path(self.image_path).exists():
            return False, f"图片文件不存在: {self.image_path}"
        
        if not self.config_path.exists():
            return False, f"配置文件不存在: {self.config_path}"
        
        if self.conf < 0 or self.conf > 1:
            return False, f"置信度阈值应在 [0, 1] 范围内: {self.conf}"
        
        if self.iou < 0 or self.iou > 1:
            return False, f"IoU阈值应在 [0, 1] 范围内: {self.iou}"
        
        if self.imgsz <= 0:
            return False, f"图像尺寸应大于 0: {self.imgsz}"
        
        return True, None


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    从YAML配置文件加载配置
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        配置对象
    """
    # 确定配置文件路径
    if config_path is None:
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "angle_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 加载YAML配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 提取路径配置
    paths = config_data.get('paths', {})
    model_path = paths.get('model', '')
    image_path = paths.get('image', '')
    output_dir_str = paths.get('output', 'outputs')
    
    # 处理输出目录（相对路径转换为绝对路径）
    if Path(output_dir_str).is_absolute():
        output_dir = Path(output_dir_str)
    else:
        script_dir = config_path.parent
        output_dir = script_dir / output_dir_str
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取预测参数
    predict = config_data.get('predict', {})
    conf = predict.get('conf', 0.25)
    iou = predict.get('iou', 0.45)
    imgsz = predict.get('imgsz', 640)
    device = predict.get('device', '')
    half = predict.get('half', False)
    max_det = predict.get('max_det', 1000)
    
    # 提取输出选项
    output_opts = config_data.get('output', {})
    show_image = output_opts.get('show_image', True)
    save_image = output_opts.get('save_image', False)
    save_json = output_opts.get('save_json', True)
    
    # 创建配置对象
    return Config(
        model_path=model_path,
        image_path=image_path,
        config_path=config_path,
        output_dir=output_dir,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        half=half,
        max_det=max_det,
        show_image=show_image,
        save_image=save_image,
        save_json=save_json
    )


def get_default_config() -> Config:
    """
    获取默认配置（从YAML文件加载）
    
    Returns:
        配置对象
    """
    return load_config()


def get_config_dict(config: Config) -> Dict[str, Any]:
    """
    获取配置字典（用于传递给其他模块）
    
    Args:
        config: 配置对象
        
    Returns:
        包含所有配置的字典（包含角度配置等）
    """
    # 加载完整的YAML配置
    with open(config.config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    
    return full_config
