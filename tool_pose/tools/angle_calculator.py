#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
关键点角度计算工具模块

主要修改内容：
1. 创建角度计算工具模块，支持从配置文件加载角度定义
2. 实现三点角度计算（中间点为顶点）
3. 实现图像角度标注功能
修改时间：2025年1月28日
最后更新：2025年11月2日
"""

import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import platform
import os

# 导入PIL用于文本绘制（支持Unicode字符）
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def load_angle_config(config_path: str) -> Dict[str, Any]:
    """
    加载角度配置文件
    
    Args:
        config_path: 角度配置文件路径
        
    Returns:
        角度配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"角度配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_system_font_path() -> Optional[str]:
    """
    获取系统字体路径（支持Unicode）
    
    Returns:
        字体文件路径，如果找不到返回None
    """
    system = platform.system()
    
    if system == 'Windows':
        # Windows字体路径
        font_paths = [
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",     # 黑体
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:  # Linux
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    
    # 返回第一个存在的字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
    
    return None


def draw_text_with_pil(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int = 20,
    color: Tuple[int, int, int] = (0, 255, 0),
    bg_color: Optional[Tuple[int, int, int]] = None,
    bg_alpha: float = 0.6
) -> np.ndarray:
    """
    使用PIL绘制文本（支持Unicode字符如°符号）
    
    Args:
        image: OpenCV图像 (BGR格式)
        text: 要绘制的文本（支持Unicode）
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文本颜色 (B, G, R)
        bg_color: 背景颜色 (B, G, R)，如果为None则不绘制背景
        bg_alpha: 背景透明度 (0-1)
        
    Returns:
        绘制后的图像
    """
    if not PIL_AVAILABLE:
        # 如果PIL不可用，降级使用cv2.putText（但°符号可能显示为??）
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return image
    
    # 将OpenCV图像转换为PIL图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 加载字体
    font_path = get_system_font_path()
    try:
        if font_path:
            font = ImageFont.truetype(font_path, size=font_size)
        else:
            # 使用默认字体（可能不支持°符号）
            font = ImageFont.load_default()
    except Exception:
        # 字体加载失败，使用默认字体
        font = ImageFont.load_default()
    
    # 获取文本边界框（基于实际绘制位置）
    bbox = draw.textbbox(position, text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 绘制背景（如果需要）
    if bg_color is not None:
        padding = 5
        # 基于实际文本边界框计算背景矩形位置
        bg_rect = [
            bbox[0] - padding,  # 左边界
            bbox[1] - padding,  # 上边界（文本的顶部）
            bbox[2] + padding,  # 右边界
            bbox[3] + padding   # 下边界（文本的底部）
        ]
        
        # 创建半透明背景层
        bg_overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg_overlay)
        bg_rgb = tuple(reversed(bg_color))  # BGR转RGB
        bg_draw.rectangle(bg_rect, fill=(bg_rgb[0], bg_rgb[1], bg_rgb[2], int(255 * bg_alpha)))
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), bg_overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_image)
    
    # 绘制文本（在背景之上）
    text_rgb = tuple(reversed(color))  # BGR转RGB
    draw.text(position, text, fill=text_rgb, font=font)
    
    # 转换回OpenCV格式
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image


def load_tool_pose_config(config_path: str) -> Dict[str, Any]:
    """
    加载工具姿态配置文件（tool_pose.yaml）
    
    Args:
        config_path: 工具姿态配置文件路径
        
    Returns:
        工具姿态配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"工具姿态配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_keypoint_index_by_name(kpt_name: str, kpt_names_list: List[str]) -> Optional[int]:
    """
    通过关键点名称字符串匹配获取关键点索引
    
    Args:
        kpt_name: 关键点名称（如 't1_1', 't2_2'）
        kpt_names_list: 关键点名称列表（如 ['t1_1', 't1_2', ...]）
        
    Returns:
        关键点索引，如果未找到返回 None
    """
    try:
        index = kpt_names_list.index(kpt_name)
        return index
    except ValueError:
        return None


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三点角度（p2为顶点）
    
    Args:
        p1: 起点坐标 [x, y]
        p2: 顶点坐标 [x, y]
        p3: 终点坐标 [x, y]
        
    Returns:
        角度值（度），范围 [0, 180]
    """
    # 转换为numpy数组
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    p3 = np.array(p3, dtype=np.float64)
    
    # 计算向量
    vec1 = p1 - p2  # 从顶点指向起点
    vec2 = p3 - p2  # 从顶点指向终点
    
    # 计算向量长度
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # 避免除零
    if norm1 == 0 or norm2 == 0:
        return None
    
    # 计算点积
    dot_product = np.dot(vec1, vec2)
    
    # 计算夹角（弧度）
    cos_angle = dot_product / (norm1 * norm2)
    
    # 限制cos值范围，避免数值误差
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # 转换为角度（度）
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)


def calculate_angles_for_object(
    class_name: str,
    kpt_data: np.ndarray,
    kpt_names: List[str],
    angle_config: Dict[str, Any],
    visibility_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    为单个对象计算所有角度
    
    Args:
        class_name: 类别名称（如 'tool1', 'tool2'）
        kpt_data: 关键点数据 [num_keypoints, 3]，每行为 [x, y, visibility]
        kpt_names: 关键点名称列表
        angle_config: 角度配置字典
        visibility_threshold: 可见性阈值
        
    Returns:
        角度计算结果字典
    """
    angles_result = {}
    
    # 获取该类别的角度配置
    if class_name not in angle_config:
        return {'error': f'类别 {class_name} 在角度配置中不存在'}
    
    class_angles = angle_config[class_name]
    
    # 遍历每个角度定义
    for angle_name, keypoint_names in class_angles.items():
        if len(keypoint_names) != 3:
            angles_result[angle_name] = {
                'value': None,
                'keypoints': keypoint_names,
                'valid': False,
                'reason': '角度定义必须包含3个关键点'
            }
            continue
        
        kpt1_name, kpt2_name, kpt3_name = keypoint_names
        
        # 获取关键点索引
        idx1 = get_keypoint_index_by_name(kpt1_name, kpt_names)
        idx2 = get_keypoint_index_by_name(kpt2_name, kpt_names)
        idx3 = get_keypoint_index_by_name(kpt3_name, kpt_names)
        
        # 检查索引是否有效
        if idx1 is None or idx2 is None or idx3 is None:
            missing = []
            if idx1 is None:
                missing.append(kpt1_name)
            if idx2 is None:
                missing.append(kpt2_name)
            if idx3 is None:
                missing.append(kpt3_name)
            
            angles_result[angle_name] = {
                'value': None,
                'keypoints': keypoint_names,
                'valid': False,
                'reason': f'关键点未找到: {", ".join(missing)}'
            }
            continue
        
        # 检查索引范围
        if idx1 >= len(kpt_data) or idx2 >= len(kpt_data) or idx3 >= len(kpt_data):
            angles_result[angle_name] = {
                'value': None,
                'keypoints': keypoint_names,
                'valid': False,
                'reason': '关键点索引超出范围'
            }
            continue
        
        # 获取关键点坐标和可见性
        kpt1 = kpt_data[idx1]
        kpt2 = kpt_data[idx2]
        kpt3 = kpt_data[idx3]
        
        x1, y1, vis1 = kpt1[0], kpt1[1], kpt1[2]
        x2, y2, vis2 = kpt2[0], kpt2[1], kpt2[2]
        x3, y3, vis3 = kpt3[0], kpt3[1], kpt3[2]
        
        # 检查关键点可见性
        if vis1 < visibility_threshold or vis2 < visibility_threshold or vis3 < visibility_threshold:
            missing_kpts = []
            if vis1 < visibility_threshold:
                missing_kpts.append(kpt1_name)
            if vis2 < visibility_threshold:
                missing_kpts.append(kpt2_name)
            if vis3 < visibility_threshold:
                missing_kpts.append(kpt3_name)
            
            angles_result[angle_name] = {
                'value': None,
                'keypoints': keypoint_names,
                'keypoint_indices': [idx1, idx2, idx3],
                'keypoint_positions': {
                    kpt1_name: [float(x1), float(y1)],
                    kpt2_name: [float(x2), float(y2)],
                    kpt3_name: [float(x3), float(y3)]
                },
                'valid': False,
                'reason': f'关键点不可见: {", ".join(missing_kpts)}'
            }
            continue
        
        # 检查坐标是否有效（不为0）
        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0) or (x3 == 0 and y3 == 0):
            angles_result[angle_name] = {
                'value': None,
                'keypoints': keypoint_names,
                'valid': False,
                'reason': '关键点坐标为(0,0)，可能未检测到'
            }
            continue
        
        # 计算角度
        try:
            angle_value = calculate_angle([x1, y1], [x2, y2], [x3, y3])
            
            angles_result[angle_name] = {
                'value': angle_value,
                'keypoints': keypoint_names,
                'keypoint_indices': [int(idx1), int(idx2), int(idx3)],
                'keypoint_positions': {
                    kpt1_name: [float(x1), float(y1)],
                    kpt2_name: [float(x2), float(y2)],
                    kpt3_name: [float(x3), float(y3)]
                },
                'valid': True  # Python原生bool，JSON可序列化
            }
        except Exception as e:
            angles_result[angle_name] = {
                'value': None,
                'keypoints': keypoint_names,
                'valid': False,
                'reason': f'角度计算失败: {str(e)}'
            }
    
    return angles_result


# ========== 弧度绘制功能（已禁用，待日后开发） ==========
# 
# 主要修改内容：
# 1. 弧度绘制功能已注释，待日后开发
# 修改时间：2025年1月31日
# 最后更新：2025年11月2日
#
# def draw_angle_arc(
#     image: np.ndarray,
#     p1: Tuple[float, float],
#     p2: Tuple[float, float],
#     p3: Tuple[float, float],
#     radius: int = 30,
#     color: Tuple[int, int, int] = (0, 255, 0),
#     thickness: int = 2,
#     angle_value: Optional[float] = None
# ) -> None:
#     """
#     在图像上绘制角度弧线（仅绘制内角，即 < 180度）
#     
#     主要修改内容：
#     1. 确保只绘制内角（< 180度）的弧线
#     2. 通过计算两个方向的角度差，选择内角方向（较小的角度差）
#     3. 直接使用角度差来确定起始和结束角度，避免方向错误
#     4. 通过验证扫过角度确保不会绘制外角
#     修改时间：2025年1月31日
#     最后更新：2025年11月2日
#     
#     Args:
#         image: 图像数组 (BGR格式)
#         p1: 起点坐标 (x, y)
#         p2: 顶点坐标 (x, y)
#         p3: 终点坐标 (x, y)
#         radius: 弧线半径（像素）
#         color: 颜色 (B, G, R)
#         thickness: 线条粗细
#         angle_value: 角度值（度），内角值（0-180度），用于选择更匹配的方向
#     """
#     # 确保角度值在有效范围内（内角 < 180度）
#     if angle_value is not None and angle_value >= 180:
#         return
#     p1 = np.array(p1, dtype=np.float64)
#     p2 = np.array(p2, dtype=np.float64)
#     p3 = np.array(p3, dtype=np.float64)
#     
#     # 计算向量（从顶点指向两个端点）
#     vec1 = p1 - p2  # 从顶点指向起点
#     vec2 = p3 - p2  # 从顶点指向终点
#     
#     # 归一化向量
#     norm1 = np.linalg.norm(vec1)
#     norm2 = np.linalg.norm(vec2)
#     
#     if norm1 == 0 or norm2 == 0:
#         return
#     
#     # 计算角度值（如果没有提供）
#     if angle_value is None:
#         vec1_unit = vec1 / norm1
#         vec2_unit = vec2 / norm2
#         dot_product = np.dot(vec1_unit, vec2_unit)
#         cos_angle = np.clip(dot_product, -1.0, 1.0)
#         angle_rad = np.arccos(cos_angle)
#         angle_value = np.degrees(angle_rad)
#     
#     # 计算两个向量相对于正X轴的角度（OpenCV格式：0度在右侧，顺时针为正）
#     # arctan2返回逆时针角度，需要转换为顺时针
#     angle1_rad = np.arctan2(vec1[1], vec1[0])  # vec1的方向（逆时针）
#     angle2_rad = np.arctan2(vec2[1], vec2[0])  # vec2的方向（逆时针）
#     
#     # 转换为OpenCV格式（顺时针角度，0度在右侧）
#     angle1_cv = -np.degrees(angle1_rad) % 360
#     angle2_cv = -np.degrees(angle2_rad) % 360
#     
#     # 计算两个方向的角度差
#     diff_cw = (angle2_cv - angle1_cv) % 360      # 顺时针方向差（从vec1到vec2）
#     diff_ccw = (angle1_cv - angle2_cv) % 360     # 逆时针方向差（从vec2到vec1）
#     
#     # 内角是较小的角度差（必须 < 180度）
#     # 选择内角方向，并确保该方向更接近angle_value
#     if diff_cw < 180:
#         if diff_ccw < 180:
#             # 两个方向都是内角，选择更接近angle_value的
#             if abs(diff_cw - angle_value) <= abs(diff_ccw - angle_value):
#                 # 内角在顺时针方向（从vec1到vec2）
#                 start_angle = angle1_cv
#                 end_angle = angle2_cv  # 直接使用angle2_cv，扫过角度 = diff_cw
#             else:
#                 # 内角在逆时针方向（从vec2到vec1），反向绘制
#                 start_angle = angle2_cv
#                 end_angle = angle1_cv  # 直接使用angle1_cv，扫过角度 = diff_ccw
#         else:
#             # 只有顺时针方向是内角
#             start_angle = angle1_cv
#             end_angle = angle2_cv
#     elif diff_ccw < 180:
#         # 只有逆时针方向是内角，反向绘制
#         start_angle = angle2_cv
#         end_angle = angle1_cv
#     else:
#         # 两个方向都不是内角（不应该发生，因为angle_value < 180）
#         # 使用angle_value强制计算，选择较小的方向
#         if diff_cw <= diff_ccw:
#             start_angle = angle1_cv
#             end_angle = angle2_cv
#         else:
#             start_angle = angle2_cv
#             end_angle = angle1_cv
#     
#     # 验证扫过角度（必须是内角 < 180度）
#     sweep_angle = (end_angle - start_angle) % 360
#     
#     # 如果扫过角度大于等于180度，说明绘制了外角，需要反向
#     if sweep_angle >= 180:
#         # 交换起始和结束角度，绘制反向弧线
#         start_angle, end_angle = end_angle, start_angle
#         sweep_angle = (end_angle - start_angle) % 360
#     
#     # 绘制弧线
#     try:
#         cv2.ellipse(
#             image,
#             tuple(map(int, p2)),  # 中心点（顶点）
#             (radius, radius),    # 轴长（圆形）
#             0,                    # 旋转角度（0度）
#             start_angle,          # 起始角度（度，OpenCV顺时针）
#             end_angle,            # 结束角度（度，OpenCV顺时针）
#             color,
#             thickness
#         )
#     except Exception as e:
#         # 如果绘制失败，静默跳过
#         pass


def annotate_angles_on_image(
    image: np.ndarray,
    angles_result: Dict[str, Any],
    keypoints_xy: np.ndarray,
    keypoint_indices_map: Dict[str, int],
    font_scale: float = 0.6,
    font_thickness: int = 2,
    arc_radius: int = 25  # 参数保留但暂不使用（弧度绘制功能已禁用）
) -> np.ndarray:
    """
    在图像上标注角度值（仅显示文本和连线，弧度绘制已禁用）
    
    主要修改内容：
    1. 弧度绘制功能已禁用，待日后开发
    2. 仅显示角度文本标注和关键点连线
    修改时间：2025年1月31日
    最后更新：2025年11月2日
    
    Args:
        image: 图像数组 (BGR格式)
        angles_result: 角度计算结果字典
        keypoints_xy: 关键点坐标数组 [num_keypoints, 2]
        keypoint_indices_map: 关键点名称到索引的映射
        font_scale: 字体大小
        font_thickness: 字体粗细（暂不使用）
        arc_radius: 角度弧线半径（已禁用，保留用于日后开发）
        
    Returns:
        标注后的图像
    """
    # 复制图像，避免修改原图
    annotated_image = image.copy()
    
    # 定义颜色
    valid_color = (0, 255, 0)  # 绿色 (BGR)
    invalid_color = (0, 0, 255)  # 红色 (BGR)
    
    # 遍历每个角度
    for angle_name, angle_info in angles_result.items():
        if not angle_info.get('valid', False):
            continue
        
        # 获取关键点位置
        kpt_names = angle_info['keypoints']
        if len(kpt_names) != 3:
            continue
        
        kpt1_name, kpt2_name, kpt3_name = kpt_names
        
        # 获取关键点坐标
        kpt_positions = angle_info.get('keypoint_positions', {})
        if not kpt_positions:
            continue
        
        p1 = tuple(map(int, kpt_positions[kpt1_name]))
        p2 = tuple(map(int, kpt_positions[kpt2_name]))  # 顶点
        p3 = tuple(map(int, kpt_positions[kpt3_name]))
        
        angle_value = angle_info['value']
        if angle_value is None:
            continue
        
        # 绘制关键点连线（p1->p2和p2->p3）
        line_color = (100, 200, 255)  # 浅蓝色连线 (BGR)
        line_thickness = 1
        cv2.line(annotated_image, p1, p2, line_color, line_thickness, cv2.LINE_AA)
        cv2.line(annotated_image, p2, p3, line_color, line_thickness, cv2.LINE_AA)
        
        # 弧度绘制功能已禁用，待日后开发
        # draw_angle_arc(annotated_image, p1, p2, p3, radius=arc_radius, color=valid_color, angle_value=angle_value)
        
        # 准备角度文本（两位小数，使用正确的度数符号）
        angle_text = f"{angle_name}: {angle_value:.2f}°"
        
        # 计算文本位置（在顶点附近，稍微偏移避免遮挡）
        text_offset_x = 15
        text_offset_y = -15
        text_pos = (p2[0] + text_offset_x, p2[1] + text_offset_y)
        
        # 使用PIL绘制文本（支持Unicode字符°符号）
        font_size = int(font_scale * 30)  # 将font_scale转换为像素大小
        annotated_image = draw_text_with_pil(
            annotated_image,
            angle_text,
            text_pos,
            font_size=font_size,
            color=valid_color,
            bg_color=(0, 0, 0),  # 黑色背景
            bg_alpha=0.6
        )
    
    return annotated_image

