# 创建者/修改者: chenliang
# 修改时间：2025年1月27日
# 主要修改内容：
# 1. 分离预测参数和可视化参数，避免混淆
# 2. 更新DetectionResult类支持可视化参数传递
# 3. 修改PoseDetector类正确处理分离的参数
# 4. 确保预测参数只包含model.predict()方法的有效参数
# 5. 将硬编码的姿态配置移到配置文件中，提高可维护性
# 
# 历史修改：2025年7月28日 - 增加分割掩码统计功能

"""
YOLO检测器核心模块

提供统一的目标检测和图像分割功能
"""

import cv2
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

from ..models.model_loader import ModelLoader
from ..utils.image_utils import convert_color_space, get_image_info
from ..config.settings import get_config

logger = logging.getLogger(__name__)

# 从配置文件获取姿态检测相关配置
def get_pose_config():
    """获取姿态检测配置"""
    try:
        config = get_config()
        pose_config = config.get('pose', {})
        keypoints_config = pose_config.get('keypoints', {})
        
        # 获取配置值，如果不存在则使用默认值
        names = keypoints_config.get('names', [
            "鼻子", "左眼", "右眼", "左耳", "右耳",
            "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
            "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"
        ])
        
        groups = keypoints_config.get('groups', {
            "头部": [0, 1, 2, 3, 4],
            "躯干": [5, 6, 11, 12],
            "上肢": [5, 6, 7, 8, 9, 10],
            "下肢": [11, 12, 13, 14, 15, 16]
        })
        
        skeleton = keypoints_config.get('skeleton', [
            [0, 1], [0, 2], [1, 3], [2, 4],  # 头部
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 上肢
            [5, 11], [6, 12], [11, 12],  # 躯干
            [11, 13], [13, 15], [12, 14], [14, 16]  # 下肢
        ])
        
        # 验证配置
        validate_pose_config(names, groups, skeleton)
        
        return {
            'names': names,
            'groups': groups,
            'skeleton': skeleton
        }
        
    except Exception as e:
        logger.warning(f"无法从配置文件读取姿态配置，使用默认配置: {e}")
        # 返回默认配置
        return {
            'names': [
                "鼻子", "左眼", "右眼", "左耳", "右耳",
                "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
                "左髋", "右髋", "左膝", "右膝", "左踝", "右踝"
            ],
            'groups': {
                "头部": [0, 1, 2, 3, 4],
                "躯干": [5, 6, 11, 12],
                "上肢": [5, 6, 7, 8, 9, 10],
                "下肢": [11, 12, 13, 14, 15, 16]
            },
            'skeleton': [
                [0, 1], [0, 2], [1, 3], [2, 4],  # 头部
                [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 上肢
                [5, 11], [6, 12], [11, 12],  # 躯干
                [11, 13], [13, 15], [12, 14], [14, 16]  # 下肢
            ]
        }

def validate_pose_config(names, groups, skeleton):
    """验证姿态配置的有效性"""
    try:
        # 验证关键点名称
        if not isinstance(names, list) or len(names) != 17:
            raise ValueError(f"关键点名称必须是包含17个元素的列表，当前: {len(names)}")
        
        # 验证关键点分组
        if not isinstance(groups, dict):
            raise ValueError("关键点分组必须是字典类型")
        
        for group_name, indices in groups.items():
            if not isinstance(indices, list):
                raise ValueError(f"分组 '{group_name}' 的索引必须是列表")
            for idx in indices:
                if not isinstance(idx, int) or idx < 0 or idx >= len(names):
                    raise ValueError(f"分组 '{group_name}' 包含无效索引: {idx}")
        
        # 验证骨架连接
        if not isinstance(skeleton, list):
            raise ValueError("骨架连接必须是列表类型")
        
        for connection in skeleton:
            if not isinstance(connection, list) or len(connection) != 2:
                raise ValueError(f"骨架连接必须是包含2个元素的列表: {connection}")
            for idx in connection:
                if not isinstance(idx, int) or idx < 0 or idx >= len(names):
                    raise ValueError(f"骨架连接包含无效索引: {idx}")
        
        logger.info("姿态配置验证通过")
        
    except Exception as e:
        logger.error(f"姿态配置验证失败: {e}")
        raise

# 获取姿态配置
POSE_CONFIG = get_pose_config()
KEYPOINT_NAMES = POSE_CONFIG['names']
KEYPOINT_GROUPS = POSE_CONFIG['groups']
SKELETON_CONNECTIONS = POSE_CONFIG['skeleton']


class DetectionResult:
    """检测结果类"""
    
    def __init__(self, raw_result, detector_type: str, visualization_params: Optional[Dict[str, Any]] = None):
        """
        初始化检测结果
        
        Args:
            raw_result: YOLO原始检测结果
            detector_type: 检测器类型
            visualization_params: 可视化参数
        """
        self.raw_result = raw_result
        self.detector_type = detector_type
        self.visualization_params = visualization_params or {}
        self._processed = False
        #缓存机制，避免重复计算
        self._statistics = None
        self._visualization = None
    
    # 这个@property装饰器用于将boxes方法变成属性调用方式，使得可以通过result.boxes而不是result.boxes()来获取检测框
    @property
    def boxes(self):
        """获取检测框"""
        return getattr(self.raw_result, 'boxes', None)
    
    # 返回的属性名称是：masks
    @property
    def masks(self):
        """获取分割掩码"""
        return getattr(self.raw_result, 'masks', None)
    
    @property
    def keypoints(self):
        """获取关键点"""
        return getattr(self.raw_result, 'keypoints', None)
    
    @property
    def names(self):
        """获取类别名称"""
        return getattr(self.raw_result, 'names', {})
    
    @property
    def path(self):
        """获取图像路径"""
        return getattr(self.raw_result, 'path', None)
    
    # “箭头”在这里指的是Python函数定义中的“->”符号，表示该函数的返回类型注解。
    # 例如：def get_statistics(self) -> Dict[str, Any]: 
    # 其中“-> Dict[str, Any]”的意思是该函数返回一个字典，键为字符串，值为任意类型。
    def get_statistics(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        if self._statistics is None:
            self._statistics = self._calculate_statistics()
        return self._statistics
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """计算检测统计信息"""
        stats = {
            'total_detections': 0,
            'classes': {},
            'confidence_scores': [],
            'has_detections': False
        }

        try:
            if self.boxes is not None and len(self.boxes) > 0:
                stats['has_detections'] = True
                stats['total_detections'] = len(self.boxes)

                # 统计各类别数量
                for box in self.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = self.names.get(cls_id, f'class_{cls_id}')
                    confidence = float(box.conf.item())

                    if cls_name in stats['classes']:
                        stats['classes'][cls_name]['count'] += 1
                        stats['classes'][cls_name]['confidences'].append(confidence)
                    else:
                        stats['classes'][cls_name] = {
                            'count': 1,
                            'confidences': [confidence]
                        }

                    stats['confidence_scores'].append(confidence)

                # 计算平均置信度
                if stats['confidence_scores']:
                    stats['avg_confidence'] = np.mean(stats['confidence_scores'])
                    stats['min_confidence'] = np.min(stats['confidence_scores'])
                    stats['max_confidence'] = np.max(stats['confidence_scores'])

            # 计算掩码统计信息（如果是分割模型）
            if self.masks is not None and len(self.masks) > 0:
                mask_stats = self._calculate_mask_statistics()
                # update() 是 Python 字典的内置方法，用于将另一个字典的键值对合并到当前字典中。如果有相同的键，则会用新字典中的值覆盖原有值。
                stats.update(mask_stats)

            # 计算关键点统计信息（如果是姿态检测模型）
            if self.keypoints is not None and len(self.keypoints) > 0:
                keypoint_stats = self._calculate_keypoint_statistics()
                stats.update(keypoint_stats)

            logger.debug(f"检测统计: {stats['total_detections']} 个对象")
            return stats

        except Exception as e:
            logger.error(f"计算检测统计失败: {e}")
            return stats

    def _calculate_mask_statistics(self) -> Dict[str, Any]:
        """计算掩码统计信息"""
        mask_stats = {
            'total_masks': 0,
            'individual_areas': [],
            'individual_areas_sum': 0,
            'actual_coverage_area': 0,
            'largest_mask_area': 0,
            'smallest_mask_area': 0,
            'mask_coverage_ratio': 0.0,
            'overlap_info': 0
        }

        try:
            if self.masks is None or len(self.masks) == 0:
                return mask_stats

            mask_stats['total_masks'] = len(self.masks)
            individual_areas = []
            all_masks_combined = None

            # 获取图像尺寸
            image_height, image_width = self.masks[0].data.shape[-2:]
            total_image_pixels = image_height * image_width

            # 计算每个掩码的面积
            for i, mask in enumerate(self.masks):
                try:
                    # 获取掩码数据并转换为numpy数组
                    mask_data = mask.data.cpu().numpy()
                    if len(mask_data.shape) == 3:
                        mask_data = mask_data[0]  # 去除batch维度

                    # 计算面积（阈值0.5）
                    binary_mask = mask_data > 0.5
                    area = float(np.sum(binary_mask))
                    individual_areas.append(area)

                    # 合并所有掩码用于计算实际覆盖面积
                    if all_masks_combined is None:
                        all_masks_combined = binary_mask.copy()
                    else:
                        all_masks_combined = np.logical_or(all_masks_combined, binary_mask)

                except Exception as e:
                    logger.warning(f"处理第{i}个掩码时出错: {e}")
                    continue

            # 计算统计指标
            if individual_areas:
                mask_stats['individual_areas'] = individual_areas
                mask_stats['individual_areas_sum'] = sum(individual_areas)
                mask_stats['largest_mask_area'] = max(individual_areas)
                mask_stats['smallest_mask_area'] = min(individual_areas)

                # 计算实际覆盖面积（去重叠）
                if all_masks_combined is not None:
                    mask_stats['actual_coverage_area'] = float(np.sum(all_masks_combined))

                    # 计算重叠信息
                    mask_stats['overlap_info'] = mask_stats['individual_areas_sum'] - mask_stats['actual_coverage_area']

                    # 计算覆盖率
                    mask_stats['mask_coverage_ratio'] = mask_stats['actual_coverage_area'] / total_image_pixels

            logger.debug(f"掩码统计: {mask_stats['total_masks']} 个掩码, 总面积: {mask_stats['individual_areas_sum']}")
            return mask_stats

        except Exception as e:
            logger.error(f"计算掩码统计失败: {e}")
            return mask_stats

    def _calculate_keypoint_statistics(self) -> Dict[str, Any]:
        """计算关键点统计信息"""
        keypoint_stats = {
            'total_persons': 0,
            'total_keypoints': 0,
            'visible_keypoints': 0,
            'keypoint_groups': {},
            'person_keypoints': [],
            'avg_keypoint_confidence': 0.0,
            'keypoint_details': []
        }

        try:
            if self.keypoints is None or len(self.keypoints) == 0:
                return keypoint_stats

            keypoint_stats['total_persons'] = len(self.keypoints)
            all_keypoint_confidences = []
            person_keypoint_counts = []

            # 初始化关键点分组统计
            for group_name in KEYPOINT_GROUPS.keys():
                keypoint_stats['keypoint_groups'][group_name] = {
                    'total': 0,
                    'visible': 0,
                    'avg_confidence': 0.0
                }

            # 处理每个人的关键点
            for person_idx, person_keypoints in enumerate(self.keypoints):
                try:
                    # 获取关键点数据
                    keypoint_data = person_keypoints.data.cpu().numpy()
                    if len(keypoint_data.shape) == 3:
                        keypoint_data = keypoint_data[0]  # 去除batch维度

                    person_visible_count = 0
                    person_confidences = []
                    person_details = []

                    # 处理每个关键点
                    for kpt_idx, (x, y, conf) in enumerate(keypoint_data):
                        if kpt_idx >= len(KEYPOINT_NAMES):
                            break
                            
                        keypoint_stats['total_keypoints'] += 1
                        person_confidences.append(conf)
                        all_keypoint_confidences.append(conf)

                        # 判断关键点是否可见（置信度阈值）
                        is_visible = conf > 0.5  # 使用配置的阈值
                        if is_visible:
                            keypoint_stats['visible_keypoints'] += 1
                            person_visible_count += 1

                        # 记录关键点详细信息
                        person_details.append({
                            'name': KEYPOINT_NAMES[kpt_idx],
                            'x': float(x),
                            'y': float(y),
                            'confidence': float(conf),
                            'visible': is_visible
                        })

                        # 更新分组统计
                        for group_name, group_indices in KEYPOINT_GROUPS.items():
                            if kpt_idx in group_indices:
                                keypoint_stats['keypoint_groups'][group_name]['total'] += 1
                                if is_visible:
                                    keypoint_stats['keypoint_groups'][group_name]['visible'] += 1

                    person_keypoint_counts.append(person_visible_count)
                    keypoint_stats['person_keypoints'].append({
                        'person_id': person_idx,
                        'visible_count': person_visible_count,
                        'total_count': len(keypoint_data),
                        'avg_confidence': np.mean(person_confidences) if person_confidences else 0.0,
                        'details': person_details
                    })

                except Exception as e:
                    logger.warning(f"处理第{person_idx}个人的关键点时出错: {e}")
                    continue

            # 计算平均置信度
            if all_keypoint_confidences:
                keypoint_stats['avg_keypoint_confidence'] = np.mean(all_keypoint_confidences)

            # 计算各分组的平均置信度
            for group_name, group_indices in KEYPOINT_GROUPS.items():
                group_confidences = []
                for person_kpts in keypoint_stats['person_keypoints']:
                    for detail in person_kpts['details']:
                        if detail['name'] in [KEYPOINT_NAMES[i] for i in group_indices]:
                            group_confidences.append(detail['confidence'])
                
                if group_confidences:
                    keypoint_stats['keypoint_groups'][group_name]['avg_confidence'] = np.mean(group_confidences)

            logger.debug(f"关键点统计: {keypoint_stats['total_persons']} 人, {keypoint_stats['visible_keypoints']} 个可见关键点")
            return keypoint_stats

        except Exception as e:
            logger.error(f"计算关键点统计失败: {e}")
            return keypoint_stats

    # 该函数的作用是将检测和掩码的统计信息格式化为可读的文本字符串，便于展示检测结果的概要，包括检测到的对象数量、平均置信度、掩码面积、实际覆盖面积、重叠面积、最大/最小实例面积、覆盖率以及各类别的统计信息等。
    def format_statistics(self) -> str:
        """格式化统计信息为文本"""
        stats = self.get_statistics()

        if not stats['has_detections']:
            return "**检测结果统计**\n未检测到任何对象"

        text_parts = ["**检测结果统计**"]
        text_parts.append(f"总计: {stats['total_detections']} 个对象")

        if stats.get('avg_confidence'):
            text_parts.append(f"平均置信度: {stats['avg_confidence']:.2f}")

        # 添加掩码统计信息（如果有）
        if stats.get('total_masks', 0) > 0:
            text_parts.append("\n**掩码统计:**")
            text_parts.append(f"- 实例数量: {stats['total_masks']}")

            if stats.get('individual_areas'):
                areas_str = ', '.join([f"{int(area)}" for area in stats['individual_areas']])
                text_parts.append(f"- 个体面积: [{areas_str}] 像素")
                text_parts.append(f"- 面积总和: {int(stats['individual_areas_sum'])} 像素")

                # 突出显示实际覆盖面积 - 使用HTML样式
                actual_coverage = int(stats['actual_coverage_area'])
                text_parts.append(f"- <span style='background-color: #ffeb3b; color: #000; font-weight: bold; padding: 2px 6px; border-radius: 3px;'>实际覆盖: {actual_coverage} 像素</span>")

                if stats['overlap_info'] > 0:
                    text_parts.append(f"- 重叠面积: {int(stats['overlap_info'])} 像素")

                text_parts.append(f"- 最大实例: {int(stats['largest_mask_area'])} 像素")
                text_parts.append(f"- 最小实例: {int(stats['smallest_mask_area'])} 像素")
                text_parts.append(f"- 覆盖率: {stats['mask_coverage_ratio']:.1%}")

        # 添加关键点统计信息（如果有）
        if stats.get('total_persons', 0) > 0:
            text_parts.append("\n**姿态检测统计:**")
            text_parts.append(f"- 检测到人数: {stats['total_persons']}")
            text_parts.append(f"- 总关键点数: {stats['total_keypoints']}")
            text_parts.append(f"- 可见关键点: {stats['visible_keypoints']}")
            text_parts.append(f"- 关键点平均置信度: {stats['avg_keypoint_confidence']:.2f}")
            
            # 关键点分组统计
            text_parts.append("\n**关键点分组统计:**")
            for group_name, group_stats in stats['keypoint_groups'].items():
                visible_ratio = group_stats['visible'] / group_stats['total'] if group_stats['total'] > 0 else 0
                text_parts.append(f"- {group_name}: {group_stats['visible']}/{group_stats['total']} 可见 ({visible_ratio:.1%})")
            
            # 个人关键点详情
            text_parts.append("\n**个人关键点详情:**")
            for person_kpts in stats['person_keypoints']:
                text_parts.append(f"- 人员{person_kpts['person_id']+1}: {person_kpts['visible_count']}/{person_kpts['total_count']} 关键点可见")

        text_parts.append("\n**类别统计:**")
        for cls_name, cls_info in stats['classes'].items():
            count = cls_info['count']
            avg_conf = np.mean(cls_info['confidences'])
            text_parts.append(f"- {cls_name}: {count}个 (置信度: {avg_conf:.2f})")

        return "\n".join(text_parts)
    
    # 该函数用于获取检测结果的可视化图像。如果尚未生成可视化图像，则会调用内部方法创建。支持可选地将图像从BGR格式转换为RGB格式，便于在不同环境下正确显示。返回值为可视化的numpy数组，或在无法生成时返回None。
    def get_visualization(self, convert_to_rgb: bool = True) -> Optional[np.ndarray]:
        """获取可视化结果"""
        if self._visualization is None:
            self._visualization = self._create_visualization()
        
        if self._visualization is not None and convert_to_rgb:
            # 确保返回RGB格式
            if len(self._visualization.shape) == 3 and self._visualization.shape[2] == 3:
                # 检查是否需要BGR到RGB转换
                return convert_color_space(self._visualization, 'BGR', 'RGB')
        
        return self._visualization
    
    def _create_visualization(self) -> Optional[np.ndarray]:
        """创建可视化图像"""
        try:
            # 这里判断self.raw_result对象是否有'plot'方法，用于确定是否支持可视化绘制
            if hasattr(self.raw_result, 'plot'):
                # 使用可视化参数调用plot方法
                return self.raw_result.plot(**self.visualization_params)
            else:
                logger.warning("检测结果不支持可视化")
                return None
        except Exception as e:
            logger.error(f"创建可视化失败: {e}")
            return None


class BaseDetector(ABC):
    """检测器基类"""
    
    def __init__(self, model_loader: ModelLoader, config=None):
        """
        初始化检测器
        
        Args:
            model_loader: 模型加载器
            config: 配置对象
        """
        self.model_loader = model_loader
        self.config = config
        self.model = None
        self.detector_type = "base"
    
    @abstractmethod
    def load_model(self) -> bool:
        """加载模型"""
        pass
    
    @abstractmethod
    def detect(self, image: Union[np.ndarray, Image.Image, str], **kwargs) -> Optional[DetectionResult]:
        """执行检测"""
        pass
    
    def _validate_input(self, image: Union[np.ndarray, Image.Image, str]) -> bool:
        """验证输入图像"""
        if image is None:
            logger.warning("输入图像为None")
            return False
        
        if isinstance(image, str):
            # 文件路径
            import os
            if not os.path.exists(image):
                logger.error(f"图像文件不存在: {image}")
                return False
        elif isinstance(image, np.ndarray):
            # numpy数组
            if image.size == 0:
                logger.error("输入图像数组为空")
                return False
        elif isinstance(image, Image.Image):
            # PIL图像
            if image.size == (0, 0):
                logger.error("输入PIL图像尺寸为0")
                return False
        else:
            logger.error(f"不支持的图像类型: {type(image)}")
            return False
        
        return True
    
    def _get_detection_params(self, **kwargs) -> Dict[str, Any]:
        """获取检测参数"""
        params = {}
        
        # 从配置获取默认参数
        if self.config:
            detection_config = self.config.get_detection_config()
            params.update(detection_config)
        
        # 用传入的参数覆盖默认值
        params.update(kwargs)
        
        return params
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None


class ObjectDetector(BaseDetector):
    """目标检测器"""
    
    def __init__(self, model_loader: ModelLoader, config=None):
        super().__init__(model_loader, config)
        self.detector_type = "detection"
    
    def load_model(self) -> bool:
        """加载检测模型"""
        try:
            self.model = self.model_loader.load_model('detection')
            if self.model:
                logger.info("目标检测模型加载成功")
                return True
            else:
                logger.error("目标检测模型加载失败")
                return False
        except Exception as e:
            logger.error(f"加载目标检测模型异常: {e}")
            return False
    
    def detect(self, image: Union[np.ndarray, Image.Image, str], **kwargs) -> Optional[DetectionResult]:
        """
        执行目标检测
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
                - conf: 置信度阈值 (默认: 0.25)
                - max_det: 最大检测数量 (默认: 1000)
                - iou: IoU阈值 (默认: 0.45)
                
        Returns:
            检测结果对象
        """
        if not self._validate_input(image):
            return None
        
        if not self.is_model_loaded():
            if not self.load_model():
                return None
        
        try:
            # 获取检测参数
            params = self._get_detection_params(**kwargs)
            
            # 执行检测
            logger.debug(f"开始目标检测，参数: {params}")
            results = self.model.predict(
                source=image,
                conf=params.get('confidence_threshold', 0.25),
                max_det=params.get('max_detections', 1000),
                iou=params.get('nms_threshold', 0.45),
                verbose=False
            )
            
            if results and len(results) > 0:
                # 获取可视化参数
                visualization_params = {}
                if 'visualization' in params:
                    visualization_params = params['visualization'].copy()
                
                result = DetectionResult(results[0], self.detector_type, visualization_params)
                logger.info("目标检测完成")
                return result
            else:
                logger.warning("检测结果为空")
                return None
                
        except Exception as e:
            logger.error(f"目标检测失败: {e}")
            return None


class SegmentationDetector(BaseDetector):
    """图像分割检测器"""
    
    def __init__(self, model_loader: ModelLoader, config=None):
        super().__init__(model_loader, config)
        self.detector_type = "segmentation"
    
    def load_model(self) -> bool:
        """加载分割模型"""
        try:
            self.model = self.model_loader.load_model('segmentation')
            if self.model:
                logger.info("图像分割模型加载成功")
                return True
            else:
                logger.error("图像分割模型加载失败")
                return False
        except Exception as e:
            logger.error(f"加载图像分割模型异常: {e}")
            return False
    
    def detect(self, image: Union[np.ndarray, Image.Image, str], **kwargs) -> Optional[DetectionResult]:
        """
        执行图像分割
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
                - conf: 置信度阈值 (默认: 0.25)
                - retina_masks: 是否使用高分辨率掩码 (默认: True)
                - show_boxes: 是否显示边界框 (默认: False)
                
        Returns:
            检测结果对象
        """
        if not self._validate_input(image):
            return None
        
        if not self.is_model_loaded():
            if not self.load_model():
                return None
        
        try:
            # 获取检测参数
            params = self._get_detection_params(**kwargs)
            
            # 获取分割特定参数
            if self.config:
                seg_config = self.config.get_segmentation_config()
                params.update(seg_config)
            
            # 执行分割
            logger.debug(f"开始图像分割，参数: {params}")
            results = self.model.predict(
                source=image,
                conf=params.get('confidence_threshold', 0.25),
                retina_masks=params.get('retina_masks', True),
                show_boxes=params.get('show_boxes', False),
                verbose=False
            )
            
            if results and len(results) > 0:
                # 获取可视化参数
                visualization_params = {}
                if 'visualization' in params:
                    visualization_params = params['visualization'].copy()
                
                result = DetectionResult(results[0], self.detector_type, visualization_params)
                logger.info("图像分割完成")
                return result
            else:
                logger.warning("分割结果为空")
                return None
                
        except Exception as e:
            logger.error(f"图像分割失败: {e}")
            return None


class PoseDetector(BaseDetector):
    """姿态检测器"""
    
    def __init__(self, model_loader: ModelLoader, config=None):
        super().__init__(model_loader, config)
        self.detector_type = "pose"
    
    def load_model(self) -> bool:
        """加载姿态检测模型"""
        try:
            self.model = self.model_loader.load_model('pose')
            if self.model:
                logger.info("姿态检测模型加载成功")
                return True
            else:
                logger.error("姿态检测模型加载失败")
                return False
        except Exception as e:
            logger.error(f"加载姿态检测模型异常: {e}")
            return False
    
    def detect(self, image: Union[np.ndarray, Image.Image, str], **kwargs) -> Optional[DetectionResult]:
        """
        执行姿态检测
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
                - conf: 置信度阈值 (默认: 0.25)
                - max_det: 最大检测数量 (默认: 1000)
                - iou: IoU阈值 (默认: 0.45)
                - imgsz: 图像尺寸 (默认: 640)
                - device: 设备选择 (默认: 自动选择)
                - half: 半精度推理 (默认: False)
                
        Returns:
            检测结果对象
        """
        if not self._validate_input(image):
            return None
        
        if not self.is_model_loaded():
            if not self.load_model():
                return None
        
        try:
            # 获取检测参数
            params = self._get_detection_params(**kwargs)
            
            # 获取姿态检测特定参数
            if self.config:
                pose_config = self.config.get_pose_config()
                params.update(pose_config)
            
            # 分离预测参数和可视化参数
            prediction_params = {}
            visualization_params = {}
            
            # 预测参数 - 只包含model.predict()方法的有效参数
            prediction_params = {
                'conf': params.get('confidence_threshold', 0.25),
                'max_det': params.get('max_det', 1000),
                'iou': params.get('iou', 0.45),
                'imgsz': params.get('imgsz', 640),
                'device': params.get('device', ''),
                'half': params.get('half', False),
                'verbose': False
            }
            
            # 可视化参数 - 用于plot()方法
            if 'visualization' in params:
                visualization_params = params['visualization'].copy()
            
            # 执行姿态检测
            logger.debug(f"开始姿态检测，预测参数: {prediction_params}")
            results = self.model.predict(source=image, **prediction_params)
            
            if results and len(results) > 0:
                result = DetectionResult(results[0], self.detector_type, visualization_params)
                logger.info("姿态检测完成")
                return result
            else:
                logger.warning("姿态检测结果为空")
                return None
                
        except Exception as e:
            logger.error(f"姿态检测失败: {e}")
            return None


class ClassificationDetector(BaseDetector):
    """图像分类检测器"""
    
    def __init__(self, model_loader: ModelLoader, config=None):
        super().__init__(model_loader, config)
        self.detector_type = "classification"
    
    def load_model(self) -> bool:
        """加载分类模型"""
        try:
            self.model = self.model_loader.load_model('classification')
            if self.model:
                logger.info("图像分类模型加载成功")
                return True
            else:
                logger.error("图像分类模型加载失败")
                return False
        except Exception as e:
            logger.error(f"加载图像分类模型异常: {e}")
            return False
    
    def detect(self, image: Union[np.ndarray, Image.Image, str], **kwargs) -> Optional[DetectionResult]:
        """
        执行图像分类
        
        Args:
            image: 输入图像
            **kwargs: 检测参数
                - conf: 置信度阈值 (默认: 0.25)
                - imgsz: 图像尺寸 (默认: 640)
                - device: 设备选择 (默认: 自动选择)
                - half: 半精度推理 (默认: False)
                
        Returns:
            检测结果对象
        """
        if not self._validate_input(image):
            return None
        
        if not self.is_model_loaded():
            if not self.load_model():
                return None
        
        try:
            # 获取检测参数
            params = self._get_detection_params(**kwargs)
            
            # 获取分类特定参数
            if self.config:
                classification_config = self.config.get_classification_config()
                params.update(classification_config)
            
            # 分离预测参数和可视化参数
            prediction_params = {}
            visualization_params = {}
            
            # 预测参数 - 只包含model.predict()方法的有效参数
            prediction_params = {
                'conf': params.get('confidence_threshold', 0.25),
                'imgsz': params.get('imgsz', 640),
                'device': params.get('device', ''),
                'half': params.get('half', False),
                'verbose': False
            }
            
            # 可视化参数 - 用于plot()方法
            if 'visualization' in params:
                visualization_params = params['visualization'].copy()
            
            # 执行图像分类
            logger.debug(f"开始图像分类，预测参数: {prediction_params}")
            results = self.model.predict(source=image, **prediction_params)
            
            if results and len(results) > 0:
                result = DetectionResult(results[0], self.detector_type, visualization_params)
                logger.info("图像分类完成")
                return result
            else:
                logger.warning("分类结果为空")
                return None
                
        except Exception as e:
            logger.error(f"图像分类失败: {e}")
            return None
