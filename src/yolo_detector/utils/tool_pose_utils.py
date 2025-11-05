# 创建者/修改者: chenliang
# 修改时间：2025年11月02日 10:30
# 主要修改内容：
# 1. 创建工具关键点检测封装模块
# 2. 复用 tool_pose/demo 中的检测逻辑
# 3. 提供简洁的统一接口
# 4. 延迟加载模型，优化性能

"""
工具关键点检测封装模块

封装 tool_pose/demo 中的检测逻辑，提供简洁统一的接口
复用：模型加载器、预测器、角度计算、结果保存等功能
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from PIL import Image
import cv2
import logging

from .logger import get_logger

logger = get_logger(__name__)

# 工具关键点检测相关模块已迁移到 src/yolo_detector 目录
# 不再需要从 tool_pose/demo 导入


class ToolPoseDetector:
    """
    工具关键点检测器
    
    封装 tool_pose/demo 中的检测逻辑，提供统一的接口
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化工具关键点检测器
        
        Args:
            config_path: 工具关键点配置文件路径，如果为None则使用默认路径
                        (src/yolo_detector/config/tool_pose_config.yaml)
        """
        # 确定配置文件路径（默认从 src/yolo_detector/config/tool_pose_config.yaml）
        if config_path is None:
            # 默认配置文件路径（相对于当前模块位置）
            config_dir = Path(__file__).parent.parent / "config"
            config_path = config_dir / "tool_pose_config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"角度配置文件不存在: {config_path}")
        
        self.config_path = config_path
        
        # 延迟加载的组件
        self._model = None
        self._model_path = None
        self._angle_config = None
        self._class_names = None
        self._kpt_names_dict = None
        self._angles_config = None
        self._predict_config = None
        
        # 导入依赖模块（已迁移到 src/yolo_detector）
        try:
            from ..core.tool_pose_model_loader import ToolPoseModelLoader
            from ..core.tool_pose_predictor import ToolPosePredictor
            from .tool_pose_angle_calculator import (
                load_angle_config,
                calculate_angles_for_object,
                annotate_angles_on_image
            )
            from .tool_pose_json_utils import save_results_to_json
            
            self.ModelLoader = ToolPoseModelLoader
            self.Predictor = ToolPosePredictor
            self.load_angle_config = load_angle_config
            self.calculate_angles_for_object = calculate_angles_for_object
            self.annotate_angles_on_image = annotate_angles_on_image
            self.save_results_to_json = save_results_to_json
            
            logger.info(f"工具关键点检测器初始化完成，配置文件: {config_path}")
            
            # 加载角度配置
            self._load_angle_config()
            
        except ImportError as e:
            logger.error(f"导入工具关键点检测模块失败: {e}", exc_info=True)
            raise ImportError(f"无法导入工具关键点检测模块，请确保模块文件存在: {e}")
    
    def _load_angle_config(self):
        """加载角度配置文件"""
        try:
            self._angle_config = self.load_angle_config(str(self.config_path))
            self._class_names = self._angle_config.get('names', {})
            self._kpt_names_dict = self._angle_config.get('kpt_names', {})
            
            # 提取角度定义
            self._angles_config = self._angle_config.get('angles', {})
            if not self._angles_config:
                # 旧格式兼容
                self._angles_config = {
                    k: v for k, v in self._angle_config.items()
                    if k not in ['names', 'kpt_names', 'paths', 'predict', 'output']
                }
            
            # 提取预测配置（用于模型预测参数）
            paths = self._angle_config.get('paths', {})
            predict = self._angle_config.get('predict', {})
            
            self._model_path = paths.get('model', '')
            self._predict_config = {
                'conf': predict.get('conf', 0.25),
                'iou': predict.get('iou', 0.45),
                'imgsz': predict.get('imgsz', 640),
                'device': predict.get('device', ''),
                'half': predict.get('half', False),
                'max_det': predict.get('max_det', 1000)
            }
            
            logger.info(f"角度配置文件加载完成: {self.config_path}")
            
        except Exception as e:
            logger.error(f"加载角度配置文件失败: {e}", exc_info=True)
            raise
    
    def load_model(self) -> bool:
        """
        加载模型（延迟加载）
        
        Returns:
            是否成功
        """
        try:
            if self._model is None:
                if not self._model_path:
                    raise ValueError("模型路径未配置，请检查 angle_config.yaml")
                
                if not Path(self._model_path).exists():
                    raise FileNotFoundError(f"模型文件不存在: {self._model_path}")
                
                logger.info(f"正在加载模型: {self._model_path}")
                loader = self.ModelLoader()
                self._model = loader.get_model(
                    self._model_path,
                    self._predict_config['device']
                )
                logger.info("模型加载完成")
            
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            return False
    
    def detect(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> Optional[Any]:
        """
        执行关键点检测
        
        Args:
            image: 输入图像（numpy数组，BGR格式或RGB格式）
            conf: 置信度阈值，如果为None则使用配置文件的默认值
            iou: IoU阈值，如果为None则使用配置文件的默认值
            imgsz: 图像尺寸，如果为None则使用配置文件的默认值
            
        Returns:
            YOLO预测结果对象，如果失败则返回None
        """
        try:
            # 确保模型已加载
            if self._model is None:
                if not self.load_model():
                    return None
            
            # 使用提供的参数或配置文件的默认值
            conf = conf if conf is not None else self._predict_config['conf']
            iou = iou if iou is not None else self._predict_config['iou']
            imgsz = imgsz if imgsz is not None else self._predict_config['imgsz']
            
            # 执行预测
            results = self._model.predict(
                source=image,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self._predict_config['device'],
                half=self._predict_config['half'],
                max_det=self._predict_config['max_det'],
                verbose=False
            )
            
            if not results or len(results) == 0:
                logger.warning("未检测到任何对象")
                return None
            
            return results[0]
            
        except Exception as e:
            logger.error(f"检测失败: {e}", exc_info=True)
            return None
    
    def calculate_angles(self, result: Any) -> List[Dict[str, Any]]:
        """
        计算角度
        
        Args:
            result: YOLO预测结果对象
            
        Returns:
            角度计算结果列表
        """
        try:
            if result is None or result.boxes is None or result.keypoints is None:
                return []
            
            angles_results = []
            keypoints_data = result.keypoints.data.cpu().numpy()  # [N, num_keypoints, 3]
            boxes_cls = result.boxes.cls.cpu().numpy().astype(int)  # [N]
            
            for obj_idx in range(len(result.boxes)):
                cls_id = boxes_cls[obj_idx]
                class_name = self._class_names.get(cls_id, f"class_{cls_id}")
                
                # 获取该对象的关键点数据
                if obj_idx < len(keypoints_data):
                    kpt_data = keypoints_data[obj_idx]  # [num_keypoints, 3]
                    
                    # 获取该类别的关键点名称列表
                    kpt_names = self._kpt_names_dict.get(cls_id, [])
                    
                    if kpt_names:
                        # 计算角度
                        angles_result = self.calculate_angles_for_object(
                            class_name=class_name,
                            kpt_data=kpt_data,
                            kpt_names=kpt_names,
                            angle_config=self._angles_config
                        )
                        
                        # 保存结果
                        obj_angles_info = {
                            'object_id': obj_idx,
                            'class_id': int(cls_id),
                            'class_name': class_name,
                            'angles': angles_result
                        }
                        angles_results.append(obj_angles_info)
            
            return angles_results
            
        except Exception as e:
            logger.error(f"角度计算失败: {e}", exc_info=True)
            return []
    
    def annotate_image(
        self,
        image: np.ndarray,
        result: Any,
        angles_results: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        在图像上标注检测结果和角度
        
        Args:
            image: 原始图像（BGR格式）
            result: YOLO预测结果对象
            angles_results: 角度计算结果列表
            
        Returns:
            标注后的图像（BGR格式）
        """
        try:
            # 获取YOLO默认标注后的图像
            annotated_image = result.plot()
            
            if annotated_image is None:
                logger.warning("无法生成标注图像，返回原始图像")
                return image.copy()
            
            # 在图像上标注角度
            if angles_results:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
                
                for obj_angles_info in angles_results:
                    obj_idx = obj_angles_info['object_id']
                    
                    if obj_idx >= len(keypoints_data):
                        continue
                    
                    cls_id = obj_angles_info['class_id']
                    angles_result = obj_angles_info['angles']
                    kpt_names = self._kpt_names_dict.get(cls_id, [])
                    
                    if kpt_names:
                        # 创建关键点名称到索引的映射
                        kpt_indices_map = {name: idx for idx, name in enumerate(kpt_names)}
                        
                        # 获取关键点坐标（xy格式，只有坐标）
                        kpt_xy = result.keypoints.xy[obj_idx].cpu().numpy()  # [num_keypoints, 2]
                        
                        # 标注角度
                        annotated_image = self.annotate_angles_on_image(
                            image=annotated_image,
                            angles_result=angles_result,
                            keypoints_xy=kpt_xy,
                            keypoint_indices_map=kpt_indices_map
                        )
            
            return annotated_image
            
        except Exception as e:
            logger.error(f"图像标注失败: {e}", exc_info=True)
            return image.copy()
    
    def detect_and_annotate(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行检测、角度计算和图像标注（一站式接口）
        
        Args:
            image: 输入图像（numpy数组，BGR格式或RGB格式）
            conf: 置信度阈值，如果为None则使用配置文件的默认值
            iou: IoU阈值，如果为None则使用配置文件的默认值
            imgsz: 图像尺寸，如果为None则使用配置文件的默认值
            
        Returns:
            包含检测结果的字典：
            {
                'success': bool,              # 是否成功
                'annotated_image': np.ndarray, # 标注后的图像（BGR格式）
                'result': Any,                 # YOLO预测结果对象
                'angles_results': List,        # 角度计算结果
                'error_message': str           # 错误信息（如果有）
            }
        """
        try:
            # 执行检测
            result = self.detect(image, conf=conf, iou=iou, imgsz=imgsz)
            
            if result is None:
                return {
                    'success': False,
                    'annotated_image': None,
                    'result': None,
                    'angles_results': [],
                    'error_message': '检测失败：未检测到任何对象'
                }
            
            # 计算角度
            angles_results = self.calculate_angles(result)
            
            # 标注图像（转换为numpy数组如果输入是PIL Image）
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    # PIL Image是RGB格式，需要转换为BGR
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_np = image.copy()
            
            annotated_image = self.annotate_image(image_np, result, angles_results)
            
            return {
                'success': True,
                'annotated_image': annotated_image,
                'result': result,
                'angles_results': angles_results,
                'error_message': None
            }
            
        except Exception as e:
            error_msg = f"检测和标注失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'annotated_image': None,
                'result': None,
                'angles_results': [],
                'error_message': error_msg
            }
    
    def prepare_json_data(
        self,
        result: Any,
        angles_results: List[Dict[str, Any]],
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        准备JSON数据（用于保存）
        
        Args:
            result: YOLO预测结果对象
            angles_results: 角度计算结果列表
            image_path: 图片路径（可选）
            
        Returns:
            JSON数据字典
        """
        try:
            from datetime import datetime
            from .tool_pose_json_utils import convert_to_json_serializable
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            output_data = {
                'image_path': image_path or 'camera_frame',
                'image_shape': result.orig_shape if result else None,
                'timestamp': timestamp,
                'predictions': []
            }
            
            if result and result.boxes is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                boxes_conf = result.boxes.conf.cpu().numpy()
                boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
                keypoints_data = result.keypoints.data.cpu().numpy() if result.keypoints else None
                
                for obj_idx in range(len(result.boxes)):
                    cls_id = boxes_cls[obj_idx]
                    class_name = self._class_names.get(cls_id, f"class_{cls_id}")
                    
                    # 找到对应的角度结果
                    angles_result = {}
                    for obj_angles_info in angles_results:
                        if obj_angles_info['object_id'] == obj_idx:
                            angles_result = obj_angles_info['angles']
                            break
                    
                    # 构建对象信息
                    obj_info = {
                        'object_id': obj_idx,
                        'class_id': int(cls_id),
                        'class_name': class_name,
                        'confidence': float(boxes_conf[obj_idx]),
                        'bbox': {
                            'x1': float(boxes_xyxy[obj_idx][0]),
                            'y1': float(boxes_xyxy[obj_idx][1]),
                            'x2': float(boxes_xyxy[obj_idx][2]),
                            'y2': float(boxes_xyxy[obj_idx][3])
                        },
                        'angles': angles_result
                    }
                    
                    # 添加关键点信息
                    if keypoints_data is not None and obj_idx < len(keypoints_data):
                        kpt_data = keypoints_data[obj_idx]
                        kpt_names = self._kpt_names_dict.get(cls_id, [])
                        keypoints = []
                        
                        for kpt_idx, (x, y, vis) in enumerate(kpt_data):
                            kpt_name = kpt_names[kpt_idx] if kpt_idx < len(kpt_names) else f"kpt_{kpt_idx}"
                            keypoints.append({
                                'index': int(kpt_idx),
                                'name': kpt_name,
                                'x': float(x),
                                'y': float(y),
                                'visibility': float(vis),
                                'visible': bool(vis > 0.5)
                            })
                        
                        obj_info['keypoints'] = keypoints
                    
                    output_data['predictions'].append(obj_info)
            
            # 转换为JSON可序列化格式
            return convert_to_json_serializable(output_data)
            
        except Exception as e:
            logger.error(f"准备JSON数据失败: {e}", exc_info=True)
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取检测器状态信息
        
        Returns:
            状态信息字典
        """
        return {
            'model_loaded': self._model is not None,
            'model_path': self._model_path,
            'config_path': str(self.config_path),
            'class_names': list(self._class_names.keys()) if self._class_names else [],
            'predict_config': self._predict_config
        }

