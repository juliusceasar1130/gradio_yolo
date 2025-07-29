# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:40；主要修改内容：创建统一的结果处理和统计模块

"""
结果处理器模块

提供统一的检测结果处理、统计和展示功能
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from .detector import DetectionResult
from ..utils.file_utils import ensure_dir, get_unique_filename

logger = logging.getLogger(__name__)


class ResultProcessor:
    """结果处理器类"""
    
    def __init__(self, config=None):
        """
        初始化结果处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.results_history = []
    
    def process_single_result(self, result: DetectionResult, 
                            image_path: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理单个检测结果
        
        Args:
            result: 检测结果对象
            image_path: 图像路径
            metadata: 额外的元数据
            
        Returns:
            处理后的结果字典
        """
        try:
            processed_result = {
                'timestamp': datetime.now().isoformat(),
                'detector_type': result.detector_type,
                'image_path': image_path,
                'statistics': result.get_statistics(),
                'formatted_stats': result.format_statistics(),
                'has_detections': result.get_statistics()['has_detections'],
                'metadata': metadata or {}
            }
            
            # 添加详细的检测信息
            if result.boxes is not None and len(result.boxes) > 0:
                processed_result['detections'] = self._extract_detection_details(result)
            
            # 添加到历史记录
            self.results_history.append(processed_result)
            
            logger.debug(f"处理单个结果完成: {processed_result['statistics']['total_detections']} 个检测")
            return processed_result
            
        except Exception as e:
            logger.error(f"处理单个结果失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'has_detections': False
            }
    
    def _extract_detection_details(self, result: DetectionResult) -> List[Dict[str, Any]]:
        """提取详细的检测信息"""
        detections = []
        
        try:
            if result.boxes is None:
                return detections
            
            for i, box in enumerate(result.boxes):
                detection = {
                    'id': i,
                    'class_id': int(box.cls.item()),
                    'class_name': result.names.get(int(box.cls.item()), f'class_{int(box.cls.item())}'),
                    'confidence': float(box.conf.item()),
                    'bbox': box.xyxy.tolist()[0] if hasattr(box, 'xyxy') else None
                }
                
                # 添加分割掩码信息（如果有）
                if result.masks is not None and i < len(result.masks):
                    mask = result.masks[i]
                    detection['mask_area'] = float(np.sum(mask.data.cpu().numpy() > 0.5))
                
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"提取检测详情失败: {e}")
            return []
    
    def process_batch_results(self, results: List[DetectionResult], 
                            image_paths: Optional[List[str]] = None,
                            metadata_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        处理批量检测结果
        
        Args:
            results: 检测结果列表
            image_paths: 图像路径列表
            metadata_list: 元数据列表
            
        Returns:
            批量处理结果
        """
        try:
            batch_result = {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(results),
                'successful_detections': 0,
                'failed_detections': 0,
                'total_objects': 0,
                'class_summary': {},
                'individual_results': []
            }
            
            for i, result in enumerate(results):
                image_path = image_paths[i] if image_paths and i < len(image_paths) else None
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
                
                processed = self.process_single_result(result, image_path, metadata)
                batch_result['individual_results'].append(processed)
                
                if processed.get('has_detections', False):
                    batch_result['successful_detections'] += 1
                    stats = processed['statistics']
                    batch_result['total_objects'] += stats['total_detections']
                    
                    # 汇总类别统计
                    for class_name, class_info in stats['classes'].items():
                        if class_name in batch_result['class_summary']:
                            batch_result['class_summary'][class_name]['count'] += class_info['count']
                            batch_result['class_summary'][class_name]['images'] += 1
                        else:
                            batch_result['class_summary'][class_name] = {
                                'count': class_info['count'],
                                'images': 1
                            }
                else:
                    batch_result['failed_detections'] += 1
            
            logger.info(f"批量处理完成: {batch_result['successful_detections']}/{batch_result['total_images']} 成功")
            return batch_result
            
        except Exception as e:
            logger.error(f"批量处理结果失败: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'total_images': len(results) if results else 0
            }
    
    def export_results_to_csv(self, results: Union[Dict[str, Any], List[Dict[str, Any]]], 
                             output_path: str) -> bool:
        """
        导出结果到CSV文件
        
        Args:
            results: 结果数据
            output_path: 输出路径
            
        Returns:
            是否导出成功
        """
        try:
            # 确保输出目录存在
            output_path = Path(output_path)
            ensure_dir(output_path.parent)
            
            # 处理不同类型的输入
            if isinstance(results, dict):
                if 'individual_results' in results:
                    # 批量结果
                    data_list = results['individual_results']
                else:
                    # 单个结果
                    data_list = [results]
            else:
                # 结果列表
                data_list = results
            
            # 准备CSV数据
            csv_data = []
            for result in data_list:
                if 'detections' in result:
                    # 每个检测对象一行
                    for detection in result['detections']:
                        row = {
                            'timestamp': result.get('timestamp', ''),
                            'image_path': result.get('image_path', ''),
                            'detector_type': result.get('detector_type', ''),
                            'class_name': detection['class_name'],
                            'class_id': detection['class_id'],
                            'confidence': detection['confidence'],
                            'bbox_x1': detection['bbox'][0] if detection['bbox'] else None,
                            'bbox_y1': detection['bbox'][1] if detection['bbox'] else None,
                            'bbox_x2': detection['bbox'][2] if detection['bbox'] else None,
                            'bbox_y2': detection['bbox'][3] if detection['bbox'] else None,
                            'mask_area': detection.get('mask_area', None)
                        }
                        csv_data.append(row)
                else:
                    # 没有检测结果的图像
                    row = {
                        'timestamp': result.get('timestamp', ''),
                        'image_path': result.get('image_path', ''),
                        'detector_type': result.get('detector_type', ''),
                        'class_name': None,
                        'class_id': None,
                        'confidence': None,
                        'bbox_x1': None,
                        'bbox_y1': None,
                        'bbox_x2': None,
                        'bbox_y2': None,
                        'mask_area': None
                    }
                    csv_data.append(row)
            
            # 创建DataFrame并保存
            df = pd.DataFrame(csv_data)
            
            # 确保文件名唯一
            unique_path = get_unique_filename(output_path)
            df.to_csv(unique_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"结果导出到CSV成功: {unique_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")
            return False
    
    def export_results_to_json(self, results: Union[Dict[str, Any], List[Dict[str, Any]]], 
                              output_path: str) -> bool:
        """
        导出结果到JSON文件
        
        Args:
            results: 结果数据
            output_path: 输出路径
            
        Returns:
            是否导出成功
        """
        try:
            # 确保输出目录存在
            output_path = Path(output_path)
            ensure_dir(output_path.parent)
            
            # 确保文件名唯一
            unique_path = get_unique_filename(output_path)
            
            # 保存JSON
            with open(unique_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"结果导出到JSON成功: {unique_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出JSON失败: {e}")
            return False
    
    def generate_summary_report(self, results: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        """
        生成汇总报告
        
        Args:
            results: 结果数据
            
        Returns:
            汇总报告文本
        """
        try:
            if isinstance(results, dict) and 'individual_results' in results:
                # 批量结果
                batch_result = results
                report_parts = [
                    "# 检测结果汇总报告",
                    f"**处理时间**: {batch_result.get('timestamp', 'N/A')}",
                    f"**总图像数**: {batch_result.get('total_images', 0)}",
                    f"**成功检测**: {batch_result.get('successful_detections', 0)}",
                    f"**失败检测**: {batch_result.get('failed_detections', 0)}",
                    f"**检测对象总数**: {batch_result.get('total_objects', 0)}",
                    "",
                    "## 类别统计"
                ]
                
                class_summary = batch_result.get('class_summary', {})
                if class_summary:
                    for class_name, info in class_summary.items():
                        report_parts.append(f"- **{class_name}**: {info['count']}个 (出现在{info['images']}张图像中)")
                else:
                    report_parts.append("未检测到任何对象")
                
            else:
                # 单个结果或结果列表
                if isinstance(results, list):
                    total_images = len(results)
                    successful = sum(1 for r in results if r.get('has_detections', False))
                    total_objects = sum(r.get('statistics', {}).get('total_detections', 0) for r in results)
                else:
                    total_images = 1
                    successful = 1 if results.get('has_detections', False) else 0
                    total_objects = results.get('statistics', {}).get('total_detections', 0)
                
                report_parts = [
                    "# 检测结果汇总报告",
                    f"**总图像数**: {total_images}",
                    f"**成功检测**: {successful}",
                    f"**检测对象总数**: {total_objects}"
                ]
            
            return "\n".join(report_parts)
            
        except Exception as e:
            logger.error(f"生成汇总报告失败: {e}")
            return f"生成报告失败: {e}"
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """获取统计汇总"""
        try:
            if not self.results_history:
                return {'total_processed': 0}
            
            total_processed = len(self.results_history)
            successful = sum(1 for r in self.results_history if r.get('has_detections', False))
            total_objects = sum(r.get('statistics', {}).get('total_detections', 0) for r in self.results_history)
            
            # 统计检测器类型
            detector_types = {}
            for result in self.results_history:
                detector_type = result.get('detector_type', 'unknown')
                detector_types[detector_type] = detector_types.get(detector_type, 0) + 1
            
            return {
                'total_processed': total_processed,
                'successful_detections': successful,
                'failed_detections': total_processed - successful,
                'total_objects_detected': total_objects,
                'detector_types': detector_types,
                'success_rate': successful / total_processed if total_processed > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"获取统计汇总失败: {e}")
            return {'error': str(e)}
    
    def clear_history(self):
        """清除历史记录"""
        self.results_history.clear()
        logger.info("结果历史记录已清除")
