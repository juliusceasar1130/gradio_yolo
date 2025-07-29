# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:40；主要修改内容：创建批量处理模块

"""
批量处理器模块

实现文件夹批量处理功能，支持结果导出和进度显示
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
import logging
from datetime import datetime

from .detector import BaseDetector, DetectionResult
from .image_processor import ImageProcessor
from .result_processor import ResultProcessor
from ..utils.file_utils import get_image_files, ensure_dir
from ..utils.image_utils import validate_image_format

logger = logging.getLogger(__name__)


class BatchProcessor:
    """批量处理器类"""
    
    def __init__(self, detector: BaseDetector, config=None):
        """
        初始化批量处理器
        
        Args:
            detector: 检测器对象
            config: 配置对象
        """
        self.detector = detector
        self.config = config
        self.image_processor = ImageProcessor(config)
        self.result_processor = ResultProcessor(config)
        
        # 从配置获取批量处理参数
        self.max_workers = 4
        self.chunk_size = 10
        self.export_format = 'csv'
        
        if config:
            batch_config = config.get_batch_config()
            self.max_workers = batch_config.get('max_workers', 4)
            self.chunk_size = batch_config.get('chunk_size', 10)
            self.export_format = batch_config.get('export_format', 'csv')
    
    def process_folder(self, input_folder: str, 
                      output_folder: Optional[str] = None,
                      recursive: bool = False,
                      progress_callback: Optional[Callable[[int, int, str], None]] = None,
                      **detection_kwargs) -> Dict[str, Any]:
        """
        处理文件夹中的所有图像
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            recursive: 是否递归搜索子文件夹
            progress_callback: 进度回调函数 (current, total, message)
            **detection_kwargs: 检测参数
            
        Returns:
            批量处理结果
        """
        try:
            start_time = time.time()
            
            # 验证输入文件夹
            if not os.path.exists(input_folder):
                raise ValueError(f"输入文件夹不存在: {input_folder}")
            
            # 获取图像文件列表
            logger.info(f"扫描图像文件: {input_folder}")
            if progress_callback:
                progress_callback(0, 0, "扫描图像文件...")
            
            image_files = get_image_files(input_folder, recursive=recursive)
            if not image_files:
                logger.warning(f"未找到图像文件: {input_folder}")
                return {
                    'success': False,
                    'error': '未找到图像文件',
                    'total_images': 0
                }
            
            logger.info(f"找到 {len(image_files)} 个图像文件")
            
            # 设置输出文件夹
            if output_folder is None:
                if self.config:
                    output_folder = self.config.get('data.output_folder', './outputs')
                else:
                    output_folder = './outputs'
            
            ensure_dir(output_folder)
            
            # 批量处理图像
            results = self._process_images_batch(
                image_files, 
                progress_callback=progress_callback,
                **detection_kwargs
            )
            
            # 处理结果
            batch_result = self.result_processor.process_batch_results(
                results, 
                image_paths=image_files
            )
            
            # 导出结果
            if progress_callback:
                progress_callback(len(image_files), len(image_files), "导出结果...")
            
            export_success = self._export_results(batch_result, output_folder)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 汇总结果
            final_result = {
                'success': True,
                'input_folder': input_folder,
                'output_folder': output_folder,
                'total_images': len(image_files),
                'successful_detections': batch_result['successful_detections'],
                'failed_detections': batch_result['failed_detections'],
                'total_objects': batch_result['total_objects'],
                'class_summary': batch_result['class_summary'],
                'processing_time': processing_time,
                'export_success': export_success,
                'batch_result': batch_result
            }
            
            logger.info(f"批量处理完成: {final_result['successful_detections']}/{final_result['total_images']} 成功，耗时 {processing_time:.2f}秒")
            return final_result
            
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_folder': input_folder
            }
    
    def _process_images_batch(self, image_files: List[str], 
                             progress_callback: Optional[Callable[[int, int, str], None]] = None,
                             **detection_kwargs) -> List[DetectionResult]:
        """批量处理图像文件"""
        results = []
        processed_count = 0
        total_count = len(image_files)
        
        try:
            # 单线程处理（避免模型并发问题）
            for i, image_file in enumerate(image_files):
                try:
                    if progress_callback:
                        progress_callback(i, total_count, f"处理: {os.path.basename(image_file)}")
                    
                    # 验证图像
                    validation = self.image_processor.validate_image_input(image_file)
                    if not validation['valid']:
                        logger.warning(f"跳过无效图像: {image_file}, 原因: {validation['error']}")
                        results.append(None)
                        continue
                    
                    # 执行检测
                    result = self.detector.detect(image_file, **detection_kwargs)
                    results.append(result)
                    
                    if result and result.get_statistics()['has_detections']:
                        processed_count += 1
                    
                    # 记录进度
                    if (i + 1) % 10 == 0:
                        logger.info(f"已处理 {i + 1}/{total_count} 张图像")
                    
                except Exception as e:
                    logger.error(f"处理图像失败: {image_file}, 错误: {e}")
                    results.append(None)
            
            logger.info(f"批量检测完成: {processed_count}/{total_count} 成功")
            return [r for r in results if r is not None]
            
        except Exception as e:
            logger.error(f"批量处理异常: {e}")
            return results
    
    def _export_results(self, batch_result: Dict[str, Any], output_folder: str) -> bool:
        """导出处理结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 导出CSV
            if self.export_format in ['csv', 'both']:
                csv_path = Path(output_folder) / f"detection_results_{timestamp}.csv"
                csv_success = self.result_processor.export_results_to_csv(batch_result, csv_path)
                if not csv_success:
                    logger.warning("CSV导出失败")
            
            # 导出JSON
            if self.export_format in ['json', 'both']:
                json_path = Path(output_folder) / f"detection_results_{timestamp}.json"
                json_success = self.result_processor.export_results_to_json(batch_result, json_path)
                if not json_success:
                    logger.warning("JSON导出失败")
            
            # 生成汇总报告
            report_path = Path(output_folder) / f"detection_summary_{timestamp}.md"
            summary_report = self.result_processor.generate_summary_report(batch_result)
            
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(summary_report)
                logger.info(f"汇总报告已保存: {report_path}")
            except Exception as e:
                logger.warning(f"保存汇总报告失败: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"导出结果失败: {e}")
            return False
    
    def process_image_list(self, image_paths: List[str],
                          output_folder: Optional[str] = None,
                          progress_callback: Optional[Callable[[int, int, str], None]] = None,
                          **detection_kwargs) -> Dict[str, Any]:
        """
        处理指定的图像列表
        
        Args:
            image_paths: 图像路径列表
            output_folder: 输出文件夹
            progress_callback: 进度回调函数
            **detection_kwargs: 检测参数
            
        Returns:
            处理结果
        """
        try:
            start_time = time.time()
            
            # 验证图像路径
            valid_paths = []
            for path in image_paths:
                if os.path.exists(path) and validate_image_format(path):
                    valid_paths.append(path)
                else:
                    logger.warning(f"跳过无效图像路径: {path}")
            
            if not valid_paths:
                return {
                    'success': False,
                    'error': '没有有效的图像路径',
                    'total_images': 0
                }
            
            # 设置输出文件夹
            if output_folder is None:
                if self.config:
                    output_folder = self.config.get('data.output_folder', './outputs')
                else:
                    output_folder = './outputs'
            
            ensure_dir(output_folder)
            
            # 批量处理
            results = self._process_images_batch(
                valid_paths,
                progress_callback=progress_callback,
                **detection_kwargs
            )
            
            # 处理结果
            batch_result = self.result_processor.process_batch_results(
                results,
                image_paths=valid_paths
            )
            
            # 导出结果
            export_success = self._export_results(batch_result, output_folder)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'total_images': len(valid_paths),
                'successful_detections': batch_result['successful_detections'],
                'failed_detections': batch_result['failed_detections'],
                'total_objects': batch_result['total_objects'],
                'class_summary': batch_result['class_summary'],
                'processing_time': processing_time,
                'export_success': export_success,
                'output_folder': output_folder,
                'batch_result': batch_result
            }
            
        except Exception as e:
            logger.error(f"处理图像列表失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_images': len(image_paths) if image_paths else 0
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'export_format': self.export_format,
            'detector_type': self.detector.detector_type,
            'model_loaded': self.detector.is_model_loaded(),
            'result_stats': self.result_processor.get_statistics_summary()
        }
    
    def estimate_processing_time(self, image_count: int, 
                               avg_time_per_image: float = 0.5) -> Dict[str, float]:
        """
        估算处理时间
        
        Args:
            image_count: 图像数量
            avg_time_per_image: 每张图像平均处理时间（秒）
            
        Returns:
            时间估算结果
        """
        estimated_seconds = image_count * avg_time_per_image
        
        return {
            'total_images': image_count,
            'estimated_seconds': estimated_seconds,
            'estimated_minutes': estimated_seconds / 60,
            'estimated_hours': estimated_seconds / 3600,
            'avg_time_per_image': avg_time_per_image
        }
