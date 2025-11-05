#!/usr/bin/env python3
"""
Axis 摄像头连接测试脚本
使用 root/root 凭据测试摄像头连接
"""

import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.yolo_detector.core.camera_capture import CameraCapture


def test_axis_camera():
    """测试 Axis 摄像头连接"""

    # 摄像头配置
    CAMERA_IP = "192.22.39.253"
    USERNAME = "root"
    PASSWORD = "root"

    print("=" * 60)
    print("Axis 摄像头连接测试")
    print("=" * 60)
    print(f"IP 地址: {CAMERA_IP}")
    print(f"用户名: {USERNAME}")
    print(f"密码: {'*' * len(PASSWORD)}")
    print()

    # 创建摄像头对象
    camera = CameraCapture(
        camera_type="axis",
        axis_ip=CAMERA_IP,
        axis_username=USERNAME,
        axis_password=PASSWORD
    )

    # 步骤 1: 基础连接测试
    print("\n【步骤 1: 基础连接测试】")
    print("-" * 60)
    test_results = camera.test_connection()

    # 显示测试结果
    print(f"总体状态: {test_results['overall_status']}")
    print(f"总体消息: {test_results['overall_message']}")
    print()

    for test in test_results['tests']:
        status_symbol = {
            "SUCCESS": "✅",
            "FAILED": "❌",
            "WARNING": "⚠️",
            "ERROR": "💥",
            "INFO": "ℹ️"
        }.get(test['status'], "❓")

        print(f"{status_symbol} {test['name']}: {test['status']}")
        print(f"   {test['message']}")

        if 'details' in test:
            for key, value in test['details'].items():
                print(f"   {key}: {value}")
        print()

    # 步骤 2: 如果基础测试通过，尝试实际连接
    if test_results['overall_status'] in ["SUCCESS", "WARNING"]:
        print("\n【步骤 2: 实际连接测试】")
        print("-" * 60)
        print("正在尝试连接摄像头...")
        print()

        success, message = camera.connect()

        if success:
            print(f"✅ 连接成功！")
            print(f"   {message}")
            print()

            # 获取状态
            status = camera.get_status()
            print("摄像头状态:")
            print(f"   连接状态: {status['is_connected']}")
            print(f"   采集状态: {status['is_capturing']}")
            print(f"   连接方式: {status.get('connection_method', 'N/A')}")
            print(f"   已采集帧数: {status['frame_count']}")
            print()

            # 测试获取帧
            print("正在测试获取帧...")
            frame = camera.get_frame()
            if frame is not None:
                print(f"✅ 成功获取帧，图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("⚠️ 无法获取帧（可能还在初始化中）")
            print()

            # 断开连接
            print("正在断开连接...")
            camera.disconnect()
            print("✅ 连接已断开")

        else:
            print(f"❌ 连接失败")
            print(f"   {message}")
            print()

            # 错误分析
            print("可能的原因:")
            print("1. 摄像头未启动或重启")
            print("2. 用户名/密码不正确（尽管您提供了 root/root）")
            print("3. 网络问题（IP 地址冲突、网线问题）")
            print("4. 防火墙阻止连接")
            print("5. 固件版本不支持 MJPEG 流")
            print()

            print("建议操作:")
            print("1. 手动访问 http://192.168.39.253 验证摄像头在线")
            print("2. 检查网络: ping 192.168.39.253")
            print("3. 确认用户名密码（尝试 admin/pass）")
            print("4. 查看摄像头系统日志")
    else:
        print("\n❌ 基础连接测试失败，请检查网络和摄像头配置")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_axis_camera()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
