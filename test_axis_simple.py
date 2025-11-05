#!/usr/bin/env python3
"""
独立的 Axis 摄像头测试脚本
不依赖项目模块，直接测试摄像头连接
"""

import socket
import requests
from requests.auth import HTTPDigestAuth
import sys
import json
from datetime import datetime

# 摄像头配置
CAMERA_IP = "192.168.39.253"
USERNAME = "root"
PASSWORD = "root"


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_test_result(name, status, message, details=None):
    """打印测试结果"""
    status_symbol = {
        "SUCCESS": "✅",
        "FAILED": "❌",
        "WARNING": "⚠️",
        "ERROR": "💥",
        "INFO": "ℹ️"
    }.get(status, "❓")

    print(f"\n{status_symbol} 测试项目: {name}")
    print(f"   状态: {status}")
    print(f"   消息: {message}")

    if details:
        print(f"   详情:")
        for key, value in details.items():
            print(f"      {key}: {value}")


def test_network_connectivity():
    """测试网络连通性"""
    print_header("测试 1: 网络连通性")

    try:
        # 测试 socket 连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((CAMERA_IP, 80))
        sock.close()

        if result == 0:
            print_test_result(
                "Socket 连接 (端口 80)",
                "SUCCESS",
                f"可以连接到 {CAMERA_IP}:80",
                {"IP": CAMERA_IP, "端口": 80}
            )
            return True
        else:
            print_test_result(
                "Socket 连接 (端口 80)",
                "FAILED",
                f"无法连接到 {CAMERA_IP}:80",
                {"错误代码": result}
            )
            return False
    except Exception as e:
        print_test_result(
            "Socket 连接 (端口 80)",
            "ERROR",
            f"测试时发生异常: {str(e)}",
            {"异常类型": type(e).__name__}
        )
        return False


def test_http_access():
    """测试 HTTP 访问"""
    print_header("测试 2: HTTP 访问")

    try:
        response = requests.get(
            f"http://{CAMERA_IP}",
            timeout=5,
            headers={'User-Agent': 'Axis Camera Test Client'}
        )

        print_test_result(
            "HTTP 主页访问",
            "SUCCESS",
            f"HTTP 响应码: {response.status_code}",
            {
                "URL": f"http://{CAMERA_IP}/",
                "状态码": response.status_code,
                "Server": response.headers.get('Server', 'Unknown'),
                "Content-Type": response.headers.get('Content-Type', 'Unknown')
            }
        )
        return True, response
    except requests.exceptions.ConnectTimeout:
        print_test_result(
            "HTTP 主页访问",
            "FAILED",
            "连接超时",
            {"URL": f"http://{CAMERA_IP}/"}
        )
        return False, None
    except requests.exceptions.ConnectionError as e:
        print_test_result(
            "HTTP 主页访问",
            "FAILED",
            f"连接错误: {str(e)}",
            {"错误类型": "ConnectionError"}
        )
        return False, None
    except Exception as e:
        print_test_result(
            "HTTP 主页访问",
            "ERROR",
            f"发生异常: {str(e)}",
            {"异常类型": type(e).__name__}
        )
        return False, None


def test_mjpeg_endpoints():
    """测试 MJPEG 端点"""
    print_header("测试 3: MJPEG 端点")

    # 定义要测试的 URL
    mjpeg_urls = [
        # 带参数的 MJPEG 流（推荐）
        {
            "url": f"http://{CAMERA_IP}/axis-cgi/mjpg/video.cgi?resolution=640x480&fps=25",
            "name": "MJPEG 流 (带参数)"
        },
        # 标准 MJPEG
        {
            "url": f"http://{CAMERA_IP}/axis-cgi/mjpg/video.cgi",
            "name": "MJPEG 流 (标准)"
        },
        # 无认证的 URL
        {
            "url": f"http://{CAMERA_IP}/mjpg/video.mjpg",
            "name": "MJPEG 文件 (替代路径)"
        },
        {
            "url": f"http://{CAMERA_IP}/mjpg/video.cgi",
            "name": "MJPEG CGI (替代路径)"
        }
    ]

    successful_urls = []

    for i, item in enumerate(mjpeg_urls, 1):
        print(f"\n--- 测试 URL {i}/{len(mjpeg_urls)} ---")
        print(f"URL: {item['url'].replace(f'http://{CAMERA_IP}', 'http://***.***.***.***')}")

        try:
            # 尝试无认证访问
            response = requests.get(
                item["url"],
                timeout=10,
                headers={'User-Agent': 'Axis MJPEG Test', 'Accept': 'multipart/x-mixed-replace'},
                stream=True
            )

            print(f"响应状态码: {response.status_code}")

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                print(f"Content-Type: {content_type}")

                # 检查是否是 MJPEG 流
                if 'jpeg' in content_type or 'multipart' in content_type:
                    print_test_result(
                        item['name'],
                        "SUCCESS",
                        f"MJPEG 流可用",
                        {
                            "Content-Type": response.headers.get('Content-Type'),
                            "Content-Length": response.headers.get('Content-Length', 'Unknown'),
                            "Accept-Ranges": response.headers.get('Accept-Ranges', 'Unknown')
                        }
                    )
                    successful_urls.append(item["url"])
                    response.close()
                    break  # 成功了就不测试下一个
                else:
                    print_test_result(
                        item['name'],
                        "WARNING",
                        f"端点存在但 Content-Type 不正确",
                        {"Content-Type": content_type}
                    )
            else:
                print_test_result(
                    item['name'],
                    "FAILED",
                    f"HTTP 错误 {response.status_code}",
                    {"状态码": response.status_code}
                )

            response.close()

        except requests.exceptions.RequestException as e:
            print_test_result(
                item['name'],
                "FAILED",
                f"请求失败: {str(e)}",
                {"错误类型": type(e).__name__}
            )
        except Exception as e:
            print_test_result(
                item['name'],
                "ERROR",
                f"发生异常: {str(e)}",
                {"异常类型": type(e).__name__}
            )

    return successful_urls


def test_authentication():
    """测试认证"""
    print_header("测试 4: 认证测试")

    auth_urls = [
        f"http://{CAMERA_IP}/axis-cgi/mjpg/video.cgi",
        f"http://{CAMERA_IP}/axis-cgi/com/ptz.cgi"
    ]

    auth_methods = [
        ("Digest Auth (root/root)", lambda url: requests.get(url, timeout=5, auth=HTTPDigestAuth(USERNAME, PASSWORD))),
        ("Basic Auth (root/root)", lambda url: requests.get(url, timeout=5, auth=(USERNAME, PASSWORD)))
    ]

    for url in auth_urls:
        print(f"\n--- 测试 URL: {url.replace(f'http://{CAMERA_IP}', 'http://***.***.***.***')} ---")

        safe_url = url.replace(f"http://{CAMERA_IP}", "http://***.***.***.***")

        for auth_name, auth_func in auth_methods:
            try:
                print(f"尝试认证方式: {auth_name}")
                response = auth_func(url)
                print(f"  状态码: {response.status_code}")

                if response.status_code == 200:
                    print_test_result(
                        f"认证测试 - {auth_name}",
                        "SUCCESS",
                        "认证成功",
                        {
                            "URL": safe_url,
                            "认证方式": auth_name.split(' ')[0],
                            "状态码": response.status_code
                        }
                    )
                    response.close()
                    return True, auth_name
                elif response.status_code == 401:
                    print(f"  ❌ 401 Unauthorized - 认证失败")
                elif response.status_code == 403:
                    print(f"  ❌ 403 Forbidden - 权限不足")

                response.close()

            except requests.exceptions.RequestException as e:
                print(f"  ❌ 请求失败: {str(e)}")
            except Exception as e:
                print(f"  ❌ 异常: {str(e)}")

    print_test_result(
        "认证测试",
        "FAILED",
        "所有认证方式都失败"
    )
    return False, None


def test_snapshot_endpoint():
    """测试快照端点（从您的 webcamera.py）"""
    print_header("测试 5: 快照端点 (来自 webcamera.py)")

    # 使用您 webcamera.py 中的命令
    snapshot_url = f"http://{CAMERA_IP}/axis-cgi/jpg/image.cgi?resolution=1280x720&compression=25&camera=1"

    print(f"URL: {snapshot_url.replace(f'http://{CAMERA_IP}', 'http://***.***.***.***')}")

    try:
        # 尝试 Digest Auth
        response = requests.get(
            snapshot_url,
            timeout=10,
            auth=HTTPDigestAuth(USERNAME, PASSWORD)
        )

        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            content_length = len(response.content)

            print_test_result(
                "快照端点 (Digest Auth)",
                "SUCCESS",
                f"成功获取快照",
                {
                    "Content-Type": content_type,
                    "图像大小": f"{content_length} 字节",
                    "分辨率": "1280x720",
                    "压缩": "25"
                }
            )

            # 尝试保存到文件
            try:
                save_path = f"D:/test_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"\n✅ 图像已保存到: {save_path}")
            except Exception as e:
                print(f"\n⚠️ 保存图像失败: {e}")

            response.close()
            return True
        else:
            print_test_result(
                "快照端点 (Digest Auth)",
                "FAILED",
                f"HTTP 错误 {response.status_code}",
                {"状态码": response.status_code}
            )
            response.close()

    except Exception as e:
        print_test_result(
            "快照端点 (Digest Auth)",
            "ERROR",
            f"发生异常: {str(e)}",
            {"异常类型": type(e).__name__}
        )

    return False


def main():
    """主测试流程"""
    print_header("Axis 摄像头连接诊断")
    print(f"摄像头 IP: {CAMERA_IP}")
    print(f"用户名: {USERNAME}")
    print(f"密码: {'*' * len(PASSWORD)}")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 执行所有测试
    results = {}

    # 1. 网络连通性
    results['network'] = test_network_connectivity()

    # 2. HTTP 访问
    http_ok, response = test_http_access()
    results['http'] = http_ok

    # 3. MJPEG 端点
    if results['http']:
        mjpeg_urls = test_mjpeg_endpoints()
        results['mjpeg'] = len(mjpeg_urls) > 0
        results['mjpeg_urls'] = mjpeg_urls
    else:
        print("\n⚠️ HTTP 访问失败，跳过 MJPEG 测试")
        results['mjpeg'] = False

    # 4. 认证测试
    if results['http']:
        auth_ok, auth_method = test_authentication()
        results['auth'] = auth_ok
        results['auth_method'] = auth_method
    else:
        print("\n⚠️ HTTP 访问失败，跳过认证测试")
        results['auth'] = False

    # 5. 快照端点
    if results['http']:
        snapshot_ok = test_snapshot_endpoint()
        results['snapshot'] = snapshot_ok
    else:
        print("\n⚠️ HTTP 访问失败，跳过快照测试")
        results['snapshot'] = False

    # 总结
    print_header("测试总结")

    total_tests = 5
    passed_tests = sum([
        1 if results['network'] else 0,
        1 if results['http'] else 0,
        1 if results['mjpeg'] else 0,
        1 if results['auth'] else 0,
        1 if results['snapshot'] else 0
    ])

    print(f"通过测试: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！摄像头连接正常")
    elif results['network'] and results['http']:
        print("\n✅ 基础连接正常，但某些功能可能需要调整")
    else:
        print("\n❌ 存在问题，请检查网络和摄像头配置")

    # 保存结果到文件
    try:
        results_file = f"D:/axis_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n📄 详细测试结果已保存到: {results_file}")
    except Exception as e:
        print(f"\n⚠️ 保存结果文件失败: {e}")

    print_header("诊断完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
