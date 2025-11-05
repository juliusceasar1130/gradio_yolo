#!/usr/bin/env python3
"""
Axis Camera Connection Test Script (English Version)
Tests camera connection with root/root credentials
"""

import socket
import requests
from requests.auth import HTTPDigestAuth
import sys
import json
from datetime import datetime

# Camera configuration
CAMERA_IP = "192.22.39.253"
USERNAME = "root"
PASSWORD = "root"


def print_header(title):
    """Print header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_test_result(name, status, message, details=None):
    """Print test result"""
    status_symbol = {
        "SUCCESS": "[OK]",
        "FAILED": "[FAIL]",
        "WARNING": "[WARN]",
        "ERROR": "[ERROR]",
        "INFO": "[INFO]"
    }.get(status, "?")

    print(f"\n{status_symbol} Test: {name}")
    print(f"   Status: {status}")
    print(f"   Message: {message}")

    if details:
        print(f"   Details:")
        for key, value in details.items():
            print(f"      {key}: {value}")


def test_network_connectivity():
    """Test network connectivity"""
    print_header("Test 1: Network Connectivity")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((CAMERA_IP, 80))
        sock.close()

        if result == 0:
            print_test_result(
                "Socket Connection (Port 80)",
                "SUCCESS",
                f"Can connect to {CAMERA_IP}:80",
                {"IP": CAMERA_IP, "Port": 80}
            )
            return True
        else:
            print_test_result(
                "Socket Connection (Port 80)",
                "FAILED",
                f"Cannot connect to {CAMERA_IP}:80",
                {"Error Code": result}
            )
            return False
    except Exception as e:
        print_test_result(
            "Socket Connection (Port 80)",
            "ERROR",
            f"Exception: {str(e)}",
            {"Exception Type": type(e).__name__}
        )
        return False


def test_http_access():
    """Test HTTP access"""
    print_header("Test 2: HTTP Access")

    try:
        response = requests.get(
            f"http://{CAMERA_IP}",
            timeout=5,
            headers={'User-Agent': 'Axis Camera Test Client'}
        )

        print_test_result(
            "HTTP Main Page",
            "SUCCESS",
            f"HTTP Status Code: {response.status_code}",
            {
                "URL": f"http://{CAMERA_IP}/",
                "Status Code": response.status_code,
                "Server": response.headers.get('Server', 'Unknown'),
                "Content-Type": response.headers.get('Content-Type', 'Unknown')
            }
        )
        return True, response
    except requests.exceptions.ConnectTimeout:
        print_test_result(
            "HTTP Main Page",
            "FAILED",
            "Connection timeout",
            {"URL": f"http://{CAMERA_IP}/"}
        )
        return False, None
    except requests.exceptions.ConnectionError as e:
        print_test_result(
            "HTTP Main Page",
            "FAILED",
            f"Connection error: {str(e)}",
            {"Error Type": "ConnectionError"}
        )
        return False, None
    except Exception as e:
        print_test_result(
            "HTTP Main Page",
            "ERROR",
            f"Exception: {str(e)}",
            {"Exception Type": type(e).__name__}
        )
        return False, None


def test_mjpeg_endpoints():
    """Test MJPEG endpoints"""
    print_header("Test 3: MJPEG Endpoints")

    mjpeg_urls = [
        {
            "url": f"http://{CAMERA_IP}/axis-cgi/mjpg/video.cgi?resolution=640x480&fps=25",
            "name": "MJPEG Stream (With Params)"
        },
        {
            "url": f"http://{CAMERA_IP}/axis-cgi/mjpg/video.cgi",
            "name": "MJPEG Stream (Standard)"
        },
        {
            "url": f"http://{CAMERA_IP}/mjpg/video.mjpg",
            "name": "MJPEG File (Alt Path)"
        },
        {
            "url": f"http://{CAMERA_IP}/mjpg/video.cgi",
            "name": "MJPEG CGI (Alt Path)"
        }
    ]

    successful_urls = []

    for i, item in enumerate(mjpeg_urls, 1):
        print(f"\n--- Testing URL {i}/{len(mjpeg_urls)} ---")
        print(f"URL: {item['url'].replace(f'http://{CAMERA_IP}', 'http://***.***.***.***')}")

        try:
            response = requests.get(
                item["url"],
                timeout=10,
                headers={'User-Agent': 'Axis MJPEG Test', 'Accept': 'multipart/x-mixed-replace'},
                stream=True
            )

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                print(f"Content-Type: {content_type}")

                if 'jpeg' in content_type or 'multipart' in content_type:
                    print_test_result(
                        item['name'],
                        "SUCCESS",
                        f"MJPEG stream available",
                        {
                            "Content-Type": response.headers.get('Content-Type'),
                            "Content-Length": response.headers.get('Content-Length', 'Unknown'),
                            "Accept-Ranges": response.headers.get('Accept-Ranges', 'Unknown')
                        }
                    )
                    successful_urls.append(item["url"])
                    response.close()
                    break
                else:
                    print_test_result(
                        item['name'],
                        "WARNING",
                        f"Endpoint exists but Content-Type is incorrect",
                        {"Content-Type": content_type}
                    )
            else:
                print_test_result(
                    item['name'],
                    "FAILED",
                    f"HTTP error {response.status_code}",
                    {"Status Code": response.status_code}
                )

            response.close()

        except requests.exceptions.RequestException as e:
            print_test_result(
                item['name'],
                "FAILED",
                f"Request failed: {str(e)}",
                {"Error Type": type(e).__name__}
            )
        except Exception as e:
            print_test_result(
                item['name'],
                "ERROR",
                f"Exception: {str(e)}",
                {"Exception Type": type(e).__name__}
            )

    return successful_urls


def test_authentication():
    """Test authentication"""
    print_header("Test 4: Authentication")

    auth_urls = [
        f"http://{CAMERA_IP}/axis-cgi/mjpg/video.cgi",
        f"http://{CAMERA_IP}/axis-cgi/com/ptz.cgi"
    ]

    auth_methods = [
        ("Digest Auth (root/root)", lambda url: requests.get(url, timeout=5, auth=HTTPDigestAuth(USERNAME, PASSWORD))),
        ("Basic Auth (root/root)", lambda url: requests.get(url, timeout=5, auth=(USERNAME, PASSWORD)))
    ]

    for url in auth_urls:
        print(f"\n--- Testing URL: {url.replace(f'http://{CAMERA_IP}', 'http://***.***.***.***')} ---")

        safe_url = url.replace(f"http://{CAMERA_IP}", "http://***.***.***.***")

        for auth_name, auth_func in auth_methods:
            try:
                print(f"Trying: {auth_name}")
                response = auth_func(url)
                print(f"  Status: {response.status_code}")

                if response.status_code == 200:
                    print_test_result(
                        f"Auth Test - {auth_name}",
                        "SUCCESS",
                        "Authentication successful",
                        {
                            "URL": safe_url,
                            "Auth Type": auth_name.split(' ')[0],
                            "Status Code": response.status_code
                        }
                    )
                    response.close()
                    return True, auth_name
                elif response.status_code == 401:
                    print(f"  [FAIL] 401 Unauthorized - Auth failed")
                elif response.status_code == 403:
                    print(f"  [FAIL] 403 Forbidden - Access denied")

                response.close()

            except requests.exceptions.RequestException as e:
                print(f"  [FAIL] Request failed: {str(e)}")
            except Exception as e:
                print(f"  [FAIL] Exception: {str(e)}")

    print_test_result(
        "Authentication Test",
        "FAILED",
        "All authentication methods failed"
    )
    return False, None


def test_snapshot_endpoint():
    """Test snapshot endpoint (from webcamera.py)"""
    print_header("Test 5: Snapshot Endpoint (from webcamera.py)")

    snapshot_url = f"http://{CAMERA_IP}/axis-cgi/jpg/image.cgi?resolution=1280x720&compression=25&camera=1"

    print(f"URL: {snapshot_url.replace(f'http://{CAMERA_IP}', 'http://***.***.***.***')}")

    try:
        response = requests.get(
            snapshot_url,
            timeout=10,
            auth=HTTPDigestAuth(USERNAME, PASSWORD)
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            content_length = len(response.content)

            print_test_result(
                "Snapshot Endpoint (Digest Auth)",
                "SUCCESS",
                f"Snapshot captured successfully",
                {
                    "Content-Type": content_type,
                    "Image Size": f"{content_length} bytes",
                    "Resolution": "1280x720",
                    "Compression": "25"
                }
            )

            try:
                save_path = f"D:/test_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"\n[OK] Image saved to: {save_path}")
            except Exception as e:
                print(f"\n[WARN] Failed to save image: {e}")

            response.close()
            return True
        else:
            print_test_result(
                "Snapshot Endpoint (Digest Auth)",
                "FAILED",
                f"HTTP error {response.status_code}",
                {"Status Code": response.status_code}
            )
            response.close()

    except Exception as e:
        print_test_result(
            "Snapshot Endpoint (Digest Auth)",
            "ERROR",
            f"Exception: {str(e)}",
            {"Exception Type": type(e).__name__}
        )

    return False


def main():
    """Main test flow"""
    print_header("Axis Camera Connection Diagnostic")
    print(f"Camera IP: {CAMERA_IP}")
    print(f"Username: {USERNAME}")
    print(f"Password: {'*' * len(PASSWORD)}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    print_header("Running Tests")
    results['network'] = test_network_connectivity()

    http_ok, response = test_http_access()
    results['http'] = http_ok

    if results['http']:
        mjpeg_urls = test_mjpeg_endpoints()
        results['mjpeg'] = len(mjpeg_urls) > 0
        results['mjpeg_urls'] = mjpeg_urls
    else:
        print("\n[WARN] HTTP access failed, skipping MJPEG test")
        results['mjpeg'] = False

    if results['http']:
        auth_ok, auth_method = test_authentication()
        results['auth'] = auth_ok
        results['auth_method'] = auth_method
    else:
        print("\n[WARN] HTTP access failed, skipping auth test")
        results['auth'] = False

    if results['http']:
        snapshot_ok = test_snapshot_endpoint()
        results['snapshot'] = snapshot_ok
    else:
        print("\n[WARN] HTTP access failed, skipping snapshot test")
        results['snapshot'] = False

    print_header("Test Summary")

    total_tests = 5
    passed_tests = sum([
        1 if results['network'] else 0,
        1 if results['http'] else 0,
        1 if results['mjpeg'] else 0,
        1 if results['auth'] else 0,
        1 if results['snapshot'] else 0
    ])

    print(f"\nPassed Tests: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("\n[SUCCESS] All tests passed! Camera connection is working.")
    elif results['network'] and results['http']:
        print("\n[OK] Basic connection works, but some features may need adjustment.")
    else:
        print("\n[FAIL] Issues detected. Please check network and camera configuration.")

    try:
        results_file = f"D:/axis_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"\n[WARN] Failed to save results file: {e}")

    print_header("Diagnostic Complete")

    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARN] Test interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
