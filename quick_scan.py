#!/usr/bin/env python3
"""
Quick Axis Camera Scanner
Fast scan for common IP ranges
"""

import socket
import requests
from requests.auth import HTTPDigestAuth
from datetime import datetime
import concurrent.futures

def check_camera(ip, username="root", password="root"):
    """Check if IP is an Axis camera"""
    try:
        # Quick socket check
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((ip, 80))
        sock.close()

        if result != 0:
            return None

        # Try HTTP request
        response = requests.get(f"http://{ip}", timeout=2)

        if response.status_code == 200:
            server = response.headers.get('Server', '')

            # Check if it's likely an Axis camera
            if 'Axis' in server or response.text.lower().find('axis') != -1:
                return {
                    'ip': ip,
                    'status': 'FOUND',
                    'server': server,
                    'accessible': True
                }
            else:
                # Still check if it requires auth
                return {
                    'ip': ip,
                    'status': 'DEVICE_FOUND',
                    'server': server,
                    'accessible': True
                }

    except Exception as e:
        return {
            'ip': ip,
            'status': 'PORT_OPEN_BUT_NO_HTTP',
            'error': str(e)[:50]
        }

    return None

def main():
    print("Quick Axis Camera Scanner")
    print("=" * 60)
    print("Scanning common IP ranges...")
    print()

    # Common IP ranges for cameras
    ip_ranges = [
        # Try the configured IP first
        ["192.22.39.253"],

        # Common camera IP ranges
        [f"192.22.39.{i}" for i in [100, 101, 102, 103, 150, 200, 253]],
        [f"192.168.1.{i}" for i in [100, 101, 102, 103, 150, 200, 253]],
        [f"192.168.0.{i}" for i in [100, 101, 102, 103, 150, 200, 253]],
    ]

    found_devices = []

    for ip_list in ip_ranges:
        print(f"Scanning: {ip_list}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ip = {executor.submit(check_camera, ip): ip for ip in ip_list}

            for future in concurrent.futures.as_completed(future_to_ip):
                result = future.result()
                if result:
                    found_devices.append(result)
                    print(f"\n>>> Device Found: {result['ip']}")
                    print(f"    Status: {result['status']}")
                    if 'server' in result:
                        print(f"    Server: {result['server']}")
                    if 'accessible' in result:
                        print(f"    HTTP Accessible: {result['accessible']}")
                    print()

    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)

    if found_devices:
        print(f"\nFound {len(found_devices)} device(s):")
        for device in found_devices:
            print(f"  - {device['ip']} ({device['status']})")
    else:
        print("\nNo devices found in common IP ranges.")
        print("\nNext steps:")
        print("1. Check camera power and network cable")
        print("2. Use Axis IP Utility to find the camera")
        print("3. Check router for DHCP client list")

    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
