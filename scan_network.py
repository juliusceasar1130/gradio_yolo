#!/usr/bin/env python3
"""
Network scanner to find Axis cameras
Scans common IP ranges
"""

import socket
import requests
import threading
from datetime import datetime

def check_ip(ip, port=80, timeout=2):
    """Check if IP:port is accessible"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

def scan_range(start_ip, end_ip):
    """Scan IP range for open port 80"""
    found = []

    for i in range(start_ip, end_ip + 1):
        ip = f"192.168.39.{i}"
        if check_ip(ip, 80, 2):
            found.append(ip)

    return found

def main():
    print("Scanning network for Axis cameras...")
    print("Scanning 192.168.39.0/24 for devices with port 80 open...")
    print()

    # Common IP ranges to scan
    ranges = [
        (1, 254, "192.168.39.0/24"),
        (1, 254, "192.168.1.0/24"),
        (1, 254, "192.168.0.0/24")
    ]

    for start, end, network in ranges:
        print(f"Scanning {network}...")
        found = scan_range(start, end)

        if found:
            print(f"\nFound {len(found)} device(s) with port 80 open:")
            for ip in found:
                print(f"  - {ip}")

                # Try to get more info
                try:
                    response = requests.get(f"http://{ip}", timeout=3)
                    server = response.headers.get('Server', 'Unknown')
                    print(f"    Server: {server}")
                except:
                    print(f"    (No HTTP response)")
        else:
            print("  No devices found.")

    print("\nScan complete!")

if __name__ == "__main__":
    main()
