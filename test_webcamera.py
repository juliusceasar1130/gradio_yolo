#!/usr/bin/env python3
"""
Test using your existing webcamera.py
Direct test of camera capture
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from your webcamera.py
from camera.webcamera import Camera

# Configuration
CAMERA_IP = "192.22.39.253"

def main():
    print("=" * 70)
    print("Testing with webcamera.py")
    print("=" * 70)
    print(f"Camera URL: http://{CAMERA_IP}")
    print(f"Username: root")
    print(f"Password: root")
    print()

    # Create camera instance
    camera = Camera(CAMERA_IP, "TestCamera")

    # Test snapshot capture
    print("Attempting to capture photo...")
    print(f"URL will be: http://{CAMERA_IP}/axis-cgi/jpg/image.cgi?resolution=1280x720&compression=25&camera=1")
    print()

    try:
        image_data = camera.capture_photo()

        if image_data:
            print("[SUCCESS] Image captured!")
            print(f"Image size: {len(image_data)} bytes")

            # Save the image
            save_path = camera.save_photo("test_axis")
            print(f"Image saved to: {save_path}")
        else:
            print("[FAIL] Failed to capture image")
            print("\nPossible reasons:")
            print("1. Camera is not reachable")
            print("2. IP address is incorrect")
            print("3. Camera is not powered on")
            print("4. Network cable is disconnected")
            print("5. Username/password is incorrect")

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
