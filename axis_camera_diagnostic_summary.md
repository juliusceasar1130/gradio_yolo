# Axis Camera Connection Diagnostic Summary

## Problem Summary

**Issue**: Unable to connect to Axis camera at IP 192.168.39.253
- Connection timeout errors
- Error code 10035 (WSAEWOULDBLOCK)
- Both OpenCV and manual HTTP methods failed

## Diagnostic Results

### Test 1: Network Connectivity
- **Result**: FAILED
- **Error**: Cannot connect to 192.168.39.253:80
- **Conclusion**: Camera is not reachable on the network

### Test 2: Quick Network Scan
- **Scanned IPs**:
  - 192.168.39.253
  - 192.168.39.100-103, 150, 200
  - 192.168.1.100-103, 150, 200
  - 192.168.0.100-103, 150, 200
- **Result**: No Axis cameras found in common IP ranges

## Root Causes

1. **Camera is not powered on**
2. **Network cable disconnected**
3. **Incorrect IP address**
4. **Camera on different network segment**
5. **Firewall blocking connection**
6. **IP address conflict**

## Solutions Provided

### 1. Enhanced Camera Capture Module
**File**: `src/yolo_detector/core/camera_capture.py`

**Improvements**:
- Support for 6 different URL formats
- Multiple authentication methods (Digest, Basic, Anonymous)
- Automatic fallback between connection methods
- Connection diagnostic tool (`test_connection()`)

**Key Functions**:
- `_connect_axis_opencv()`: Try multiple URL patterns with OpenCV
- `_connect_axis_manual()`: Try multiple URL patterns with requests + Digest Auth
- `test_connection()`: Diagnostic test without actual connection

### 2. Diagnostic Test Scripts

#### Script 1: Simple Test (English)
**File**: `test_axis_en.py`
- Tests network connectivity
- Tests HTTP access
- Tests MJPEG endpoints
- Tests authentication
- Tests snapshot capture

#### Script 2: Existing Code Test
**File**: `test_webcamera.py`
- Tests using your existing `webcamera.py` code
- Captures and saves snapshot
- Uses Digest Auth (root/root)

### 3. Network Scanner
**File**: `quick_scan.py`
- Fast scan for cameras in common IP ranges
- Uses multi-threading for speed
- Identifies potential Axis cameras

## Quick Testing Commands

### Test with webcamera.py
```bash
cd D:\00deeplearn\yolo11\gradio_tool
python test_webcamera.py
```

### Manual HTTP Test
```bash
# Test if camera is reachable
ping 192.168.39.253

# Try HTTP access
curl -v http://192.168.39.253

# Try MJPEG stream
curl -v http://192.168.39.253/axis-cgi/mjpg/video.cgi
```

### Find Camera IP

#### Method 1: Router DHCP Client List
1. Access router web interface (usually http://192.168.39.1)
2. Check DHCP client list
3. Look for "Axis" or similar device names

#### Method 2: Axis IP Utility
1. Download from: https://www.axis.com/support/tools
2. Install and run
3. It will automatically find Axis cameras on your network

#### Method 3: Command Line Scan
```cmd
# Check local network range
ipconfig

# Use nmap if available
nmap -sn 192.168.39.0/24
```

## Verification Steps

Once you find the correct camera IP:

1. **Verify accessibility**:
   ```python
   import requests
   response = requests.get("http://CORRECT_IP")
   print(response.status_code)  # Should be 200
   ```

2. **Test snapshot capture**:
   ```python
   from camera.webcamera import Camera
   cam = Camera("CORRECT_IP", "Test")
   img = cam.capture_photo()
   if img:
       print("Success!")
   ```

3. **Update configuration**:
   - Update camera IP in your application
   - Test connection with new IP

## Code Changes Summary

### Modified Files
1. **`src/yolo_detector/core/camera_capture.py`**
   - Enhanced `_connect_axis_opencv()` with 6 URL formats
   - Enhanced `_connect_axis_manual()` with 3 auth methods
   - Added `test_connection()` diagnostic method

### Created Files
1. **`docs/axis_camera_setup.md`** - Configuration guide
2. **`test_axis_camera.py`** - Full diagnostic script (requires project dependencies)
3. **`test_axis_simple.py`** - Simplified test (Chinese, encoding issues)
4. **`test_axis_en.py`** - English version diagnostic script
5. **`test_webcamera.py`** - Test using existing webcamera.py
6. **`scan_network.py`** - Network scanner (slow, full range)
7. **`quick_scan.py`** - Fast scanner for common IPs

## Next Steps

1. **Verify Camera Status**:
   - Check power indicator light
   - Check network indicator light
   - Restart camera (unplug power, wait 10 seconds, plug back in)

2. **Find Correct IP**:
   - Use Axis IP Utility
   - Check router DHCP client list
   - Run quick_scan.py to find device

3. **Test with Found IP**:
   - Run test_webcamera.py with correct IP
   - Capture test image

4. **Update Application**:
   - Update camera IP in code
   - Test complete integration

## Emergency Fallback

If you cannot connect to the Axis camera at all:

1. **Use local USB camera**:
   ```python
   from src.yolo_detector.core.camera_capture import CameraCapture
   camera = CameraCapture(camera_type="local", camera_index=0)
   camera.connect()
   ```

2. **Use network camera with different IP**:
   - Change IP in camera configuration
   - Update application settings

## Support Resources

- Axis Documentation: https://www.axis.com/documentation
- Axis IP Utility: https://www.axis.com/support/tools
- Camera Model Manual: Check your specific model documentation
- Network Troubleshooting: Check router documentation

---

**Generated**: 2025-11-04 17:30:00
**Status**: Awaiting camera IP verification
