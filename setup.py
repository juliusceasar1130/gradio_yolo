# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:32；主要修改内容：创建项目安装配置文件

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="yolo-detector",
    version="1.0.0",
    author="chenliang",
    author_email="",
    description="基于YOLOv11的图像检测和分割工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "yolo-detector=yolo_detector.main:main",
        ],
    },
)
