# 创建者/修改者: chenliang；修改时间：2025年7月27日 22:45；主要修改内容：创建测试运行脚本

"""
测试运行脚本

提供便捷的测试运行和报告生成功能
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", coverage=False, verbose=False, output_dir="test_reports"):
    """
    运行测试
    
    Args:
        test_type: 测试类型 ("all", "unit", "integration", "slow")
        coverage: 是否生成覆盖率报告
        verbose: 是否详细输出
        output_dir: 输出目录
    """
    project_root = Path(__file__).parent.parent
    output_path = project_root / output_dir
    output_path.mkdir(exist_ok=True)
    
    # 基础命令
    cmd = ["python", "-m", "pytest"]
    
    # 添加测试类型标记
    if test_type != "all":
        cmd.extend(["-m", test_type])
    
    # 添加详细输出
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # 添加覆盖率选项
    if coverage:
        cmd.extend([
            "--cov=src/yolo_detector",
            f"--cov-report=html:{output_path}/coverage_html",
            f"--cov-report=xml:{output_path}/coverage.xml",
            f"--cov-report=term"
        ])
    
    # 添加JUnit XML报告
    cmd.extend([
        f"--junit-xml={output_path}/junit.xml"
    ])
    
    # 添加HTML报告
    cmd.extend([
        f"--html={output_path}/report.html",
        "--self-contained-html"
    ])
    
    print(f"运行命令: {' '.join(cmd)}")
    print(f"输出目录: {output_path}")
    
    try:
        # 运行测试
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        # 输出结果
        print("\n=== 测试输出 ===")
        print(result.stdout)
        
        if result.stderr:
            print("\n=== 错误输出 ===")
            print(result.stderr)
        
        # 保存输出到文件
        with open(output_path / "test_output.txt", "w", encoding="utf-8") as f:
            f.write("=== 测试命令 ===\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("=== 标准输出 ===\n")
            f.write(result.stdout + "\n\n")
            if result.stderr:
                f.write("=== 错误输出 ===\n")
                f.write(result.stderr + "\n")
        
        # 返回结果
        if result.returncode == 0:
            print(f"\n✅ 测试通过！报告已保存到: {output_path}")
            return True
        else:
            print(f"\n❌ 测试失败！返回码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return False


def install_test_dependencies():
    """安装测试依赖"""
    dependencies = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-html>=3.0.0",
        "pytest-mock>=3.0.0"
    ]
    
    print("安装测试依赖...")
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
            print(f"✅ 已安装: {dep}")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败: {dep} - {e}")
            return False
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行YOLO检测器测试")
    
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "slow"],
        default="all",
        help="测试类型"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="生成覆盖率报告"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    
    parser.add_argument(
        "--output-dir",
        default="test_reports",
        help="输出目录"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="安装测试依赖"
    )
    
    args = parser.parse_args()
    
    # 安装依赖
    if args.install_deps:
        if not install_test_dependencies():
            sys.exit(1)
    
    # 运行测试
    success = run_tests(
        test_type=args.type,
        coverage=args.coverage,
        verbose=args.verbose,
        output_dir=args.output_dir
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
