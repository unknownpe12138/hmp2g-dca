#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VSCode 一键运行 DCA 环境
双击运行或在 VSCode 中按 F5 运行
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50 + "\n")


def check_python():
    """检查 Python 环境"""
    print_section("检查 Python 环境")
    print(f"Python 版本: {sys.version}")
    print(f"Python 路径: {sys.executable}")
    return True


def install_dependencies():
    """安装依赖"""
    print_section("安装依赖包")

    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print(f"警告: {requirements_file} 不存在，跳过依赖安装")
        return True

    print(f"从 {requirements_file} 安装依赖...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ 依赖安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖安装失败: {e}")
        return False


def build_cython():
    """编译 Cython 模块"""
    print_section("编译 Cython 模块")

    setup_file = Path(__file__).parent / "setup.py"
    if not setup_file.exists():
        print("警告: setup.py 不存在，跳过 Cython 编译")
        return True

    print("编译 Cython 扩展模块...")
    try:
        subprocess.check_call([
            sys.executable, "setup.py", "build_ext", "--inplace"
        ])
        print("✓ Cython 模块编译成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Cython 编译失败: {e}")
        print("\n提示: 如果编译失败，请确保已安装 C 编译器:")
        if platform.system() == "Windows":
            print("  - 安装 Microsoft Visual C++ Build Tools")
            print("  - 或安装 MinGW-w64")
        else:
            print("  - Ubuntu/Debian: sudo apt-get install build-essential")
            print("  - CentOS/RHEL: sudo yum groupinstall 'Development Tools'")
        return False


def create_directories():
    """创建必要的目录"""
    print_section("创建目录结构")

    dirs_to_create = ["TEMP", "TEMP/build", "RESULT"]

    for dir_name in dirs_to_create:
        dir_path = Path(__file__).parent / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_name}")

    return True


def get_default_config():
    """获取默认配置文件路径"""
    config_dir = Path(__file__).parent / "DOCS" / "examples" / "dca"
    config_files = [
        "example_dca.jsonc",
        "train_old_dca.jsonc",
    ]

    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            return config_path

    return None


def run_dca(config_path=None):
    """运行 DCA 环境"""
    print_section("启动 DCA 环境")

    # 确定配置文件
    if config_path is None:
        config_path = get_default_config()

    if config_path is None or not config_path.exists():
        print("✗ 未找到配置文件!")
        print("\n可用的配置文件:")
        print("  1. DOCS/examples/dca/example_dca.jsonc")
        print("  2. DOCS/examples/dca/train_old_dca.jsonc")
        print("\n请手动运行:")
        print("  python main.py -c <配置文件路径>")
        return False

    print(f"使用配置: {config_path.relative_to(Path(__file__).parent)}")
    print("\n开始运行...\n")

    # 运行主程序
    main_file = Path(__file__).parent / "main.py"
    try:
        subprocess.run([
            sys.executable, str(main_file), "-c", str(config_path)
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 运行失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n用户中断运行")
        return True


def main():
    """主函数"""
    print("\n" + "=" * 50)
    print("  HMP2G-DCA 一键启动程序")
    print("=" * 50)

    # 检查命令行参数
    args = sys.argv[1:]
    skip_deps = "--skip-deps" in args
    skip_build = "--skip-build" in args

    # 清理参数
    if skip_deps:
        args.remove("--skip-deps")
    if skip_build:
        args.remove("--skip-build")

    # 如果有配置文件参数
    config_path = None
    if args:
        config_path = Path(args[0])

    # 步骤 1: 检查 Python
    if not check_python():
        print("\n✗ Python 环境检查失败")
        return 1

    # 步骤 2: 安装依赖
    if not skip_deps:
        if not install_dependencies():
            print("\n✗ 依赖安装失败")
            return 1
    else:
        print("\n跳过依赖安装 (--skip-deps)")

    # 步骤 3: 编译 Cython
    if not skip_build:
        if not build_cython():
            print("\n✗ Cython 编译失败")
            return 1
    else:
        print("\n跳过 Cython 编译 (--skip-build)")

    # 步骤 4: 创建目录
    if not create_directories():
        print("\n✗ 目录创建失败")
        return 1

    # 步骤 5: 运行 DCA
    if not run_dca(config_path):
        return 1

    print("\n" + "=" * 50)
    print("  程序结束")
    print("=" * 50 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
