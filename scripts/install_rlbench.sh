#!/bin/bash
# RLBench Installation Script
# 用于Linux/WSL环境安装RLBench

set -e  # 遇到错误立即退出

echo "=================================="
echo "RLBench Installation Script"
echo "=================================="
echo

# 检查操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Warning: This script is designed for Linux/WSL. You're on $OSTYPE"
fi

# 检查Python版本
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "   Error: Python 3.8+ required"
    exit 1
fi
echo "   ✓ Python version OK"
echo

# 检查并安装系统依赖
echo "2. Installing system dependencies..."

if [[ "$OS" == "linux" ]]; then
    sudo apt-get update

    # 基础依赖
    sudo apt-get install -y \
        libglfw3 \
        libglfw3-dev \
        libgl1-mesa-glx \
        libglu1-mesa \
        libglu1-mesa-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1

    echo "   ✓ System dependencies installed"
elif [[ "$OS" == "macos" ]]; then
    brew install glfw
    echo "   ✓ System dependencies installed"
fi
echo

# 创建虚拟环境（可选）
if [ "$USE_VENV" = "true" ]; then
    echo "3. Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "   ✓ Virtual environment created and activated"
    echo
fi

# 升级pip
echo "4. Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "   ✓ Pip upgraded"
echo

# 安装RLBench
echo "5. Installing RLBench..."
pip install rlbench
echo "   ✓ RLBench installed"
echo

# 安装yarr（RLBench推荐）
echo "6. Installing yarr..."
pip install yarr
echo "   ✓ yarr installed"
echo

# 安装其他依赖
echo "7. Installing additional dependencies..."
pip install \
    numpy \
    opencv-python \
    matplotlib \
    pyyaml \
    tqdm \
    tensorboard
echo "   ✓ Additional dependencies installed"
echo

# 创建数据目录
echo "8. Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/rlbench_cache
echo "   ✓ Data directories created"
echo

# 测试安装
echo "9. Testing RLBench installation..."
python3 << 'EOF'
import os
os.environ['PYREPCG_GL'] = 'egl'  # 使用EGL进行无头渲染

try:
    from rlbench import Environment
    from rlbench.tasks import StackBlocks

    print("   Testing RLBench import...")
    print("   ✓ RLBench import successful")

    # 创建测试环境
    print("   Creating test environment...")
    env = Environment(
        action_mode='joint_velocity',
        headless=True
    )
    env.launch()
    print("   ✓ Environment launched")

    # 测试任务加载
    task = env.get_task(StackBlocks)
    print("   ✓ Task loaded successfully")

    # 关闭环境
    env.shutdown()
    print("   ✓ Environment closed")
    print("\n   ✅ RLBench installation test PASSED!")

except Exception as e:
    print(f"   ✗ RLBench installation test FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
EOF

echo
echo "=================================="
echo "Installation completed!"
echo "=================================="
echo
echo "Next steps:"
echo "  1. Generate demo data: python scripts/generate_demos.py --tasks stack_blocks --num_demos 10"
echo "  2. Test closed-loop: python scripts/test_closed_loop.py --task stack_blocks"
echo
echo "Note: First time running RLBench will download environment data (~2GB)"
echo "      This will be cached in ~/.cache/rlbench/"
echo