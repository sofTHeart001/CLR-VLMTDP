# RLBench 安装指南

RLBench 主要在 Linux 环境下运行。Windows 用户需要使用 WSL (Windows Subsystem for Linux)。

---

## 🖥️ Windows 用户安装指南 (推荐: WSL)

### 前置条件

1. Windows 10/11
2. 支持 WSL 2 的系统

### 步骤 1: 启用 WSL

以管理员身份打开 PowerShell，运行：

```powershell
wsl --install
```

完成后重启电脑。

### 步骤 2: 安装 Ubuntu (推荐)

```powershell
wsl --install -d Ubuntu-22.04
```

### 步骤 3: 配置 WSL

首次启动 Ubuntu 后，设置用户名和密码，然后运行：

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y python3 python3-pip python3-venv git
```

### 步骤 4: 安装 RLBench

在 WSL Ubuntu 环境中运行：

```bash
# 导航到项目目录（假设在Windows D盘挂载为/mnt/d）
cd /mnt/d/Desktop/github_project/CLR-VLMTDP

# 运行安装脚本
bash scripts/install_rlbench.sh
```

或者手动安装：

```bash
# 安装系统依赖
sudo apt-get update
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

# 升级pip
pip install --upgrade pip setuptools wheel

# 安装RLBench
pip install rlbench yarr

# 安装项目依赖
pip install -r requirements.txt
```

### 步骤 5: 设置无头渲染

创建或编辑 `~/.bashrc`，添加：

```bash
export PYREPCG_GL='egl'  # 无头渲染
export MESA_GL_VERSION_OVERRIDE='3.3'  # OpenGL版本
```

重新加载配置：

```bash
source ~/.bashrc
```

### 步骤 6: 测试安装

```bash
cd /mnt/d/Desktop/github_project/CLR-VLMTDP

# 测试RLBench
python -c "
import os
os.environ['PYREPCG_GL'] = 'egl'
from rlbench import Environment
print('RLBench installed successfully!')
"
```

---

## 🐧 Linux 用户安装指南

### 前置条件

- Ubuntu 20.04+ 或其他主流 Linux 发行版
- Python 3.8+
- NVIDIA 驱动（如果使用GPU）

### 自动安装

```bash
cd /path/to/CLR-VLMTDP

# 运行安装脚本
bash scripts/install_rlbench.sh
```

### 手动安装

```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install -y \
    libglfw3 \
    libglfw3-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglu1-mesa-dev \
    ffmpeg

# 安装Python依赖
pip install --upgrade pip
pip install rlbench yarr

# 安装项目依赖
pip install -r requirements.txt

# 设置环境变量
export PYREPCG_GL='egl'
```

---

## 🧪 测试安装

### 1. 测试基础功能

```bash
python3 -c "
import os
os.environ['PYREPCG_GL'] = 'egl'
from rlbench import Environment
from rlbench.tasks import StackBlocks

env = Environment(headless=True)
env.launch()
task = env.get_task(StackBlocks)
print('RLBench test PASSED!')
env.shutdown()
"
```

### 2. 生成演示数据

```bash
# 使用模拟窗口测试（首次会下载环境数据）
python scripts/rlbench_environment.py

# 或使用无头模式生成演示数据
python scripts/generate_demos.py \
    --tasks stack_blocks \
    --num_demos 5 \
    --no_headless  # 显示GUI便于观察
```

### 3. 测试闭环控制

```bash
# 使用真实RLBench环境
python scripts/test_closed_loop.py \
    --task stack_blocks \
    --config config/deploy.yaml
```

---

## 🔧 常见问题

### Q: 首次运行时卡住或报错

A: 首次运行会下载 RLBench 环境数据（约2GB），会自动缓存到 `~/.cache/rlbench/`。请耐心等待。

### Q: WSL 中无显示支持

A: 使用 `headless=True` 参数，或安装 X Server（如 VcXsrv）以支持GUI显示。

### Q: 导入错误: `No module named 'rlbench'`

A: 确保在正确的虚拟环境中，重新运行安装步骤。

### Q: EGL 错误: `EGL not available`

A: 设置环境变量 `export PYREPCG_GL='egl'`，或使用 `headless=False` 测试。

### Q: OpenGL 错误

A: 安装 Mesa 库:
```bash
sudo apt-get install mesa-utils
glxinfo | grep "OpenGL version"
```

---

## 📚 支持的任务

RLBench 支持 100+ 标准任务，项目已集成以下任务：

| 任务名称 | 任务描述 |
|---------|----------|
| `stack_blocks` | 堆叠方块 |
| `put_item_in_drawer` | 将物品放入抽屉 |
| `open_drawer` | 打开抽屉 |
| `close_drawer` | 关闭抽屉 |
| `pick_up_cup` | 抓起杯子 |
| `reach_target` | 伸向目标 |
| `water_plants` | 浇植物 |
| `close_microwave` | 关闭微波炉 |
| `open_wine_bottle` | 打开酒瓶 |
| `sweep_to_dustpan` | 扫地 |
| `phone_on_base` | 将手机放回底座 |

---

## 🚀 快速开始

```bash
# 1. 安装RLBench
bash scripts/install_rlbench.sh

# 2. 生成演示数据
python scripts/generate_demos.py --tasks stack_blocks --num_demos 100

# 3. 测试环境
python scripts/test_closed_loop.py --task stack_blocks --no_headless

# 4. 训练模型
python scripts/train_flow_tdp.py --config config/train.yaml
```

---

## 📖 参考资料

- [RLBench GitHub](https://github.com/stepjam/RLBench)
- [RLBench Paper](https://arxiv.org/abs/1909.12240)
- [Franka Panda Documentation](https://www.franka.de/)
- [CoppeliaSim (V-REP)](http://www.coppeliarobotics.com/)

---

**安装问题？** 请查看 [常见问题](#-常见问题) 或提交 Issue。