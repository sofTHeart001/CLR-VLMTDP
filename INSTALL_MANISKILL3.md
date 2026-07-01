# ManiSkill3 安装指南

**ManiSkill3** - 支持Windows的GPU加速机器人仿真环境

---

## 🚀 快速安装（Windows）

### 步骤 1: 创建Conda环境

```bash
# 创建环境（需要时间，约10-15分钟）
conda env create -f environment_maniskill3.yml

# 激活环境
conda activate clr_vlmtdp_maniskill3
```

### 步骤 2: 安装Visual C++工具（Windows必需）

如果提示Visual C++错误，需要安装：

1. 下载 [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 安装时选择 "C++ build tools"
3. 重启电脑

### 步骤 3: 测试安装

```bash
# 运行测试脚本
python scripts/install_maniskill3.py

# 或直接测试环境
python scripts/maniskill_environment.py
```

---

## ✅ 验证安装

成功后你应该看到：

```
✅ All ManiSkill3 tests PASSED!

Available ManiSkill3 environments:
Total: 1000+
  1. PickCube-v0
  2. StackCube-v0
  3. OpenCabinetDrawer-v0
  4. CloseCabinetDrawer-v0
  ...
```

---

## 📊 支持的任务

ManiSkill3支持1000+任务，包括：

### 基础抓取类
- `PickCube-v0` - 抓取立方体
- `PickYCB-v0` - 抓取YCB物体
- `GraspBlock-v0` - 抓取方块

### 堆叠类
- `StackCube-v0` - 堆叠立方体
- `StackCubes2-v0` - 堆叠2个立方体
- `StackBlocks-v0` - 堆叠多个方块

### 开关类
- `OpenCabinetDrawer-v0` - 打开抽屉
- `CloseCabinetDrawer-v0` - 关闭抽屉
- `OpenCabinetDoor-v0` - 打开门

### 容器类
- `PutObjectInContainer-v0` - 放入容器
- `PutUObjectInContainer-v0` - 放U型物体

### 复杂操作类
- `AssembleLego-v0` - 搭建乐高
- `PlugCharger-v0` - 插入充电器
- `CutRope-v0` - 切断绳子

查看完整列表：
```python
from mani_skill2.utils.registration import REGISTERED_ENVS
print(list(REGISTERED_ENVS.keys()))
```

---

## 🎯 项目集成

### 使用ManiSkill3环境

```bash
# 1. 激活环境
conda activate clr_vlmtdp_maniskill3

# 2. 测试闭环控制（使用ManiSkill环境）
python scripts/test_closed_loop.py --task PickCube-v0 --env maniskill

# 3. 生成演示数据
python scripts/generate_demos_maniskill.py --tasks PickCube-v0 --num_demos 100
```

### 代码中切换环境

```python
from scripts.maniskill_environment import create_maniskill_env

# 创建ManiSkill环境
env = create_maniskill_env(
    task_name="PickCube-v0",
    obs_mode="rgbd",
    num_envs=1
)

# 使用（与RLBench相同的接口）
obs = env.reset()
obs, reward, done, info = env.step(action)
```

---

## ⚡ ManiSkill3 vs RLBench 对比

| 特性 | ManiSkill3 | RLBench |
|------|------------|---------|
| **Windows支持** | ✅ 原生支持 | ❌ 需要WSL |
| **GPU加速** | ✅ 100x faster | ❌ CPU |
| **任务数量** | 1000+ | 100+ |
| **并行仿真** | ✅ 支持 | ❌ 单一 |
| **物理引擎** | SAPIEN | CoppeliaSim |
| **观测模式** | RGBD, 点云 | RGB, 深度 |
| **机器人** | 多种 | Franka Panda |

---

## 🔧 常见问题

### Q: 安装失败，提示缺少Visual C++

A: Windows需要Visual C++编译器：
```bash
# 下载并安装
https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 选择 "C++ build tools" 组件
```

### Q: CUDA错误或GPU不可用

A: 检查CUDA安装：
```bash
nvidia-smi  # 检查CUDA驱动

# 如果没有CUDA，可以降级使用CPU模式
pip uninstall mani-skill sapien
pip install mani-skill sapien --no-build-isolation  # 使用CPU版本
```

### Q: 导入错误

A: 确保激活了正确的环境：
```bash
conda activate clr_vlmtdp_maniskill3

# 重新安装
conda env update -f environment_maniskill3.yml
```

### Q: 环境运行很慢

A: ManiSkill3默认使用GPU加速，确保CUDA工作：
```bash
# 测试CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如果False，安装CUDA或使用CPU模式
```

---

## 📖 参考资料

- **GitHub**: [mani-skill/ManiSkill](https://github.com/mani-skill/ManiSkill)
- **文档**: [maniskill.readthedocs.io](https://maniskill.readthedocs.io/)
- **论文**: ManiSkill3: GPU Parallelized Robotics Simulation

---

## 🚀 快速开始

```bash
# 1. 安装环境
conda env create -f environment_maniskill3.yml
conda activate clr_vlmtdp_maniskill3

# 2. 测试安装
python scripts/install_maniskill3.py

# 3. 测试环境
python scripts/maniskill_environment.py

# 4. 生成演示数据
python scripts/generate_demos_maniskill.py --tasks PickCube-v0 --num_demos 10

# 5. 测试闭环控制
python scripts/test_closed_loop.py --task PickCube-v0 --env maniskill
```

---

**需要帮助？** 查看上方常见问题或提交Issue。