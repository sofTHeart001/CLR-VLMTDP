# VLM-TDP 论文解析总结

**论文标题**: VLM-TDP: VLM-guided Trajectory-conditioned Diffusion Policy for Robust Long-Horizon Manipulation
**arXiv**: 2507.04524
**机构**: 腾讯 Robotics X
**年份**: 2025

---

## 📄 核心贡献

### 1. 问题解决

| 问题 | 解决方案 | 效果 |
|------|----------|------|
| 长时序任务成功率低 | VLM任务分解为子任务 | 成功率提升 44% |
| 噪声环境性能衰减 | 体素轨迹引导 | 性能衰减减少 20% |
| 视觉依赖过强 | 多模态条件输入 | 更强的鲁棒性 |

### 2. 方法创新

**VLM-TDP 框架**:
- **任务分解**: 将长时序任务分解为可管理的子任务
- **轨迹生成**: VLM生成体素空间的3D轨迹
- **条件扩散**: 轨迹条件化的扩散策略

---

## 🔧 技术细节

### 体素轨迹表示

**体素尺寸**: 6 × 6 × 6
- **M (X方向)**: 6
- **N (Y方向)**: 6
- **K (Z方向/高度)**: 6

**生成方式**:
1. 前相机图像投影为俯视图
2. 分割为 M×N×K 网格
3. 使用 mask-based visual prompting
4. VLM 选择路径上的 voxel

**表示方法**:
- 轨迹上的 voxel 标记为出现顺序
- 非轨迹 voxel 标记为 0
- 形成紧凑的 6×6×6 矩阵表示

### 子任务定义

**定义**: 操作任务的离散阶段
- **开始**: 打开或关闭夹爪
- **结束**: 关闭或打开夹爪
- **含义**: 表示机器人与对象的完整交互

**示例** (Put Item in Drawer):
1. "grasp the bottom drawer handle"
2. "pull the drawer out"
3. "grasp the item"
4. "put the item into the bottom drawer"

### 扩散策略 (原始 DDPM)

**训练目标**: DDPM (Denoising Diffusion Probabilistic Models)

**损失函数**:
```
L = MSE(εk, εθ(Ot, T, A0t + εk, k))
```

其中:
- εk: 添加的真实噪声
- εθ: 预测的噪声
- Ot: 观测 (RGB图像)
- T: 轨迹条件 (体素矩阵)
- A0t: 原始动作
- k: 扩散步数

**参数设置**:
- **预测步数 (T)**: 12
- **执行步数 (N)**: 8
- **训练轮数**: 500 epochs
- **评估**: 每 50 epochs 评估 20 episodes
- **结果**: 取最高的 5 次成功率平均值

### 轨迹编码器

**架构**: 3层 3D CNN
- 输入: 6×6×6 体素矩阵
- 输出: 空间特征向量
- 作用: 编码空间信息供扩散策略使用

---

## 🏗️ 系统架构

### 数据流

```
输入任务描述 + 初始图像
    ↓
[VLM] 任务分解
    ↓
子任务列表 {S1, S2, ..., Sn}
    ↓
循环每个子任务:
    [VLM] 生成体素轨迹 T (6×6×6)
    ↓
    [3D CNN] 编码轨迹 → 特征向量
    ↓
    [扩散策略] 生成动作序列
    ↓
    执行前 N 步
    ↓
    进入下一个子任务
```

### 条件融合

**策略输入** (在每时间步):
1. 历史图像观测编码
2. 机器人状态
3. **轨迹编码** (每个子任务只加入一次)

**假设**: 轨迹在每个子任务内保持不变

---

## 📊 仿真环境

### RLBench

**平台**: CoppeliaSim
**机器人**: Franka Panda (7自由度)
**相机设置**:
- 单视图: 前相机
- 多视图: 前相机 + 手腕相机

**评估任务**:
| 任务类别 | 具体任务 |
|----------|----------|
| 堆叠 | Stack Blocks |
| 操作 | Put Item in Drawer, Open Drawer, Close Drawer |
| 抓取 | Pick Up (various objects) |
| 精细操作 | Phone on Base, Open Wine Bottle |
| 复杂 | Water Plants, Sweep to Dustpan |

---

## 🧪 实验结果

### 性能对比 (RLBench任务)

| 任务 | Diffusion Policy | TDP (Ours) | VLM-TDP (Ours) |
|------|------------------|------------|----------------|
| Put Item in Drawer | 0.49 | 0.71 | 0.69 |
| Water Plants | 0.32 | 0.54 | 0.52 |
| Close Microwave | 0.41 | 0.72 | 0.70 |
| Open Drawer | 0.95 | 0.96 | 0.97 |
| Open Wine Bottle | 0.70 | 0.81 | 0.80 |
| Sweep to Dustpan | 0.38 | 0.62 | 0.61 |
| Phone on Base | 0.57 | 0.76 | 0.67 |
| **平均** | **0.55** | **0.71** | **0.67** |

### 长时序任务 (Stack Blocks)

| 配置 | Diffusion Policy | TDP | VLM-TDP |
|------|------------------|-----|---------|
| 1 Block (Pick) | 0.58 | 0.88 | 0.88 |
| 1 Block (Combine) | 0.57 | 0.87 | 0.87 |
| 2 Blocks (Combine) | 0.32 | 0.65 | 0.62 |
| 4 Blocks (Combine) | 0.00 | 0.05 | 0.04 |

**改进**: 在复杂长时序任务上提升超过 100%

### 鲁棒性测试

**噪声环境** (图像噪声 σ):
- σ = 0.32 时: 原始策略降至 0%，TDP 保持 17% 原始性能
- 轨迹条件帮助在短距离补偿噪声

**环境变异** (Colosseum benchmark):
- 背景纹理变化
- 对象纹理变化
- 对象大小变化
- 在所有变异上性能下降更小

---

## 🌍 真实世界实验

### 设置

- **机器人**: Franka Emika Panda
- **夹爪**: 平行夹爪
- **相机**: Intel RealSense D435i
- **数据收集**: SpaceMouse 6-DOF 远程操作
- **每个任务**: 40 成功演示

### 任务

| 任务 | Diffusion Policy | TDP (Ours) |
|------|------------------|------------|
| Pick One Orange | 0.70 | 0.85 |
| Pick Two Bananas | 0.70 | 0.95 |
| Pick Three Bananas | 0.20 | 0.70 |

**观察**: 随着任务复杂度增加，性能差距更加明显

---

## 🎯 与本项目改进的对应关系

### 原始 VLM-TDP

| 组件 | 实现 |
|------|------|
| 扩散模型 | DDPM (16步去噪) |
| 体素编码 | 标准3D CNN |
| 规划方式 | 开环 (一次性生成所有子任务) |

### CLR-VLMTDP 改进

| 改进 | 目标 | 预期效果 |
|------|------|----------|
| **Flow Matching** | 替代DDPM | 训练-60%, 推理+40% |
| **轻量编码器** | 深度可分离3D卷积 | 参数-40% |
| **闭环重规划** | 生成-执行-验证循环 | 成功率+15% |

---

## 📚 关键引用

1. **DDPM**: Ho et al., "Denoising diffusion probabilistic models", NeurIPS 2020
2. **Diffusion Policy**: Chi et al., "Diffusion policy", arXiv 2023
3. **RLBench**: James et al., "RLBench: The robot learning benchmark", IEEE RA-L 2020
4. **Flow Matching**: Lipman et al., "Flow matching for generative modeling", arXiv 2022

---

## ✅ 实现要点总结

### 必须遵循的论文实现

- [x] 体素尺寸: 6×6×6
- [x] 子任务定义: 夹爪开/闭为边界
- [x] 训练参数: T=12, N=8, 500 epochs
- [x] 评估方式: 20 episodes × 50 epochs
- [x] 损失函数: MSE
- [x] 仿真平台: RLBench (Franka Panda)
- [x] 相机设置: 前视图 + 手腕视图 (长时序)

### 项目改进实现

- [ ] Flow Matching 替代 DDPM
- [ ] 深度可分离3D卷积
- [ ] 闭环重规划机制
- [ ] 本地 VLM 部署 (替代 GPT-4o)

---

**文档更新时间**: 2025-06-30
**解析工具**: pypdf