# CLR-VLMTDP 项目实施路线图

> **本文档是接下来所有工作的参考路线图。**
> 包含：当前状态盘点、下一步任务清单、每个任务的详细规格说明、设计决策记录、风险与降级方案。
> 最后更新：2026-07-01

---

## 目录

1. [当前状态盘点](#1-当前状态盘点)
2. [总体目标与设计文档承诺](#2-总体目标与设计文档承诺)
3. [已完成任务（Phase A + B + C + D1）](#3-已完成任务phase-a--b--c--d1)
4. [下一步任务清单（按优先级）](#4-下一步任务清单按优先级)
5. [Phase D2-D5 详细规格](#5-phase-d2-d5-详细规格)
6. [Phase E（高级功能）详细规格](#6-phase-e高级功能详细规格)
7. [实验矩阵](#7-实验矩阵)
8. [数据资产清单](#8-数据资产清单)
9. [代码架构约束](#9-代码架构约束)
10. [风险登记与降级方案](#10-风险登记与降级方案)
11. [里程碑时间表](#11-里程碑时间表)

---

## 1. 当前状态盘点

### 1.1 已完成模块（commit 82b47cc）

| 模块 | 文件 | 状态 | 验证 |
|---|---|---|---|
| 轻量体素编码器 | `models/light_voxel_encoder.py` | ✅ 完成 | param -88%, 单元测试 PASS |
| FlowTDP 策略 | `models/flow_tdp.py` | ✅ 完成 | 单步 Euler 推理；use_voxel 开关；loss 下降测试 PASS |
| VLM 抽象层 | `models/vlm_wrapper.py` | ✅ 完成 | OpenAIVLMWrapper + LocalLLaVAWrapper + 工厂函数；JSON 鲁棒解析 PASS |
| 闭环控制器骨架 | `scripts/test_closed_loop.py` | ⚠️ 部分 | 框架在，依赖 MockEnvironment 跑通；真实环境 rollout 未实现 |
| 体素轨迹生成 | `utils/voxel_trajectory.py` | ✅ 完成 | 单元测试 PASS（含合成数据 blob 投影） |
| 视觉提示 | `utils/visual_prompt.py` | ✅ 完成 | grid overlay + 3D 边框 PASS |
| 轨迹提取 | `utils/trajectory_extraction.py` | ✅ 完成 | 支持 7D 输入、interpolation、clipping PASS |
| Franka FK | `utils/franka_fk.py` | ✅ 完成 | ManiSkill agent + fallback 双路径（fallback 自动触发） |
| Sub-task 切分 | `utils/subtask.py` | ✅ 完成 | 夹爪状态变化检测 PASS |
| 相机参数加载 | `utils/camera_params.py` | ✅ 完成 | 从 h5 读 + 默认 fallback PASS |
| ManiSkill Dataset | `data/maniskill_dataset.py` | ✅ 完成 | 合成数据 PASS；真实数据需要 ManiSkill + 真演示 |
| 实验 1 脚本 | `scripts/exp1_traj_ablation.py` | ✅ 完成 | 跑通；JSON 结果保存 |
| 实验 1 图表 | `scripts/plot_exp1.py` | ✅ 完成 | matplotlib 柱状图生成 |

### 1.2 测试覆盖

```
tests/test_models.py              20 PASS
tests/test_smoke.py                4 PASS
tests/test_voxel_trajectory.py    17 PASS
                                 ──────
                                  41 PASS  (6.68s)
```

### 1.3 关键实验数据

**实验 1**（合成数据，4000 步，GPU 上跑通）：

```
Metric            With-Voxel    Without-Voxel    Delta
Final Loss        0.9988        1.0040           -0.005
Action MSE        0.977         0.981            -0.4%
Inference (ms)    7.5           15.6             -52%  ⭐
Train Time (s)    339.5         357.0            -5%
```

**核心发现**：
- ✅ With-Voxel 模型推理速度 **2x 更快**（特征更稀疏）
- ⚠️ 合成数据上 Action MSE 差异不显著（合成数据体素只含 2 个非零 cell）
- **需要真实 ManiSkill PickCube-v0 演示才能验证论文 Table I 的 +16% 成功率提升**

### 1.4 已知问题 / 局限

| 问题 | 影响 | 缓解 |
|---|---|---|
| 合成数据无法显示体素条件优势 | 不能复现论文 Table I 数字 | 接真实 ManiSkill 数据 |
| `compute_flow_matching_loss` 内有 `torch.autocast` 调用（已移除）| 训练慢 | 已用 fp32 跑通 |
| `Panda()` 在 ManiSkill 3.x 构造需要 scene | FK 回退到 fallback | 自动降级；不影响功能 |
| `load_config` 在多 yaml 文件用相对路径时 | 已通过 Path.resolve 处理 | — |
| VLM-TDP 推理时无 ManiSkill env 跑 rollout | 评测只能用 proxy MSE | 需要接入 ManiSkill env 真跑 |

---

## 2. 总体目标与设计文档承诺

设计文档（`CLR-VLMTDP项目设计说明书.md`）承诺：

| 指标 | 原始 VLM-TDP | CLR-VLMTDP 目标 | 当前实测 |
|---|---|---|---|
| 长时序成功率 | 75% | 87% (+16%) | 未测（需真环境 rollout） |
| 训练时间 | 10h | 4h (-60%) | 设计上 Flow Matching 应更快，未实测 vs DDPM |
| 单步推理 | 14ms | 8.4ms (-40%) | 7.5ms ✅ 已达到 |
| 噪声环境衰减 | 20% | 8% (-60%) | 未测（需 noise_augmentation） |
| 部署方式 | GPT-4o API | 完全本地离线 | ⚠️ 当前用 OpenAI API（与 README 承诺冲突） |

**架构核心**：
- **闭环重规划**（§2.1）：生成-执行-验证-重规划 ✅ 设计在 test_closed_loop.py
- **Flow Matching**（§2.2）：替代 DDPM ✅ 已实现
- **轻量编码器**（§2.3）：深度可分离 3D 卷积 ✅ 已实现（参数 -88%）

---

## 3. 已完成任务（Phase A + B + C + D1）

### 3.1 Phase A — Bug 修复

| 文件 | 改动 |
|---|---|
| `models/flow_tdp.py::sample_action` | Euler 积分从 x_1 ~ N(0,I) 到 x_0 |
| `models/light_voxel_encoder.py` | 删除 `input_proj` 和 `positional_encoding` buffer 死代码 |
| `models/light_voxel_encoder.py::get_parameter_count` | 与 StandardVoxelEncoder 真对比 |
| `scripts/test_closed_loop.py::load_config` | 递归合并 `extends:` |
| `scripts/train_flow_tdp.py::load_config` | 同上 |
| `models/flow_tdp.py::compute_flow_matching_loss` | 移除 autocast（避免 sm_120 dtype 错误） |

### 3.2 Phase B — 真实 VLM

`models/vlm_wrapper.py` 重写：
- `VLMWrapperBase(nn.Module, ABC)` — 抽象基类
- `OpenAIVLMWrapper` — 主实现（OpenAI 兼容 API）
  - 图像输入 → base64 PNG → `image_url` content block
  - JSON 鲁棒解析（retry 一次后抛 VLMJSONParseError）
  - 体素格式归一化（216/6×6×6/6×6/6×(6×6)）
- `LocalLLaVAWrapper` — 离线占位（不主动加载模型）
- `VLMWrapper(*args, backend=...)` — 工厂函数

`config/default.yaml::vlm` 加 `backend`, `api_key`, `base_url`, `model`, `image_max_side`, `temperature`, `timeout`, `max_retries` 字段。

### 3.3 Phase C — 体素轨迹生成

| 文件 | 功能 |
|---|---|
| `utils/voxel_trajectory.py` | 图像+深度 → 6×6×6 体素网格；默认 Franka 相机参数 |
| `utils/visual_prompt.py` | `draw_voxel_grid_overlay`（6×6 编号）+ `draw_voxel_grid_on_image`（3D 边框） |
| `utils/trajectory_extraction.py` | 演示末端轨迹 → 6×6×6 时序 grid；支持 7D 输入 |

### 3.4 Phase D1 — 数据加载 + 实验 1

| 文件 | 功能 |
|---|---|
| `utils/franka_fk.py` | Franka FK；ManiSkill 内置优先，构造失败时降级到 fallback |
| `utils/subtask.py` | 夹爪状态变化切分 sub-task（含 min_length 噪声过滤） |
| `utils/camera_params.py` | 从 h5 读相机参数；缺失时 fallback |
| `data/maniskill_dataset.py` | PyTorch Dataset；支持合成 + 真实 h5 |
| `models/flow_tdp.py` | 加 `use_voxel: bool` 构造参数 |
| `scripts/exp1_traj_ablation.py` | 训两模型 + 评测 + JSON 结果 |
| `scripts/plot_exp1.py` | matplotlib 柱状图生成 |
| `results/exp1_traj_ablation_v2.json` | 实验 1 结果 |
| `results/figures/exp1_*.png` | 图表 |

### 3.5 基础设施修复

| 改动 | 原因 |
|---|---|
| 装 PyTorch 2.12 nightly + cu128 | RTX 5060 sm_120 需要 |
| `scripts/exp1_traj_ablation.py::_safe_device` | 真 backward 测试检测 GPU 可用性 |
| `.gitignore` 更新 | 加 `.venv/` + 排除 `paper_text.txt` + 允许 `results/*.json` + `results/figures/*.png` |

---

## 4. 下一步任务清单（按优先级）

| 优先级 | 任务 | 工作量 | 价值 | 依赖 |
|---|---|---|---|---|
| 🔴 P0 | **D2**: 实验 2 — Flow Matching 步数扫描 | 0.5 天 | 验证推理速度 vs 质量权衡 | D1 |
| 🔴 P0 | **D3**: 实验 3 — VLM 质量评估 (IoU) | 0.5 天 | 验证 VLM 选 cell 能力 | D1 + OpenAI API |
| 🟡 P1 | **真实 ManiSkill 数据接入** | 1-2 天 | 让实验数字有意义 | ManiSkill 安装 |
| 🟡 P1 | **ManiSkill 环境 rollout 评测** | 2 天 | 把 proxy MSE 换成真实成功率 | D1 + 真实数据 |
| 🟡 P1 | **DDPM baseline** | 1 天 | 实验 2 对照组 | — |
| 🟢 P2 | **E1**: 噪声鲁棒性实验 (Fig 3 复现) | 0.5 天 | 复现 Fig 3 的 +17% Pick 鲁棒性 | 真实数据 |
| 🟢 P2 | **E2**: 子任务消融 (有/无 sub-task) | 1 天 | 验证设计文档 §2.1 闭环承诺 | 真实数据 |
| 🟢 P2 | **闭环 vs 开环消融** | 1 天 | 验证 +15% 成功率 | 真实数据 + 真环境 |
| 🟢 P3 | **本地 VLM 切换** | 1-2 天 | 让 README 与设计文档承诺对齐 | LLaVA 本地部署 |
| 🟢 P3 | **Colosseum 变异** | 1 天 | 复现 Fig 4 (ManiSkill 不支持，需降级) | 真实数据 |
| 🟢 P4 | **TensorRT 部署** | 2-3 天 | 推理加速 | — |
| ⚪ P5 | **真实 Franka 机器人** | 依赖硬件 | 论文 §V 真实世界实验 | — |

---

## 5. Phase D2-D5 详细规格

### 5.1 🔴 D2: 实验 2 — Flow Matching 步数扫描

**目标**：复现论文 §III.C 的"单步推理即可"承诺

**假设**：1 步推理 ≈ 4 步推理的成功率（因为 Flow Matching 的速度场是常数场 / 单步足够好）

#### 详细规格

**输入**：实验 1 训好的 `FlowTDP-w-traj` 模型（或重新训一个）

**实验设计**：
```python
# scripts/exp2_fm_steps.py
def run_step_sweep(model, env_or_dataset, num_steps_list=[1, 2, 4, 8, 16]):
    for ns in num_steps_list:
        actions = model.sample_action(image, voxel, num_steps=ns)
        # 评测：成功率（或 MSE proxy）
        # 记录：推理时间 ms
    return results
```

**指标**：
- 成功率 vs 步数（曲线）
- 单步推理耗时 vs 步数（应该线性）
- 总耗时 vs 步数

**预期结果**：
- 1 步成功率 ≥ 4 步的 90%
- 推理耗时线性增长（1 步 < 2 步 < 4 步 < ...）
- 1 步 ≈ 8.4ms（设计文档承诺）

**验收标准**：
- 图表保存到 `results/figures/exp2_fm_steps.png`
- JSON 结果保存到 `results/exp2_fm_steps.json`

**代码位置**：
- 新建 `scripts/exp2_fm_steps.py`
- 复用 `models/FlowTDP::sample_action(num_steps=...)`（已实现多步）

#### 实现步骤

1. **复用实验 1 模型**：跑 `exp1` 时同时保存 checkpoint 到 `weights/exp1_w_traj.pt`
2. **写评测函数**：`@torch.no_grad()` 装饰；遍历 `num_steps` 列表
3. **测时间**：`time.time()` 包住 `sample_action` 调用；多次取平均
4. **画双 y 轴图**：x 轴步数，y1 成功率，y2 耗时
5. **写测试**：断言 1 步和 16 步的输出 shape 相同；耗时比在合理范围（[2x, 20x]）

---

### 5.2 🔴 D3: 实验 3 — VLM 质量评估 (IoU)

**目标**：量化 GPT-4o-mini 在 ManiSkill 演示图像上选 cell 的能力

**假设**：VLM 选出的 6×6×6 体素和真值（演示轨迹提取）IoU ≥ 0.3

#### 详细规格

**输入**：
- ManiSkill PickCube-v0 演示 20 张关键帧（每条 demo 取 1 张）
- 或合成数据 20 张（如果没装 ManiSkill）

**实验设计**：
```python
# scripts/exp3_vlm_quality.py
for image, ground_truth_voxel in test_set:
    # 画 6×6 网格 overlay 在图像上
    overlay = draw_voxel_grid_overlay(image)
    # VLM 输出
    pred_voxel = vlm.decompose_task(overlay, task_desc)
    # 计算 IoU
    iou = compute_iou(pred_voxel, ground_truth_voxel)
    # 记录
```

**指标**：
- 体素 IoU（predicted_occupied vs ground_truth_occupied）
- 中心位置误差 (centroid distance)
- VLM 调用 token 数 / 成本

**预期**：
- IoU ≥ 0.3（VLM 视觉推理不算精确，但应能指对大致区域）
- gpt-4o > gpt-4o-mini（明显）

**验收标准**：
- 5 张示例图保存到 `results/figures/exp3_examples/`
- 指标 JSON 保存到 `results/exp3_vlm_iou.json`

**成本估算**：
- 20 张图 × gpt-4o-mini = ~$0.05（gpt-4o ≈ $1）

#### 实现步骤

1. **构造测试集**：从 ManiSkill 演示（或合成数据）取 20 张关键帧
2. **画 overlay**：用 `draw_voxel_grid_overlay`
3. **VLM 调用**：用 `VLMWrapper(backend="openai")`；prompt 用 `PromptTemplate.format_task_decomposition`
4. **IoU 计算**：`(pred & gt).sum() / (pred | gt).sum()`
5. **可视化**：保存 5 张 overlay + VLM 输出 + 真值的对比图

---

### 5.3 🟡 P1: 真实 ManiSkill 数据接入

**目标**：用 ManiSkill 官方演示替换合成数据

#### 详细规格

**步骤**：
1. 安装 ManiSkill 到 conda env（如果还没装）
2. 下载 PickCube-v0 演示：
   ```bash
   python -m mani_skill.utils.download_demo -t PickCube-v0 -o data/maniskill/
   ```
3. 修改 `data/maniskill_dataset.py` 支持真实 ManiSkill 路径
4. 跑 `exp1` 时用真实数据（不再用 `--synthetic`）

**已知 ManiSkill 兼容性问题**：
- `Panda()` agent 构造需要 scene（已在 franka_fk.py 加 try/except）
- 需要装 `pinocchio`（conda-forge 包）以获得更好的 FK 精度

**验收**：
- 真实数据加载 → 训练 → 实验 1 结果显著优于合成数据
- 真实数据的 Action MSE < 1.0（合成数据 0.97）

---

### 5.4 🟡 P1: ManiSkill 环境 rollout 评测

**目标**：把 Action MSE proxy 换成真实成功率

#### 详细规格

**接口需求**：
```python
class ManiSkillAdapter:
    """把 ManiSkill env 适配到 controller 期望的接口"""
    def reset(self) -> dict: ...  # 返回 {'rgb': (3, H, W)}
    def step(self, action: torch.Tensor) -> tuple[dict, float, bool, dict]:
        """返回 (obs, reward, done, info)"""
```

**评测流程**：
```python
def evaluate(model, env, num_episodes=30):
    successes = 0
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = model.sample_action(obs['rgb'], voxel)
            obs, reward, done, info = env.step(action)
        if info.get('success'):
            successes += 1
    return successes / num_episodes
```

**任务**：PickCube-v0，30 episodes，500 步上限

**验收**：
- With-Voxel 成功率 ≥ Without-Voxel 成功率
- 报告保存到 `results/exp1_real_env.json`

---

### 5.5 🟡 P1: DDPM baseline

**目标**：实验 2 的对照组，证明 Flow Matching 不只是比"无 DDPM 训练"快

#### 详细规格

**新文件**：`models/ddpm_policy.py`

**设计**：
- 复用 FlowTDP 的 encoder / decoder 架构
- 训练目标：预测噪声（DDPM 标准）
- 推理：16 步去噪
- 与 FlowTDP 用**同一数据**训

**指标**：
- 训练时间（5000 步）
- 推理时间（16 步去噪）
- 成功率（如果有真实数据）

**验收**：
- FlowTDP 训练 ≤ DDPM 训练时间的 50%（4h vs 10h 承诺）
- FlowTDP 1 步推理 ≤ DDPM 16 步推理时间的 30%

---

### 5.6 🟢 E1: 噪声鲁棒性实验

**目标**：复现论文 Fig 3 的 σ ∈ {0.08, 0.16, 0.32, 0.64} 噪声衰减曲线

#### 详细规格

**新增文件**：`utils/noise_augmentation.py`

```python
def add_gaussian_noise(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """给图像加高斯噪声 (论文 Fig 3)"""
    noise = torch.randn_like(image) * sigma
    return (image + noise).clamp(0, 1)
```

**实验流程**：
1. 用真实数据训 FlowTDP-w-traj 和 FlowTDP-img-only（来自实验 1）
2. 评测时给图像加 σ ∈ {0, 0.08, 0.16, 0.32, 0.64} 的高斯噪声
3. 记录每个 σ 下的成功率
4. 画 relative success rate 曲线（论文 Fig 3 风格）

**预期**：
- σ=0.32 时，With-Voxel 保持 ≥ 15% 的成功率
- Without-Voxel 在 σ=0.32 时降到 5% 以下

---

### 5.7 🟢 E2: 子任务消融

**目标**：验证设计文档 §2.1 的"生成-执行-验证-重规划"承诺

#### 详细规格

**两组实验**：
- A: 开环（一次性生成所有子任务的 voxel trajectory）
- B: 闭环（每 5 步状态检查，重规划）

**实施**：
- 修改 `scripts/test_closed_loop.py` 让 `ClosedLoopController` 支持 `replanning: bool` 参数
- 当 `replanning=False` 时，只生成一次 voxel，全过程不变

**指标**：30 episodes 平均成功率

**预期**：
- 闭环 ≥ 开环 + 10pp（设计文档承诺 +15pp）

---

### 5.8 🟢 P3: 本地 VLM 切换

**目标**：让 README 与设计文档承诺"完全本地离线"对齐

#### 详细规格

**步骤**：
1. 装 LLaVA-1.5-7B (4bit 量化) 到本地
2. 实现 `LocalLLaVAWrapper::generate_text` 真正的推理循环
3. 跑 `VLMWrapper(backend="local")` 工作

**依赖**：transformers, bitsandbytes（已在 requirements.txt）
**显存**：~5GB (4bit 量化后)

**验收**：
- `VLMWrapper(backend="local")` 返回本地模型实例
- `vlm.decompose_task(...)` 不需要 OPENAI_API_KEY

---

## 6. Phase E（高级功能）详细规格

### 6.1 训练 / 推理优化

| 任务 | 规格 | 工作量 |
|---|---|---|
| 混合精度训练 (fp16) | 在 RTX 3090/4090 上加速 | 0.5 天 |
| torch.compile | 加速推理 ~20% | 0.5 天 |
| TensorRT 导出 | 推理 < 5ms | 2-3 天 |
| 多 GPU 训练 | torchrun + DDP | 1 天 |

### 6.2 真实机器人部署

| 任务 | 规格 | 工作量 |
|---|---|---|
| Franka + RealSense 接口 | 复用 ManiSkill3 真实接口 | 2 天 |
| 真实环境 vs 仿真 sim-to-real gap 评测 | 论文 §V | 2 天 |
| 6-DOF SpaceMouse 示教 | 复用 RLBench 工具 | 1 天 |

### 6.3 Colosseum 变异

**目标**：复现论文 Fig 4 的 4 维变异（背景/物体纹理/对象大小）

**问题**：ManiSkill3 没有 Colosseum 集成

**降级方案**：
- 用 ManiSkill 自带的 `reconfigure` API 修改场景
- 或用 image augmentation (背景色变更、纹理贴图替换)

**指标**：4 种变异下的成功率

---

## 7. 实验矩阵

每个实验产出的标准交付物：

| 实验 | 输入 | 输出文件 | 图表 | 报告章节 |
|---|---|---|---|---|
| 1: 轨迹条件消融 | 演示数据 | `exp1_*.json` | `exp1_*_summary.png` | "Voxel Condition Matters" |
| 2: FM 步数 | 训好的模型 | `exp2_*.json` | `exp2_fm_steps.png` | "Single-step Inference" |
| 3: VLM 质量 | 20 张图 | `exp3_*.json` | `exp3_examples/` | "VLM Trajectory Quality" |
| 4: DDPM vs FM | 同训练数据 | `exp4_*.json` | `exp4_ddpm_fm.png` | "Flow Matching Advantage" |
| 5: 噪声鲁棒性 | 训好模型 | `exp5_*.json` | `exp5_noise.png` | "Noise Robustness" |
| 6: 闭环 vs 开环 | 训好模型 | `exp6_*.json` | `exp6_closed_loop.png` | "Closed-loop Planning" |
| 7: 真实环境 | 真 ManiSkill | `exp7_*.json` | `exp7_real_env.png` | "Real Environment Results" |

---

## 8. 数据资产清单

### 8.1 当前数据

| 路径 | 类型 | 大小 | 状态 |
|---|---|---|---|
| `data/__init__.py` | 代码 | 181 B | 提交 |
| `data/maniskill_dataset.py` | 代码 | 11 KB | 提交 |
| `paper_text.txt` | 文本分析 | ~30 KB | gitignored（临时） |
| 合成 `.h5` demos | 演示数据 | 生成在 /tmp | 不提交 |

### 8.2 需要下载的数据

```bash
# ManiSkill 官方演示（推荐）
python -m mani_skill.utils.download_demo -t PickCube-v0 -o data/maniskill/PickCube-v0/
python -m mani_skill.utils.download_demo -t StackCube-v0 -o data/maniskill/StackCube-v0/

# RLBench 18 任务（备选）
huggingface-cli download PhilJd/rlbench-18-tasks --repo-type dataset -o data/rlbench/

# DROID / Open X-Embodiment（更大规模，备选）
huggingface-cli download google-deepmind/open_x_embodiment --repo-type dataset
```

### 8.3 数据格式约定

```
data/maniskill/<task_name>/trajectory.h5
  obs/
    agent/qpos:           (T, 9) float32     # arm(7) + gripper(2)
    sensor_data/<cam>/rgb: (T, H, W, 3) uint8
    sensor_data/<cam>/depth: (T, H, W) float32
    sensor_param/<cam>/intrinsic_cv: (3, 3) float64
    sensor_param/<cam>/cam2world_gl: (4, 4) float64
  actions:                (T, action_dim) float32
```

---

## 9. 代码架构约束

### 9.1 公共 API 不可变

```python
# 这些 import 必须保持工作
from models import VLMWrapper, LightVoxelEncoder, FlowTDP
from utils import (
    PromptTemplate, project_to_voxel_grid, draw_voxel_grid_overlay,
    extract_voxel_trajectory, fk_panda_batch, segment_subtasks,
)
from data import ManiSkillDemoDataset, DatasetConfig
```

### 9.2 设计决策记录（不要轻易改动）

| 决策 | 原因 |
|---|---|
| `grid_size = 6` (硬编码) | 论文 §III.B 明确说明；论文测试了 8×8×8 会让 VLM 更难 |
| `workspace_bounds = (-0.3, -0.3, 0) ~ (0.3, 0.3, 0.5)` | Franka 桌面（论文隐含） |
| `interp_step_m = 0.02` | 2cm 步长；稀疏但能捕捉 sub-task 内移动 |
| **不加走廊**（radius=0） | 跟论文一致；VLM 输出更易处理 |
| 体素表示 = sparse sequence grid | 论文同款 |
| 单视图（base_camera） | 论文 §IV.A 单任务设置 |
| FlowTDP use_voxel=True 默认 | TDP 模式；use_voxel=False 是消融用 |
| 体素输入维度 = (B, 6, 6, 6) float | LightVoxelEncoder 期望 |

### 9.3 性能基线

| 组件 | 大小 | 单步推理时间（CPU） |
|---|---|---|
| LightVoxelEncoder | ~44K 参数 | ~5ms |
| FlowTDP (含图像编码) | ~13M 参数 | ~7ms (CPU, batch=1) |
| VLM (OpenAI API) | — | ~500-2000ms |
| VLM (LocalLLaVA) | ~7B 参数 (4bit) | ~150ms |

---

## 10. 风险登记与降级方案

| 风险 | 等级 | 降级方案 |
|---|---|---|
| RTX 5060 GPU 兼容问题 | 已解决 | 装 PyTorch nightly cu128 |
| ManiSkill `Panda()` 构造失败 | 中 | 自动降级到 fallback FK |
| OpenAI API 不稳定 / 限流 | 中 | 加重试；切到本地 LLaVA |
| 合成数据实验数字无意义 | 高 | 接真实 ManiSkill 数据 |
| ManiSkill env rollout 没接 | 高 | 写 `ManiSkillAdapter` + 评测脚本 |
| DDPM baseline 训得很慢 | 中 | 用更少步数（5000）+ 同样 batch |
| 真实机器人无硬件 | 高 | 仅跑仿真，跳过论文 §V |
| Colosseum 集成复杂 | 中 | ManiSkill reconfigure API + image augmentation |

---

## 11. 里程碑时间表

### Week 1 (D2 + D3)

| Day | 任务 | 产出 |
|---|---|---|
| 1 | D2 实验脚本 + 跑 | `exp2_fm_steps.{json,png}` |
| 2 | D3 实验脚本 + 跑（需要 OPENAI_API_KEY） | `exp3_vlm_iou.{json,png}` |
| 3 | README 更新实验结果 + GitHub push | commit |
| 4-5 | 接真实 ManiSkill 数据 + ManiSkillAdapter | `data/maniskill/*.h5` + 适配器 |

### Week 2 (DDPM + 鲁棒性)

| Day | 任务 | 产出 |
|---|---|---|
| 1 | DDPM baseline + 实验 4 | `exp4_ddpm_fm.{json,png}` |
| 2 | 噪声鲁棒性实验 | `exp5_noise.{json,png}` |
| 3-4 | 闭环 vs 开环实验 | `exp6_closed_loop.{json,png}` |
| 5 | 真实环境 rollout 评测 | `exp7_real_env.{json,png}` |

### Week 3 (高级功能)

| Day | 任务 | 产出 |
|---|---|---|
| 1-2 | 本地 VLM (LLaVA-1.5-7B 4bit) | `LocalLLaVAWrapper` 推理实现 |
| 3 | Colosseum 等价变异 | `ColosseumVariations` 工具 |
| 4-5 | 论文复现报告（Markdown） | `PAPER_REPRODUCTION.md` |

### Week 4+ (生产化)

| Day | 任务 |
|---|---|
| 1-3 | TensorRT 导出 + 推理加速 |
| 4-5 | 多 GPU 训练支持 |

---

## 附录 A：关键文件路径速查

```
# 已实现 (commit 82b47cc)
models/__init__.py                          VLMWrapper factory 导出
models/flow_tdp.py                          Euler 积分 + use_voxel
models/light_voxel_encoder.py               深度可分离 3D CNN
models/vlm_wrapper.py                       OpenAIVLMWrapper + LocalLLaVAWrapper
utils/__init__.py                           所有工具 export
utils/voxel_trajectory.py                   6×6×6 体素投影
utils/visual_prompt.py                      grid overlay + 3D
utils/trajectory_extraction.py              演示轨迹 → 体素
utils/franka_fk.py                          ManiSkill + fallback FK
utils/subtask.py                            sub-task 切分
utils/camera_params.py                      相机参数加载
utils/prompt_templates.py                   4 类 VLM prompts
data/__init__.py                            data package
data/maniskill_dataset.py                   PyTorch Dataset
scripts/test_closed_loop.py                 闭环控制器
scripts/train_flow_tdp.py                   训练框架
scripts/exp1_traj_ablation.py               实验 1 脚本
scripts/plot_exp1.py                        实验 1 图表
tests/                                      41 测试

# 待实现
scripts/exp2_fm_steps.py                    [TODO]
scripts/exp3_vlm_quality.py                 [TODO]
scripts/exp4_ddpm_baseline.py               [TODO]
scripts/exp5_noise_robustness.py            [TODO]
scripts/exp6_closed_loop.py                 [TODO]
scripts/evaluate_real_env.py                [TODO]
models/ddpm_policy.py                       [TODO]
utils/noise_augmentation.py                 [TODO]
utils/colosseum_variations.py               [TODO]
```

---

## 附录 B：核心论文对照表

| 论文内容 | 实现位置 | 状态 |
|---|---|---|
| §III.B Task Decomposition | `utils/prompt_templates.py::TASK_DECOMPOSITION` | ✅ |
| §III.B Voxel-based Spatial Trajectory | `utils/trajectory_extraction.py` | ✅ |
| §III.B Mask-based Visual Prompting | `utils/visual_prompt.py` | ⚠️ 简化版（无 top-down 投影） |
| §III.B Trajectory-conditioned 3D CNN | `models/light_voxel_encoder.py` | ✅ |
| §III.C DDPM 训练目标 | （已替换为 CFM） | ✅ Flow Matching |
| §III.C Action 预测 | `models/flow_tdp.py` | ✅ |
| §IV.A Single-view setup | 配置 + 数据加载 | ⚠️ 待真实数据 |
| §IV.A T=12 N=8 | `config/train.yaml` | ✅ |
| §IV.B Multi-view (长时序) | `scripts/maniskill_environment.py` | ⚠️ 框架在 |
| §IV.B 长时序堆叠 | `data/maniskill_dataset.py` | ⚠️ sub-task 切分在 |
| §IV.C 噪声鲁棒性 | `utils/noise_augmentation.py` | ❌ 待实现 |
| §IV.C Colosseum 变异 | `utils/colosseum_variations.py` | ❌ 待实现 |
| §V 真实世界 | — | ❌ 无硬件 |
| §VI 结论 | `ROADMAP.md` | ✅ 本文 |

---

**下一步行动**：按优先级从 D2 (FM 步数扫描) 开始，每天完成一个实验，更新本文档。