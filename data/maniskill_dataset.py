"""
ManiSkill Dataset
从 ManiSkill3 `.h5` 演示文件加载 (image, voxel_trajectory, action) 三元组。

Pipeline:
    1. 读取 .h5 demo: qpos / actions / rgb / camera_params
    2. 用 ManiSkill 内置 FK 算 EE 位置 (qpos → ee_positions in world)
    3. 按夹爪状态切 sub-task
    4. 对每个 sub-task, 用 extract_voxel_trajectory 生成 6×6×6 体素
    5. 对每个 timestep t: image[t], voxel_traj[所属 sub-task], action[t:t+T]

返回值 (per item):
    {
        "image": tensor (3, H, W),  float, [0, 1]
        "voxel_trajectory": tensor (6, 6, 6),  long (cell label) 或 float (binary)
        "action": tensor (T, action_dim),  float
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    """Dataset 配置."""
    h5_paths: List[str]                              # demo 文件列表
    camera_name: str = "base_camera"                 # 相机名
    image_size: Tuple[int, int] = (256, 256)         # 训练时图像 resize
    prediction_horizon: int = 12                     # T (动作预测窗口)
    execution_horizon: int = 8                       # N (执行窗口, 仅记录)
    workspace_bounds: Tuple[float, ...] = (-0.3, -0.3, 0.0, 0.3, 0.3, 0.5)
    grid_size: int = 6
    interp_step_m: float = 0.02                      # 体素提取步长 (2cm)
    gripper_threshold: float = 0.04                  # sub-task 切分阈值
    min_subtask_length: int = 5
    voxel_repr: str = "sequence"                     # "sequence" | "binary"
    use_fk: str = "auto"                             # "maniskill" | "fallback" | "auto"


def _resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize (H, W, 3) uint8 → (target_h, target_w, 3) uint8."""
    try:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img)
        pil = pil.resize((target_size[1], target_size[0]), PILImage.BILINEAR)
        return np.array(pil, dtype=np.uint8)
    except ImportError:
        # fallback: simple nearest-neighbor
        h, w = img.shape[:2]
        th, tw = target_size
        y_idx = (np.arange(th) * h / th).astype(int)
        x_idx = (np.arange(tw) * w / tw).astype(int)
        return img[y_idx][:, x_idx]


class ManiSkillDemoDataset(Dataset):
    """
    从 ManiSkill `.h5` 演示加载训练样本。

    数据流 (per demo):
      .h5 → qpos, actions, rgb
        → FK: qpos → ee_positions (T, 3)
        → sub-task segmentation
        → for each sub-task: ee_poses[st.start:st.end] → voxel_traj (6, 6, 6)
        → for each timestep t: image[t], voxel_traj_of_its_subtask, actions[t:t+T]

    Returns (per item):
      image: (3, H, W) float [0, 1]
      voxel_trajectory: (6, 6, 6) long (sequence) 或 float (binary)
      action: (T, action_dim) float
    """

    def __init__(self, config: DatasetConfig):
        self.cfg = config

        # 预加载所有 demo 到内存 (h5 文件不大)
        self.episodes: List[dict] = []
        self.index_map: List[Tuple[int, int]] = []  # (episode_idx, timestep)

        self._load_all_episodes()

    def _load_all_episodes(self):
        """加载所有 .h5 文件, 预计算体素轨迹."""
        from utils.franka_fk import fk_panda_batch
        from utils.subtask import segment_subtasks
        from utils.trajectory_extraction import extract_voxel_trajectory

        for path in self.cfg.h5_paths:
            try:
                ep = self._load_single_episode(path, fk_panda_batch, segment_subtasks, extract_voxel_trajectory)
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}")
                continue
            if ep is None:
                continue

            ep_idx = len(self.episodes)
            self.episodes.append(ep)

            # 每个 timestep (减去 T 的边界) 都是一个样本
            T = ep["actions"].shape[0]
            usable = max(0, T - self.cfg.prediction_horizon + 1)
            for t in range(usable):
                self.index_map.append((ep_idx, t))

    def _load_single_episode(
        self,
        path: str,
        fk_fn,
        seg_fn,
        extract_voxel_fn,
    ) -> Optional[dict]:
        with h5py.File(path, "r") as f:
            # 读取关键字段
            try:
                qpos = np.array(f["obs/agent/qpos"])              # (T, 9)
                actions = np.array(f["actions"])                   # (T, action_dim)
                rgb = np.array(f[f"obs/sensor_data/{self.cfg.camera_name}/rgb"])  # (T, H, W, 3)
            except KeyError as e:
                print(f"Warning: missing key in {path}: {e}")
                return None

        # FK: qpos → EE positions
        try:
            ee_positions = fk_fn(qpos)  # (T, 3)
        except ImportError as e:
            print(f"Warning: FK failed for {path}: {e}")
            return None

        # Sub-task 切分
        subtasks = seg_fn(qpos, threshold=self.cfg.gripper_threshold,
                          min_length=self.cfg.min_subtask_length)

        # 对每个 sub-task 算体素轨迹
        voxel_trajs_per_t = np.zeros((qpos.shape[0], self.cfg.grid_size,
                                       self.cfg.grid_size, self.cfg.grid_size),
                                      dtype=np.int64)
        for st in subtasks:
            sub_ee = ee_positions[st.start:st.end]
            if len(sub_ee) < 2:
                continue
            vt = extract_voxel_fn(
                sub_ee,
                workspace_bounds=self.cfg.workspace_bounds,
                grid_size=self.cfg.grid_size,
                interpolate=True,
                interp_step_m=self.cfg.interp_step_m,
                return_sequence=True,
            ).numpy()
            # 整个 sub-task 内所有 timestep 共享这个 vt
            voxel_trajs_per_t[st.start:st.end] = vt

        return {
            "rgb": rgb,
            "actions": actions,
            "voxel_trajs": voxel_trajs_per_t,
            "qpos": qpos,
            "ee_positions": ee_positions,
        }

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self.index_map[idx]
        ep = self.episodes[ep_idx]

        # 图像
        img = ep["rgb"][t]  # (H, W, 3) uint8
        img = _resize_image(img, self.cfg.image_size)  # (h, w, 3) uint8
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, h, w)

        # 体素轨迹
        vt = ep["voxel_trajs"][t]  # (6, 6, 6) int
        if self.cfg.voxel_repr == "sequence":
            vt_t = torch.from_numpy(vt).long()
        else:  # binary
            vt_t = (torch.from_numpy(vt) > 0).float()

        # 动作窗口
        T = self.cfg.prediction_horizon
        action_window = ep["actions"][t: t + T]  # (T, action_dim)
        if action_window.shape[0] < T:
            # 边界情况: 末尾 pad 0
            pad = np.zeros((T - action_window.shape[0], action_window.shape[1]))
            action_window = np.concatenate([action_window, pad], axis=0)
        action_t = torch.from_numpy(action_window).float()

        return {
            "image": img_t,
            "voxel_trajectory": vt_t,
            "action": action_t,
        }


# ---------------------------------------------------------------------------
# 合成 demo 生成 (ManiSkill 不可用时, 用假数据测试 dataloader)
# ---------------------------------------------------------------------------


def make_synthetic_h5(path: str, T: int = 60, image_hw: Tuple[int, int] = (200, 300),
                       action_dim: int = 8) -> None:
    """
    生成一个假的 ManiSkill .h5 demo (用于 dataloader 单元测试).

    模拟一次 PickCube-v0 风格:
      - 0~9  张开夹爪, EE 在 home 位置
      - 10~29 闭合夹爪 (抓取), EE 移动到 cube
      - 30~49 张开夹爪, EE 移动到目标
      - 50~59 张开, EE 回到 home
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    qpos = np.zeros((T, 9), dtype=np.float32)
    # arm joints: 模拟一个简单的"伸出→收回"轨迹
    for i in range(T):
        if i < 30:
            qpos[i, :7] = np.linspace(0, -0.3, 30)[min(i, 29)]
        else:
            qpos[i, :7] = np.linspace(-0.3, 0, 30)[min(i - 30, 29)]
    # gripper
    qpos[:10, -2:] = 0.08   # open
    qpos[10:30, -2:] = 0.0   # close
    qpos[30:, -2:] = 0.08    # open

    # actions
    actions = np.random.randn(T, action_dim).astype(np.float32) * 0.05

    # rgb images
    h, w = image_hw
    rgb = np.random.randint(50, 200, (T, h, w, 3), dtype=np.uint8)

    with h5py.File(path, "w") as f:
        f.create_dataset("obs/agent/qpos", data=qpos)
        f.create_dataset("actions", data=actions)
        f.create_dataset("obs/sensor_data/base_camera/rgb", data=rgb)
        # 简单的相机参数
        param = f.create_group("obs/sensor_param/base_camera")
        K = np.array([[300., 0, w/2], [0, 300., h/2], [0, 0, 1]], dtype=np.float64)
        param.create_dataset("intrinsic_cv", data=K)
        param.create_dataset("cam2world_gl", data=np.eye(4))


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def self_test():
    print("ManiSkillDemoDataset smoke checks:")

    import tempfile
    tmpdir = tempfile.mkdtemp()

    # 生成 3 个假 demo
    paths = []
    for i in range(3):
        p = str(Path(tmpdir) / f"demo_{i}.h5")
        make_synthetic_h5(p, T=60)
        paths.append(p)

    cfg = DatasetConfig(
        h5_paths=paths,
        image_size=(64, 64),  # 缩小加快测试
        prediction_horizon=12,
        workspace_bounds=(-0.3, -0.3, 0.0, 0.3, 0.3, 0.5),
        voxel_repr="sequence",
        use_fk="fallback",  # 强制用 fallback FK (不依赖 ManiSkill)
    )

    ds = ManiSkillDemoDataset(cfg)
    print(f"  Dataset size: {len(ds)} (expected ~3 demos * (60-12+1) = 147)")
    assert len(ds) > 0, "Dataset is empty"

    # 取一个样本
    sample = ds[0]
    print(f"  Sample shapes: image={tuple(sample['image'].shape)}, "
          f"voxel={tuple(sample['voxel_trajectory'].shape)}, "
          f"action={tuple(sample['action'].shape)}")
    assert sample["image"].shape == (3, 64, 64)
    assert sample["voxel_trajectory"].shape == (6, 6, 6)
    assert sample["action"].shape == (12, 8)

    # 体素应该有非零 cell (因为 EE 真的移动了)
    vt = sample["voxel_trajectory"]
    print(f"  Non-zero voxel cells: {int((vt > 0).sum())}")
    assert (vt > 0).sum() > 0, "voxel trajectory should have occupied cells"

    # 二进制模式
    cfg.voxel_repr = "binary"
    ds_bin = ManiSkillDemoDataset(cfg)
    sample_bin = ds_bin[0]
    print(f"  Binary voxel dtype: {sample_bin['voxel_trajectory'].dtype}, "
          f"unique: {sorted(set(sample_bin['voxel_trajectory'].flatten().tolist()))}")
    assert set(sample_bin["voxel_trajectory"].flatten().tolist()).issubset({0.0, 1.0})

    print("  All smoke checks passed.")


if __name__ == "__main__":
    self_test()