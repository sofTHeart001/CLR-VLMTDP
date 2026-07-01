"""
Camera Parameters Loader
从 ManiSkill `.h5` 演示文件加载相机内外参；缺失时用 Franka 桌面默认。

ManiSkill `.h5` 文件里通常包含:
  - obs/sensor_param/<cam_name>/intrinsic_cv: (3, 3) numpy array
  - obs/sensor_param/<cam_name>/cam2world_gl: (4, 4) numpy array (外参, world→camera 还是 camera→world 看版本)
  - obs/sensor_param/<cam_name>/extrinsic_cv: 类似

我们关注的是:
  K = intrinsic (3, 3)
  T_world_cam = camera 在世界坐标系下的 pose (用于把世界点反投影到图像)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


# 默认参数（franka 桌面，800×600 图像，相机在 (0, 0, 1) 朝下看原点）
DEFAULT_INTRINSICS_800x600 = np.array([
    [400.0,   0.0, 400.0],
    [  0.0, 400.0, 300.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)


def load_camera_params_from_h5(
    h5_path: str,
    camera_name: str = "base_camera",
) -> Dict[str, np.ndarray]:
    """
    从 ManiSkill `.h5` demo 读取相机参数。

    Args:
        h5_path: demo 文件路径
        camera_name: 相机名（默认 base_camera）

    Returns:
        dict with keys:
          - intrinsic: (3, 3) 内参 K
          - cam2world: (4, 4) 相机到世界变换 (P_world = T_cam2world @ P_cam)
          - image_size: (W, H)
          - source: "h5" 或 "default"
    """
    out = {
        "intrinsic": DEFAULT_INTRINSICS_800x600.copy(),
        "cam2world": np.eye(4, dtype=np.float64),
        "image_size": (800, 600),
        "source": "default",
    }

    try:
        with h5py.File(h5_path, "r") as f:
            param_root = f"obs/sensor_param/{camera_name}"
            if param_root not in f:
                return out

            grp = f[param_root]

            # 内参
            for key in ("intrinsic_cv", "intrinsic", "K"):
                if key in grp:
                    K = np.array(grp[key])
                    if K.shape == (3, 3):
                        out["intrinsic"] = K.astype(np.float64)
                        out["source"] = "h5"
                    break

            # 外参
            for key in ("cam2world_gl", "cam2world", "extrinsic_cv"):
                if key in grp:
                    T = np.array(grp[key])
                    if T.shape == (4, 4):
                        out["cam2world"] = T.astype(np.float64)
                        out["source"] = "h5"
                    break

            # 图像尺寸 (从 rgb 推断)
            rgb_key = f"obs/sensor_data/{camera_name}/rgb"
            if rgb_key in f:
                rgb_shape = f[rgb_key].shape  # (T, H, W, 3)
                if len(rgb_shape) >= 3:
                    out["image_size"] = (rgb_shape[2], rgb_shape[1])

    except (OSError, KeyError) as e:
        # 文件不存在或结构不匹配 → 返回默认
        pass

    return out


def get_workspace_default_intrinsics(image_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
    """获取 Franka 桌面默认内参 (按 image_size 缩放)."""
    from utils.voxel_trajectory import get_default_intrinsics
    return get_default_intrinsics(image_size)


def get_workspace_default_extrinsics() -> np.ndarray:
    """获取 Franka 桌面默认外参 (相机在 (0, 0, 1) 朝下看)."""
    from utils.voxel_trajectory import get_default_extrinsics
    return get_default_extrinsics()


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------


def self_test():
    print("Camera params smoke checks:")

    # 默认参数
    K = get_workspace_default_intrinsics((800, 600))
    T = get_workspace_default_extrinsics()
    print(f"  Default K shape={K.shape}, fx={K[0,0]:.0f}")
    print(f"  Default T shape={T.shape}, camera_z={T[2,3]:.1f}")

    # 不存在的 h5 → 默认
    out = load_camera_params_from_h5("nonexistent.h5", "base_camera")
    print(f"  Missing h5 -> source={out['source']}, K shape={out['intrinsic'].shape}")
    assert out["source"] == "default"

    # 临时造一个 h5 测试
    import tempfile
    tmpdir = tempfile.mkdtemp()
    test_path = str(Path(tmpdir) / "test_cam.h5")
    with h5py.File(test_path, "w") as f:
        param_grp = f.create_group("obs/sensor_param/base_camera")
        param_grp.create_dataset("intrinsic_cv", data=np.eye(3) * 500)
        param_grp.create_dataset("cam2world_gl", data=np.eye(4))
        rgb_grp = f.create_group("obs/sensor_data/base_camera")
        rgb_grp.create_dataset("rgb", data=np.zeros((5, 100, 200, 3), dtype=np.uint8))

    out = load_camera_params_from_h5(test_path, "base_camera")
    print(f"  Synthetic h5 -> source={out['source']}, K[0,0]={out['intrinsic'][0,0]:.0f}")
    print(f"  Synthetic h5 -> image_size={out['image_size']}")
    assert out["source"] == "h5"
    assert out["intrinsic"][0, 0] == 500
    assert out["image_size"] == (200, 100)

    print("  All smoke checks passed.")


if __name__ == "__main__":
    self_test()