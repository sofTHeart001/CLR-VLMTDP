"""Quick FK test"""
import sys, time
sys.path.insert(0, '.')
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
import h5py
import numpy as np

t0 = time.time()
print(f"[{time.time()-t0:.1f}s] imports", flush=True)

# Read qpos
path = "D:/Desktop/github_project/CLR-VLMTDP/data/maniskill/PickCube-v1/motionplanning/trajectory.h5"
with h5py.File(path, "r") as f:
    print(f"[{time.time()-t0:.1f}s] opened h5", flush=True)
    qpos_flat = f["traj_0/env_states/articulations/panda"][:5]
    print(f"[{time.time()-t0:.1f}s] loaded qpos: shape={qpos_flat.shape}", flush=True)

# 试试 fallback FK - 直接调用, 不经过 franka_fk.py 包装
import importlib.util
print(f"[{time.time()-t0:.1f}s] before import franka_fk module", flush=True)

# Test the inner DH-based FK directly
import utils.franka_fk as ffk
print(f"[{time.time()-t0:.1f}s] imported franka_fk module", flush=True)

# Inspect
print(f"  _MANISKILL_AGENT: {ffk._MANISKILL_AGENT}", flush=True)
print(f"  PANDA_DH_FALLBACK shape: {ffk.PANDA_DH_FALLBACK.shape}", flush=True)

# 直接调用 fallback FK
qpos_9d = qpos_flat[:5, 13:22].astype(np.float64)  # 9 维
print(f"[{time.time()-t0:.1f}s] 9-D qpos shape: {qpos_9d.shape}", flush=True)
print(f"  qpos_9d values:\n{qpos_9d}", flush=True)

positions = ffk.fk_panda_batch_fallback(qpos_9d)
print(f"[{time.time()-t0:.1f}s] FK done: shape={positions.shape}", flush=True)
print(f"  positions:\n{positions}", flush=True)
print(f"[{time.time()-t0:.1f}s] DONE", flush=True)