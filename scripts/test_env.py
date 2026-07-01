"""Test ManiSkill env creation."""
import os
import sys
import time
import warnings
import traceback

warnings.filterwarnings('ignore')

def main():
    t0 = time.time()
    print(f"[{time.time()-t0:.1f}s] start", flush=True)

    try:
        import gymnasium as gym
        print(f"[{time.time()-t0:.1f}s] gym imported", flush=True)
        import mani_skill.envs
        print(f"[{time.time()-t0:.1f}s] mani_skill envs imported", flush=True)

        env = gym.make(
            'PickCube-v1',
            obs_mode='rgbd',
            control_mode='pd_joint_pos',
            render_mode='rgb_array',
            sim_backend='cpu',
        )
        print(f"[{time.time()-t0:.1f}s] env created", flush=True)

        obs, info = env.reset(seed=0)
        print(f"[{time.time()-t0:.1f}s] reset OK", flush=True)
        if isinstance(obs, dict):
            for k, v in obs.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape={v.shape}", flush=True)

        env.close()
        print(f"[{time.time()-t0:.1f}s] DONE", flush=True)

    except Exception as e:
        print(f"[{time.time()-t0:.1f}s] EXC: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    main()