"""
ManiSkill3 安装和测试脚本
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并显示输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def test_maniskill3():
    """测试ManiSkill3安装"""
    print("\n" + "="*60)
    print("Testing ManiSkill3 Installation")
    print("="*60 + "\n")

    try:
        # 测试1: 导入测试
        print("Test 1: Importing ManiSkill3...")
        from mani_skill2.envs import (
            PickCubeEnv,
            StackCubeEnv,
            OpenCabinetDrawerEnv,
            CloseCabinetDrawerEnv
        )
        print("  ✓ Import successful\n")

        # 测试2: 创建环境
        print("Test 2: Creating environment...")
        import gymnasium as gym
        from mani_skill2.utils.registration import REGISTERED_ENVS

        print(f"  Available environments: {len(REGISTERED_ENVS)}")
        for env_name in sorted(REGISTERED_ENVS.keys())[:10]:
            print(f"    - {env_name}")
        print()

        # 测试3: 简单环境测试
        print("Test 3: Testing simple environment (PickCube)...")
        env = gym.make("PickCube-v0", num_envs=1, obs_mode="rgbd")
        print("  ✓ Environment created")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")

        # 测试4: 重置环境
        print("\nTest 4: Resetting environment...")
        obs, info = env.reset()
        print(f"  ✓ Environment reset")
        print(f"  Observation shape: {obs['image'].shape}")

        # 测试5: 执行随机动作
        print("\nTest 5: Executing random action...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Action executed")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")

        # 关闭环境
        env.close()
        print("\n  ✓ Environment closed")

        print("\n" + "="*60)
        print("✅ All ManiSkill3 tests PASSED!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          ManiSkill3 Installation & Test Script              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 检查Python版本
    print("1. Checking Python version...")
    python_version = sys.version
    print(f"   Python: {python_version}")

    if sys.version_info < (3, 8):
        print("   ✗ Python 3.8+ required")
        return False
    print("   ✓ Python version OK\n")

    # 检查CUDA
    print("2. Checking CUDA availability...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   PyTorch CUDA: {cuda_available}")
        if cuda_available:
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print()
    except ImportError:
        print("   PyTorch not installed yet")
        print()

    # 测试ManiSkill3
    success = test_maniskill3()

    if success:
        print("""
╔══════════════════════════════════════════════════════════════╗
║               Installation Successful!                        ║
╚══════════════════════════════════════════════════════════════╝

Next steps:
  1. Generate demo data:
     python scripts/generate_maniskill_demos.py --tasks PickCube --num_demos 10

  2. Test closed-loop:
     python scripts/test_closed_loop_maniskill.py --task PickCube

  3. Train model:
     python scripts/train_flow_tdp.py --config config/train.yaml

Note: ManiSkill3 uses GPU acceleration for fast simulation.
      Make sure you have CUDA drivers installed.
        """)
    else:
        print("""
╔══════════════════════════════════════════════════════════════╗
║              Installation Failed!                             ║
╚══════════════════════════════════════════════════════════════╝

Troubleshooting:
  1. Make sure you activated the conda environment:
     conda activate clr_vlmtdp_maniskill3

  2. Reinstall dependencies:
     conda env remove -n clr_vlmtdp_maniskill3
     conda env create -f environment_maniskill3.yml

  3. Check CUDA installation:
     nvidia-smi

  4. Install Visual C++ (Windows):
     https://visualstudio.microsoft.com/visual-cpp-build-tools/
        """)

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)