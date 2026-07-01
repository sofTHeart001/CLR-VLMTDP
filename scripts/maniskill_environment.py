"""
ManiSkill3 Environment Wrapper
ManiSkill3仿真环境包装类（Windows支持）
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from mani_skill2.utils.registration import REGISTERED_ENVS


class ManiSkillConfig:
    """ManiSkill环境配置"""

    def __init__(
        self,
        obs_mode: str = "rgbd",  # rgb, depth, rgbd
        num_envs: int = 1,
        control_mode: str = "pd_ee_pos",  # pd_ee_pos, pd_joint_pos
        reward_mode: str = "dense",
        horizon: Optional[int] = None
    ):
        self.obs_mode = obs_mode
        self.num_envs = num_envs
        self.control_mode = control_mode
        self.reward_mode = reward_mode
        self.horizon = horizon


class ManiSkillEnvironment:
    """
    ManiSkill3环境包装类

    支持的功能:
    - GPU加速仿真
    - 1000+操作任务
    - 多相机观测
    - Windows原生支持
    """

    def __init__(
        self,
        config: ManiSkillConfig,
        task_name: str = "PickCube-v0"
    ):
        """
        初始化ManiSkill3环境

        Args:
            config: 环境配置
            task_name: 任务名称（需要加-v0等版本号）
        """
        self.config = config
        self.task_name = task_name

        # 创建环境
        self.env = gym.make(
            task_name,
            num_envs=config.num_envs,
            obs_mode=config.obs_mode,
            control_mode=config.control_mode,
            reward_mode=config.reward_mode,
            horizon=config.horizon
        )

        # 重置环境
        self._reset()

    def _reset(self) -> Dict:
        """重置环境"""
        obs, info = self.env.reset()
        self.current_obs = obs
        return self._process_observation(obs)

    def _process_observation(self, obs: Dict) -> Dict:
        """
        处理观测，转换为标准格式

        ManiSkill3观测格式:
        {
            "image": (num_envs, H, W, 3),
            "state": (num_envs, state_dim),
            ...
        }
        """
        result = {}

        # 图像观测
        if "image" in obs:
            # ManiSkill3返回(N, H, W, C)，转换为(C, H, W)
            img = obs["image"][0]  # 取第一个环境
            result["rgb"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 深度观测
        if "depth" in obs:
            depth = obs["depth"][0]
            result["depth"] = torch.from_numpy(depth).float()

        # 状态观测（关节位置等）
        if "state" in obs:
            state = obs["state"][0]
            result["state"] = torch.from_numpy(state).float()

        # 夹爪状态
        if "gripper" in obs:
            result["gripper_open"] = float(obs["gripper"][0] > 0.5)

        return result

    def reset(self) -> Dict:
        """
        重置环境

        Returns:
            初始观测字典
        """
        return self._reset()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作向量

        Returns:
            (obs, reward, done, info)
        """
        # 转换动作格式
        if isinstance(action, torch.Tensor):
            action = action.numpy()

        # 如果是单个环境的动作，扩展为批次
        if action.ndim == 1:
            action = action[np.newaxis, :]

        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated[0] or truncated[0]
        reward_val = float(reward[0])

        # 处理观测
        processed_obs = self._process_observation(obs)

        # 构建info
        info = {
            "task_name": self.task_name,
            "success": info.get("success", [False])[0]
        }

        return processed_obs, reward_val, done, info

    def get_task_description(self) -> str:
        """获取当前任务的描述"""
        return self.task_name

    def close(self):
        """关闭环境"""
        self.env.close()

    def __del__(self):
        """析构函数，确保环境关闭"""
        try:
            self.close()
        except:
            pass


def create_maniskill_env(
    task_name: str = "PickCube-v0",
    obs_mode: str = "rgbd",
    num_envs: int = 1,
    control_mode: str = "pd_ee_pos"
) -> ManiSkillEnvironment:
    """
    工厂函数：创建ManiSkill3环境

    Args:
        task_name: 任务名称
        obs_mode: 观测模式 (rgb, depth, rgbd, pointcloud)
        num_envs: 并行环境数量
        control_mode: 控制模式

    Returns:
        ManiSkillEnvironment实例
    """
    config = ManiSkillConfig(
        obs_mode=obs_mode,
        num_envs=num_envs,
        control_mode=control_mode
    )

    return ManiSkillEnvironment(config, task_name)


# 测试代码
if __name__ == "__main__":
    print("Testing ManiSkill3 environment...")

    # 列出可用的环境
    print("\nAvailable ManiSkill3 environments:")
    print(f"Total: {len(REGISTERED_ENVS)}\n")

    # 显示前20个环境
    env_list = sorted(REGISTERED_ENVS.keys())[:20]
    for i, env_name in enumerate(env_list, 1):
        print(f"  {i}. {env_name}")

    print(f"\n... and {len(REGISTERED_ENVS) - 20} more")

    # 测试单个环境
    print("\n" + "="*60)
    print("Testing PickCube-v0 environment")
    print("="*60 + "\n")

    try:
        env = create_maniskill_env(
            task_name="PickCube-v0",
            obs_mode="rgbd",
            num_envs=1
        )

        print(f"Environment created: {env.task_name}")
        print(f"Config: obs_mode={env.config.obs_mode}, control_mode={env.config.control_mode}")

        # 重置环境
        obs = env.reset()
        print(f"\nReset successful!")
        print(f"  RGB shape: {obs['rgb'].shape}")
        if 'depth' in obs:
            print(f"  Depth shape: {obs['depth'].shape}")

        # 执行几步随机动作
        print("\nExecuting random actions...")
        for i in range(3):
            action = env.env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.3f}, done={done}, success={info.get('success', False)}")
            if done:
                print(f"  Task completed!")
                break

        env.close()
        print("\n✅ ManiSkill3 test passed!")

    except Exception as e:
        print(f"\n✗ ManiSkill3 test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nAll tests completed!")