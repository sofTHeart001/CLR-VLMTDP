"""
RLBench Environment Wrapper
真实RLBench仿真环境包装类
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from rlbench import Environment
from rlbench.tasks import *
from rlbench.observation import Observation
from pyrep.objects import VisionSensor

# 支持的任务映射
TASKS = {
    "stack_blocks": StackBlocks,
    "put_item_in_drawer": PutItemInDrawer,
    "open_drawer": OpenDrawer,
    "close_drawer": CloseDrawer,
    "pick_up_cup": PickUpCup,
    "reach_target": ReachTarget,
    "water_plants": WaterPlants,
    "close_microwave": CloseMicrowave,
    "open_wine_bottle": OpenWineBottle,
    "sweep_to_dustpan": SweepToDustpan,
    "phone_on_base": PhoneOnBase,
}


class RLBenchConfig:
    """RLBench环境配置"""

    def __init__(
        self,
        headless: bool = True,
        random_seed: int = 42,
        image_width: int = 800,
        image_height: int = 600,
        action_mode: str = "joint_velocity",
        include_gripper: bool = True,
    ):
        self.headless = headless
        self.random_seed = random_seed
        self.image_width = image_width
        self.image_height = image_height
        self.action_mode = action_mode
        self.include_gripper = include_gripper


class RLBenchEnvironment:
    """
    RLBench环境包装类

    支持的功能:
    - Franka Panda机器人 (7 DOF)
    - 前视图 + 手腕视图相机
    - 多任务支持
    - 专家演示生成
    """

    def __init__(
        self,
        config: RLBenchConfig,
        task_name: str = "stack_blocks",
        multi_view: bool = False
    ):
        """
        初始化RLBench环境

        Args:
            config: 环境配置
            task_name: 任务名称
            multi_view: 是否使用多视图（前视图+手腕视图）
        """
        self.config = config
        self.task_name = task_name
        self.multi_view = multi_view

        # 设置action mode
        from rlbench.action_modes import ActionMode
        if config.action_mode == "joint_velocity":
            action_mode = ActionMode.JOINT_VELOCITY
        elif config.action_mode == "joint_position":
            action_mode = ActionMode.JOINT_POSITION
        elif config.action_mode == "joint_torque":
            action_mode = ActionMode.JOINT_TORQUE
        elif config.action_mode == "end_effector_pose":
            action_mode = ActionMode.END_EFFECTOR_POSE
        else:
            action_mode = ActionMode.JOINT_VELOCITY

        # 设置observation配置
        from rlbench.observation_config import ObservationConfig
        obs_config = ObservationConfig()

        # 设置图像观测
        obs_config.set_all(False)  # 先禁用所有
        obs_config.front_rgb = True
        obs_config.front_depth = True
        obs_config.front_mask = False

        # 多视图时添加手腕相机
        if multi_view:
            obs_config.wrist_rgb = True
            obs_config.wrist_depth = True
            obs_config.wrist_mask = False

        # 关节状态
        obs_config.joint_positions = True
        obs_config.joint_velocities = True
        obs_config.joint_forces = True

        # 末端执行器
        obs_config.gripper_pose = True
        obs_config.gripper_open = True
        obs_config.gripper_joint_positions = True

        # 任务相关
        obs_config.task_low_dim_state = True

        # 设置图像尺寸
        obs_config.image_size = [config.image_height, config.image_width]

        # 创建环境
        self.env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=config.headless,
            random_seed=config.random_seed
        )

        # 启动环境
        self.env.launch()

        # 加载任务
        self.task = self._load_task(task_name)

        # 设置相机
        self._setup_cameras()

        # 重置环境
        self._reset()

    def _load_task(self, task_name: str):
        """加载指定任务"""
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task: {task_name}. Available tasks: {list(TASKS.keys())}"
            )
        return self.env.get_task(TASKS[task_name])

    def _setup_cameras(self):
        """设置相机（前视图+手腕视图）"""
        # 前视图相机配置（论文中的固定设置）
        self.front_camera = VisionSensor(self.env.pyrep, 0)  # 前相机

        # 手腕视图相机（长时序任务使用）
        if self.multi_view:
            # 手腕相机通常附着在末端执行器上
            self.wrist_camera = VisionSensor(self.env.pyrep, 1)  # 手腕相机

    def _reset(self) -> Dict:
        """重置环境"""
        # RLBench demo + reset pattern
        descriptions, obs = self.task.reset()
        self.current_obs = obs
        self.descriptions = descriptions

        return self._process_observation(obs)

    def _process_observation(self, obs: Observation) -> Dict:
        """
        处理观测，转换为标准格式

        Returns:
            {
                "rgb_front": (H, W, 3) numpy array,
                "depth_front": (H, W) numpy array,
                "rgb_wrist": (H, W, 3) numpy array (可选),
                "depth_wrist": (H, W) numpy array (可选),
                "joint_positions": (7,) numpy array,
                "joint_velocities": (7,) numpy array,
                "gripper_open": float,
                "gripper_pose": (7,) numpy array,
                "task_low_dim_state": array
            }
        """
        result = {}

        # 前视图
        result["rgb_front"] = obs.front_rgb
        result["depth_front"] = obs.front_depth

        # 手腕视图
        if self.multi_view:
            result["rgb_wrist"] = obs.wrist_rgb
            result["depth_wrist"] = obs.wrist_depth

        # 关节状态 (Franka Panda有7个关节)
        result["joint_positions"] = obs.joint_positions
        result["joint_velocities"] = obs.joint_velocities
        result["joint_forces"] = obs.joint_forces

        # 末端执行器
        result["gripper_open"] = obs.gripper_open
        result["gripper_pose"] = obs.gripper_pose
        result["gripper_joint_positions"] = obs.gripper_joint_positions

        # 任务状态
        result["task_low_dim_state"] = obs.task_low_dim_state

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
            action: 动作向量 (7维关节 + 1维夹爪) 或 (7维关节)

        Returns:
            (obs, reward, done, info)
        """
        # 执行动作
        obs, reward, done = self.task.step(action)

        # 处理观测
        processed_obs = self._process_observation(obs)

        # 构建info
        info = {
            "task_descriptions": self.descriptions,
            "success": done  # RLBench中done表示任务成功
        }

        return processed_obs, reward, done, info

    def get_task_description(self) -> List[str]:
        """获取当前任务的描述"""
        return self.descriptions

    def get_demonstration(self, num_demos: int = 1) -> List[List[Dict]]:
        """
        获取专家演示

        Args:
            num_demos: 演示数量

        Returns:
            演示列表，每个演示是一系列观测和动作
        """
        demos = self.task.get_demos(num_demos)

        processed_demos = []
        for demo in demos:
            episode = []
            for obs in demo:
                # 提取动作（每个obs保存了到下一个obs的动作）
                action = obs._gripper_joint_positions  # 或其他action

                episode_data = {
                    "observation": self._process_observation(obs),
                    "action": action
                }
                episode.append(episode_data)
            processed_demos.append(episode)

        return processed_demos

    def close(self):
        """关闭环境"""
        self.env.shutdown()

    def __del__(self):
        """析构函数，确保环境关闭"""
        try:
            self.close()
        except:
            pass


def create_rlbench_env(
    task_name: str = "stack_blocks",
    headless: bool = True,
    multi_view: bool = False,
    image_size: Tuple[int, int] = (800, 600),
    action_mode: str = "joint_velocity"
) -> RLBenchEnvironment:
    """
    工厂函数：创建RLBench环境

    Args:
        task_name: 任务名称
        headless: 无头模式（不显示GUI）
        multi_view: 使用多视图（前+手腕）
        image_size: 图像尺寸 (width, height)
        action_mode: 动作模式

    Returns:
        RLBenchEnvironment实例
    """
    config = RLBenchConfig(
        headless=headless,
        random_seed=42,
        image_width=image_size[0],
        image_height=image_size[1],
        action_mode=action_mode
    )

    return RLBenchEnvironment(
        config=config,
        task_name=task_name,
        multi_view=multi_view
    )


# 测试代码
if __name__ == "__main__":
    print("Testing RLBench environment...")

    # 测试单视图环境
    print("\n1. Testing single-view environment...")
    try:
        env = create_rlbench_env(
            task_name="stack_blocks",
            headless=False,  # 显示窗口便于观察
            multi_view=False
        )

        print(f"   Environment created successfully!")
        print(f"   Task: {env.task_name}")

        # 重置环境
        obs = env.reset()
        print(f"   Reset successful!")
        print(f"   Front RGB shape: {obs['rgb_front'].shape}")
        print(f"   Joint positions: {obs['joint_positions']}")
        print(f"   Gripper open: {obs['gripper_open']}")

        # 执行几步随机动作
        print("\n   Executing random actions...")
        for i in range(5):
            action = np.random.uniform(-1, 1, 8)  # 7 joints + gripper
            obs, reward, done, info = env.step(action)
            print(f"   Step {i+1}: reward={reward:.3f}, done={done}")
            if done:
                print(f"   Task completed!")
                break

        env.close()
        print("   Single-view test passed!")

    except Exception as e:
        print(f"   Single-view test failed: {e}")
        import traceback
        traceback.print_exc()

    # 测试多视图环境
    print("\n2. Testing multi-view environment...")
    try:
        env = create_rlbench_env(
            task_name="stack_blocks",
            headless=False,
            multi_view=True
        )

        obs = env.reset()
        print(f"   Multi-view environment created!")
        print(f"   Front RGB shape: {obs['rgb_front'].shape}")
        print(f"   Wrist RGB shape: {obs['rgb_wrist'].shape}")

        env.close()
        print("   Multi-view test passed!")

    except Exception as e:
        print(f"   Multi-view test failed: {e}")

    print("\nAll tests completed!")