"""
Closed-loop Testing Script
测试闭环控制系统的完整流程
"""

import os
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
from tqdm import tqdm

from models import FlowTDP, LightVoxelEncoder, VLMWrapper
from utils import PromptTemplate

# 导入真实RLBench环境
try:
    from scripts.rlbench_environment import create_rlbench_env, TASKS
    RLBENCH_AVAILABLE = True
except ImportError:
    RLBENCH_AVAILABLE = False
    print("Warning: RLBench not available. Using mock environment.")


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class ClosedLoopController:
    """
    闭环控制器
    实现生成-执行-验证-重规划的闭环流程
    """

    def __init__(
        self,
        vlm: VLMWrapper,
        voxel_encoder: LightVoxelEncoder,
        policy_model: FlowTDP,
        config: dict
    ):
        self.vlm = vlm
        self.voxel_encoder = voxel_encoder
        self.policy = policy_model
        self.config = config

        # 闭环控制参数
        self.check_interval = config["closed_loop"]["check_interval"]
        self.deviation_threshold = config["closed_loop"]["deviation_threshold"]
        self.max_replan_attempts = config["closed_loop"]["max_replan_attempts"]

        # 推理参数
        self.max_steps_per_subtask = config["closed_loop_inference"]["max_steps_per_subtask"]
        self.max_total_steps = config["closed_loop_inference"]["max_total_steps"]

        # 调试模式
        self.save_debug = config["closed_loop_inference"]["save_debug_images"]
        self.debug_dir = Path(config["closed_loop_inference"]["debug_dir"])
        if self.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def execute_task(
        self,
        task_name: str,
        task_description: str,
        initial_image: torch.Tensor,
        environment
    ) -> Dict:
        """
        执行完整的长时序任务

        Args:
            task_name: 任务名称
            task_description: 任务描述
            initial_image: 初始图像
            environment: 环境实例（RLBench等）

        Returns:
            执行结果字典
        """
        print(f"\n{'='*60}")
        print(f"Executing task: {task_name}")
        print(f"Description: {task_description}")
        print(f"{'='*60}\n")

        # 初始化状态
        current_image = initial_image
        completed_subtasks = []
        total_steps = 0
        replan_counts = []
        execution_log = []

        # VLM初始任务分析
        print("[VLM] Analyzing task...")
        analysis_prompt = PromptTemplate.format_initial_analysis(task_description)
        analysis = self.vlm.generate_text(analysis_prompt)
        print(f"Task analysis: {analysis}\n")

        # 主循环：生成-执行-验证-重规划
        while total_steps < self.max_total_steps:
            # 1. VLM生成下一个子任务
            print(f"[{total_steps}] VLM generating next subtask...")
            subtask, voxel_trajectory = self.vlm.decompose_task(
                current_image,
                task_description,
                completed_subtasks
            )

            print(f"  → Subtask: {subtask}")

            # 检查是否所有任务完成
            if subtask == "DONE":
                print("\n✓ All subtasks completed!")
                break

            # 2. 执行子任务
            subtask_steps = 0
            replan_attempts = 0
            subtask_success = False

            while subtask_steps < self.max_steps_per_subtask:
                # 编码体素轨迹
                trajectory_features = self.voxel_encoder(voxel_trajectory)

                # 策略模型生成动作
                action = self.policy.sample_action(
                    current_image,
                    trajectory_features,
                    num_steps=1,
                    guidance_scale=self.config["inference"]["guidance_scale"]
                )

                # 执行动作
                next_obs, reward, done, info = environment.step(action)
                current_image = next_obs["rgb"]  # 假设环境返回RGB图像

                subtask_steps += 1
                total_steps += 1

                # 3. 状态检查（每check_interval步）
                if subtask_steps % self.check_interval == 0:
                    print(f"  [{total_steps}] Checking state...")
                    state_check = self.vlm.check_state(
                        current_image,
                        subtask,
                        self.deviation_threshold
                    )

                    print(f"    → Done: {state_check['subtask_done']}, Replan: {state_check['need_replan']}")
                    execution_log.append({
                        "step": total_steps,
                        "subtask": subtask,
                        "state_check": state_check
                    })

                    # 如果子任务完成
                    if state_check["subtask_done"]:
                        print(f"  ✓ Subtask completed: {subtask}")
                        completed_subtasks.append(subtask)
                        subtask_success = True
                        break

                    # 如果需要重规划
                    if state_check["need_replan"] and replan_attempts < self.max_replan_attempts:
                        print(f"  → Replanning subtask (attempt {replan_attempts + 1}/{self.max_replan_attempts})")
                        subtask, voxel_trajectory = self.vlm.decompose_task(
                            current_image,
                            task_description,
                            completed_subtasks
                        )
                        replan_attempts += 1
                        replan_counts.append(replan_attempts)
                        continue

            if not subtask_success:
                print(f"  ✗ Subtask failed: {subtask}")
                break

        # 返回执行结果
        result = {
            "task_name": task_name,
            "task_description": task_description,
            "success": len(completed_subtasks) > 0,
            "completed_subtasks": completed_subtasks,
            "total_steps": total_steps,
            "replan_counts": replan_counts,
            "execution_log": execution_log,
            "num_replans": len(replan_counts)
        }

        return result

    def evaluate_on_tasks(
        self,
        tasks: List[Dict],
        environment
    ) -> Dict:
        """
        在多个任务上评估系统

        Args:
            tasks: 任务列表，每个任务包含name和description
            environment: 环境实例

        Returns:
            评估结果汇总
        """
        results = []

        for task in tasks:
            # 重置环境
            initial_obs = environment.reset()
            initial_image = initial_obs["rgb"]

            # 执行任务
            result = self.execute_task(
                task["name"],
                task["description"],
                initial_image,
                environment
            )
            results.append(result)

        # 汇总结果
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        avg_steps = np.mean([r["total_steps"] for r in results])
        avg_replans = np.mean([r["num_replans"] for r in results])

        summary = {
            "total_tasks": len(tasks),
            "success_count": sum(1 for r in results if r["success"]),
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_replans": avg_replans,
            "results": results
        }

        return summary


class RLBenchAdapter:
    """
    RLBench环境适配器
    将RLBench环境接口适配到控制器所需的接口
    """

    def __init__(
        self,
        task_name: str,
        config: Dict,
        multi_view: bool = False,
        headless: bool = True
    ):
        """
        初始化适配器

        Args:
            task_name: 任务名称
            config: 环境配置
            multi_view: 是否使用多视图
            headless: 无头模式
        """
        self.task_name = task_name
        self.multi_view = multi_view

        # 创建RLBench环境
        self.env = create_rlbench_env(
            task_name=task_name,
            headless=headless,
            multi_view=multi_view,
            action_mode=config.get("action_mode", "joint_velocity")
        )

        # 记录任务描述
        self.task_descriptions = None

    def reset(self) -> Dict:
        """
        重置环境

        Returns:
            包含rgb和depth的字典
        """
        obs = self.env.reset()
        self.task_descriptions = self.env.get_task_description()

        # 转换为控制器期望的格式
        result = {
            "rgb": torch.from_numpy(obs["rgb_front"]).permute(2, 0, 1).float() / 255.0,  # (3, H, W)
            "depth": torch.from_numpy(obs["depth_front"]).float(),
        }

        # 如果是多视图，添加手腕相机
        if self.multi_view:
            result["rgb_wrist"] = torch.from_numpy(obs["rgb_wrist"]).permute(2, 0, 1).float() / 255.0
            result["depth_wrist"] = torch.from_numpy(obs["depth_wrist"]).float()

        # 添加其他观测信息
        result["joint_positions"] = torch.from_numpy(obs["joint_positions"]).float()
        result["gripper_open"] = float(obs["gripper_open"])

        return result

    def step(self, action: torch.Tensor) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步

        Args:
            action: 动作张量 (7维关节 + 1维夹爪) 或 (7维关节)

        Returns:
            (obs, reward, done, info)
        """
        # 转换为numpy
        action_np = action.numpy() if isinstance(action, torch.Tensor) else action

        # 执行动作
        obs, reward, done, info = self.env.step(action_np)

        # 转换观测
        result = {
            "rgb": torch.from_numpy(obs["rgb_front"]).permute(2, 0, 1).float() / 255.0,
            "depth": torch.from_numpy(obs["depth_front"]).float(),
        }

        if self.multi_view:
            result["rgb_wrist"] = torch.from_numpy(obs["rgb_wrist"]).permute(2, 0, 1).float() / 255.0
            result["depth_wrist"] = torch.from_numpy(obs["depth_wrist"]).float()

        result["joint_positions"] = torch.from_numpy(obs["joint_positions"]).float()
        result["gripper_open"] = float(obs["gripper_open"])

        return result, reward, done, info

    def close(self):
        """关闭环境"""
        self.env.close()


class MockEnvironment:
    """
    模拟环境（用于测试）
    TODO: 替换为真实的RLBench环境
    """

    def __init__(self):
        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        """重置环境"""
        self.step_count = 0
        return {
            "rgb": torch.randn(3, 600, 800),
            "depth": torch.randn(600, 800)
        }

    def step(self, action):
        """执行一步"""
        self.step_count += 1
        reward = 0.1
        done = self.step_count >= self.max_steps
        info = {}

        return {
            "rgb": torch.randn(3, 600, 800),
            "depth": torch.randn(600, 800)
        }, reward, done, info


def main():
    parser = argparse.ArgumentParser(description="Test closed-loop control")
    parser.add_argument("--config", type=str, default="config/deploy.yaml",
                        help="Path to deployment config")
    parser.add_argument("--task", type=str, default="stack_blocks",
                        help="Task name to test")
    parser.add_argument("--task_desc", type=str, default="Stack all blocks in the center",
                        help="Task description")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--mock_env", action="store_true",
                        help="Use mock environment for testing")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print("Loading models...")

    # TODO: 实际部署时加载预训练权重
    policy_model = FlowTDP().to(device)
    voxel_encoder = LightVoxelEncoder().to(device)
    vlm = VLMWrapper(
        model_name=config["vlm"]["model_name"],
        load_in_4bit=config["vlm"]["load_in_4bit"],
        max_new_tokens=config["vlm"]["max_new_tokens"],
        cache_dir=config["vlm"]["cache_dir"]
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        policy_model.load_state_dict(checkpoint["model_state_dict"])
        if "voxel_encoder_state_dict" in checkpoint:
            voxel_encoder.load_state_dict(checkpoint["voxel_encoder_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")

    # 创建闭环控制器
    controller = ClosedLoopController(vlm, voxel_encoder, policy_model, config)

    # 创建环境
    if args.mock_env:
        environment = MockEnvironment()
        print("Using mock environment")
    elif not RLBENCH_AVAILABLE:
        print("ERROR: RLBench not installed. Use --mock_env for testing or install RLBench first.")
        return
    else:
        # 创建真实RLBench环境
        environment = RLBenchAdapter(
            task_name=args.task,
            config=config["rlbench"],
            multi_view=len(args.task.split("_")) > 1 or "stack" in args.task,
            headless=True
        )
        print(f"Using real RLBench environment: {args.task}")

    # 执行任务
    task = {
        "name": args.task,
        "description": args.task_desc
    }

    initial_obs = environment.reset()
    initial_image = initial_obs["rgb"]

    result = controller.execute_task(
        task["name"],
        task["description"],
        initial_image,
        environment
    )

    # 打印结果
    print("\n" + "="*60)
    print("Task Execution Result")
    print("="*60)
    print(f"Task: {result['task_name']}")
    print(f"Success: {'✓' if result['success'] else '✗'}")
    print(f"Total Steps: {result['total_steps']}")
    print(f"Completed Subtasks: {len(result['completed_subtasks'])}")
    print(f"Replans: {result['num_replans']}")

    if result['completed_subtasks']:
        print("\nCompleted Subtasks:")
        for i, subtask in enumerate(result['completed_subtasks'], 1):
            print(f"  {i}. {subtask}")

    # 保存结果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"{args.task}_result.json"

    # 转换为可序列化的格式
    serializable_result = result.copy()
    serializable_result["completed_subtasks"] = result["completed_subtasks"]
    serializable_result["execution_log"] = []

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2)

    print(f"\nResult saved to {result_path}")


if __name__ == "__main__":
    main()