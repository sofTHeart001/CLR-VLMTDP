"""
RLBench Demo Data Generator
生成专家演示数据用于训练
"""

import os
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import pickle

from scripts.rlbench_environment import (
    create_rlbench_env,
    TASKS,
    RLBenchEnvironment
)


def save_episode(episode_data: List[Dict], save_path: Path):
    """
    保存单个episode的数据

    Args:
        episode_data: episode的观测和动作列表
        save_path: 保存路径
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(episode_data, f)


def save_metadata(metadata: Dict, save_path: Path):
    """
    保存元数据

    Args:
        metadata: 元数据字典
        save_path: 保存路径
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def generate_demos(
    task_name: str,
    num_demos: int,
    output_dir: Path,
    headless: bool = True,
    multi_view: bool = False
) -> Dict:
    """
    生成指定任务的专家演示

    Args:
        task_name: 任务名称
        num_demos: 生成演示数量
        output_dir: 输出目录
        headless: 无头模式
        multi_view: 多视图

    Returns:
        元数据统计
    """
    print(f"\n{'='*80}")
    print(f"Generating demos for task: {task_name}")
    print(f"Number of demos: {num_demos}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # 创建环境
    print("Creating RLBench environment...")
    env = create_rlbench_env(
        task_name=task_name,
        headless=headless,
        multi_view=multi_view
    )

    # 统计信息
    stats = {
        "task_name": task_name,
        "num_demos": num_demos,
        "success_count": 0,
        "episode_lengths": [],
        "demos": []
    }

    # 生成演示
    print(f"\nGenerating {num_demos} demonstrations...")
    for demo_idx in tqdm(range(num_demos), desc=f"{task_name}"):
        try:
            # 重置环境
            obs = env.reset()

            # 获取专家演示
            demo_demos = env.get_demonstration(num_demos=1)

            if len(demo_demos) > 0:
                demo_data = demo_demos[0]

                # 保存episode数据
                save_path = output_dir / task_name / f"episode_{demo_idx:04d}.pkl"
                save_episode(demo_data, save_path)

                # 统计
                episode_length = len(demo_data)
                stats["episode_lengths"].append(episode_length)
                stats["success_count"] += 1
                stats["demos"].append({
                    "episode_id": demo_idx,
                    "length": episode_length,
                    "save_path": str(save_path)
                })

                if demo_idx % 10 == 0:
                    avg_len = np.mean(stats["episode_lengths"])
                    print(f"  Episode {demo_idx}: length={episode_length}, avg_length={avg_len:.1f}")

        except Exception as e:
            print(f"\n  Warning: Demo {demo_idx} failed: {e}")
            continue

    # 计算统计信息
    stats["success_rate"] = stats["success_count"] / num_demos if num_demos > 0 else 0
    stats["avg_episode_length"] = np.mean(stats["episode_lengths"]) if stats["episode_lengths"] else 0
    stats["std_episode_length"] = np.std(stats["episode_lengths"]) if stats["episode_lengths"] else 0

    # 保存元数据
    metadata_path = output_dir / task_name / "metadata.json"
    save_metadata(stats, metadata_path)

    # 关闭环境
    env.close()

    return stats


def generate_multi_task_demos(
    tasks: List[str],
    num_demos_per_task: int,
    output_dir: Path,
    headless: bool = True,
    multi_view: bool = False
) -> Dict:
    """
    为多个任务生成演示数据

    Args:
        tasks: 任务名称列表
        num_demos_per_task: 每个任务的演示数量
        output_dir: 输出目录
        headless: 无头模式
        multi_view: 多视图

    Returns:
        所有任务的统计信息
    """
    print(f"\n{'='*80}")
    print(f"Generating demos for {len(tasks)} tasks")
    print(f"Demos per task: {num_demos_per_task}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    all_stats = {
        "tasks": tasks,
        "num_demos_per_task": num_demos_per_task,
        "total_demos": len(tasks) * num_demos_per_task,
        "task_stats": {}
    }

    for task_name in tasks:
        print(f"\n--- Task: {task_name} ({tasks.index(task_name)+1}/{len(tasks)}) ---")
        try:
            stats = generate_demos(
                task_name=task_name,
                num_demos=num_demos_per_task,
                output_dir=output_dir,
                headless=headless,
                multi_view=multi_view
            )
            all_stats["task_stats"][task_name] = stats
        except Exception as e:
            print(f"Error generating demos for {task_name}: {e}")
            import traceback
            traceback.print_exc()

    # 保存全局元数据
    global_metadata_path = output_dir / "metadata.json"
    save_metadata(all_stats, global_metadata_path)

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Generate RLBench demonstration data")
    parser.add_argument("--tasks", type=str, nargs="+", default=["stack_blocks"],
                        help=f"Tasks to generate demos for. Available: {list(TASKS.keys())}")
    parser.add_argument("--num_demos", type=int, default=100,
                        help="Number of demos per task")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Output directory for demo data")
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run in headless mode (no GUI)")
    parser.add_argument("--no_headless", action="store_false", dest="headless",
                        help="Show GUI")
    parser.add_argument("--multi_view", action="store_true",
                        help="Use multi-view (front + wrist camera)")
    parser.add_argument("--single_task", action="store_true",
                        help="Generate demos for a single task")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 验证任务名称
    invalid_tasks = [t for t in args.tasks if t not in TASKS]
    if invalid_tasks:
        print(f"Error: Invalid tasks: {invalid_tasks}")
        print(f"Available tasks: {list(TASKS.keys())}")
        return

    # 生成演示
    if len(args.tasks) == 1:
        # 单任务
        stats = generate_demos(
            task_name=args.tasks[0],
            num_demos=args.num_demos,
            output_dir=output_dir,
            headless=args.headless,
            multi_view=args.multi_view
        )

        # 打印统计
        print(f"\n{'='*80}")
        print(f"Summary for {args.tasks[0]}:")
        print(f"{'='*80}")
        print(f"  Total demos requested: {args.num_demos}")
        print(f"  Successful demos: {stats['success_count']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Average episode length: {stats['avg_episode_length']:.1f}")
        print(f"  Std episode length: {stats['std_episode_length']:.1f}")

    else:
        # 多任务
        stats = generate_multi_task_demos(
            tasks=args.tasks,
            num_demos_per_task=args.num_demos,
            output_dir=output_dir,
            headless=args.headless,
            multi_view=args.multi_view
        )

        # 打印统计
        print(f"\n{'='*80}")
        print(f"Summary for all tasks:")
        print(f"{'='*80}")
        print(f"  Total tasks: {len(args.tasks)}")
        print(f"  Demos per task: {args.num_demos}")
        print(f"  Total demos generated: {stats['total_demos']}")

        for task_name, task_stats in stats["task_stats"].items():
            print(f"\n  {task_name}:")
            print(f"    Success: {task_stats['success_count']}/{args.num_demos} ({task_stats['success_rate']:.2%})")
            print(f"    Avg length: {task_stats['avg_episode_length']:.1f}")

    print(f"\nData saved to: {output_dir}")


if __name__ == "__main__":
    main()