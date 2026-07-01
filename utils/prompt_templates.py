"""
Prompt Templates for VLM
定义所有VLM交互所需的Prompt模板
"""

from typing import Dict, List, Optional


class PromptTemplate:
    """VLM Prompt模板管理器"""

    # 任务拆解模板
    TASK_DECOMPOSITION = """You are an expert robot task planner. Your goal is to decompose complex long-horizon tasks into atomic subtasks and generate corresponding voxel trajectories.

## Context
- Overall task: {task_description}
- Completed subtasks: {completed_subtasks}
- Current scene: [RGB image provided]

## Instructions
1. Analyze the current scene and understand the task progress
2. Generate the NEXT subtask to execute
3. Create a 6×6×6 voxel trajectory matrix for this subtask

## Voxel Trajectory Format
The voxel trajectory is a 3D matrix where:
- Dimensions: 6×6×6 (x, y, z)
- Values: 0 = empty, 1 = occupied
- Represents the spatial path the robot should follow

## Output Format
Return JSON in the following format:
```json
{{
    "subtask": "detailed description of the next atomic subtask",
    "voxel_trajectory": [
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]
}}
```

## Constraints
- Subtasks must be atomic and executable within ~50 robot steps
- Voxel trajectory must be physically feasible
- If all tasks are completed, return {{"subtask": "DONE", "voxel_trajectory": [[0,...],...]}}"""

    # 状态检查模板
    STATE_CHECK = """You are a robot state evaluator. Your goal is to verify if the current subtask has been completed or if replanning is needed.

## Context
- Current subtask: {current_subtask}
- Expected outcome: {expected_outcome}
- Current scene: [RGB image provided]

## Evaluation Criteria
1. **Subtask Done**: Has the main objective of the subtask been achieved?
2. **Need Replan**: Has there been a significant deviation that requires re-planning?

## Output Format
Return JSON in the following format:
```json
{{
    "subtask_done": true/false,
    "need_replan": true/false,
    "reason": "brief explanation of your assessment",
    "confidence": 0.0-1.0
}}
```

## Guidelines
- Be conservative: only mark as DONE if confident the objective is achieved
- Mark for REPLAN if the robot deviates significantly from the intended path
- Provide a clear reason for your decision"""

    # 初始任务分析模板
    INITIAL_TASK_ANALYSIS = """You are a robotics task analyst. Analyze the given task and provide a high-level plan.

## Task
{task_description}

## Instructions
1. Understand the overall task goal
2. Identify key objects and their relationships
3. Estimate the number of atomic subtasks needed
4. Identify potential failure modes

## Output Format
```json
{{
    "task_type": "single_object | multi_object | sequential",
    "estimated_subtasks": integer,
    "critical_objects": ["obj1", "obj2", ...],
    "potential_failures": ["failure1", "failure2", ...],
    "suggested_checkpoints": ["checkpoint1", "checkpoint2", ...]
}}
```"""

    # 错误恢复模板
    ERROR_RECOVERY = """You are a robot error recovery planner. The robot has encountered an error during task execution.

## Context
- Original subtask: {original_subtask}
- Error type: {error_type}
- Error description: {error_description}
- Current state: {current_state}
- Current scene: [RGB image provided]

## Instructions
1. Assess the severity of the error
2. Determine if the error is recoverable
3. Generate a recovery plan if possible

## Output Format
```json
{{
    "recoverable": true/false,
    "recovery_action": "action to recover",
    "modified_subtask": "new subtask description if needed",
    "reset_required": true/false
}}
```"""

    # 轨迹修正模板
    TRAJECTORY_CORRECTION = """You are a trajectory correction planner. The robot is deviating from the planned path.

## Context
- Current subtask: {current_subtask}
- Planned trajectory: {planned_trajectory}
- Current position: {current_position}
- Current velocity: {current_velocity}
- Deviation amount: {deviation}
- Current scene: [RGB image provided]

## Instructions
1. Assess if the deviation is critical
2. Generate a corrected voxel trajectory if needed

## Output Format
```json
{{
    "correction_needed": true/false,
    "corrected_trajectory": [
        [0, 1, 0, 0, 0, 0],
        ...
    ],
    "reason": "explanation of correction"
}}
```"""

    @classmethod
    def format_task_decomposition(
        cls,
        task_description: str,
        completed_subtasks: List[str]
    ) -> str:
        """格式化任务拆解Prompt"""
        completed_str = ", ".join(completed_subtasks) if completed_subtasks else "None"
        return cls.TASK_DECOMPOSITION.format(
            task_description=task_description,
            completed_subtasks=completed_str
        )

    @classmethod
    def format_state_check(
        cls,
        current_subtask: str,
        expected_outcome: Optional[str] = None
    ) -> str:
        """格式化状态检查Prompt"""
        if expected_outcome is None:
            expected_outcome = "Complete the described subtask"
        return cls.STATE_CHECK.format(
            current_subtask=current_subtask,
            expected_outcome=expected_outcome
        )

    @classmethod
    def format_initial_analysis(cls, task_description: str) -> str:
        """格式化初始任务分析Prompt"""
        return cls.INITIAL_TASK_ANALYSIS.format(
            task_description=task_description
        )

    @classmethod
    def format_error_recovery(
        cls,
        original_subtask: str,
        error_type: str,
        error_description: str,
        current_state: str
    ) -> str:
        """格式化错误恢复Prompt"""
        return cls.ERROR_RECOVERY.format(
            original_subtask=original_subtask,
            error_type=error_type,
            error_description=error_description,
            current_state=current_state
        )

    @classmethod
    def format_trajectory_correction(
        cls,
        current_subtask: str,
        planned_trajectory: str,
        current_position: str,
        current_velocity: str,
        deviation: float
    ) -> str:
        """格式化轨迹修正Prompt"""
        return cls.TRAJECTORY_CORRECTION.format(
            current_subtask=current_subtask,
            planned_trajectory=planned_trajectory,
            current_position=current_position,
            current_velocity=current_velocity,
            deviation=deviation
        )


# 便捷函数
def get_decomposition_prompt(
    task: str,
    completed: List[str]
) -> str:
    """获取任务拆解Prompt"""
    return PromptTemplate.format_task_decomposition(task, completed)


def get_state_check_prompt(
    subtask: str,
    expected: Optional[str] = None
) -> str:
    """获取状态检查Prompt"""
    return PromptTemplate.format_state_check(subtask, expected)


def get_analysis_prompt(task: str) -> str:
    """获取任务分析Prompt"""
    return PromptTemplate.format_initial_analysis(task)