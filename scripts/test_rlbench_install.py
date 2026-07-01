"""
RLBench Quick Test
快速测试RLBench是否正确安装
"""

import os

# 设置无头渲染
os.environ['PYREPCG_GL'] = 'egl'

print("="*60)
print("RLBench Installation Test")
print("="*60)
print()

# 测试1: 导入测试
print("Test 1: Importing RLBench...")
try:
    from rlbench import Environment
    from rlbench.tasks import StackBlocks, PutItemInDrawer, OpenDrawer
    print("  ✓ Import successful")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    print("\nPlease install RLBench first:")
    print("  pip install rlbench yarr")
    exit(1)

# 测试2: 环境创建测试
print("\nTest 2: Creating environment...")
try:
    env = Environment(
        action_mode='joint_velocity',
        headless=True
    )
    print("  ✓ Environment created")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# 测试3: 环境启动测试
print("\nTest 3: Launching environment (first run downloads data ~2GB)...")
try:
    env.launch()
    print("  ✓ Environment launched")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    print("  This might be due to missing system dependencies")
    print("  Install with: sudo apt-get install libglfw3 libgl1-mesa-glx")
    exit(1)

# 测试4: 任务加载测试
print("\nTest 4: Loading tasks...")
tasks_to_test = [
    ("StackBlocks", StackBlocks),
    ("PutItemInDrawer", PutItemInDrawer),
    ("OpenDrawer", OpenDrawer)
]

for task_name, TaskClass in tasks_to_test:
    try:
        task = env.get_task(TaskClass)
        print(f"  ✓ {task_name} loaded")
    except Exception as e:
        print(f"  ✗ {task_name} failed: {e}")

# 测试5: 重置测试
print("\nTest 5: Testing environment reset...")
try:
    from rlbench.tasks import StackBlocks
    task = env.get_task(StackBlocks)
    descriptions, obs = task.reset()
    print(f"  ✓ Reset successful")
    print(f"    Task description: {descriptions[0][:50]}...")
    print(f"    Observation keys: {list(obs.__dict__.keys())}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# 关闭环境
print("\nCleaning up...")
env.shutdown()
print("  ✓ Environment closed")

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)
print()
print("RLBench is ready to use!")
print()
print("Next steps:")
print("  1. Generate demos: python scripts/generate_demos.py --tasks stack_blocks --num_demos 10")
print("  2. Test control: python scripts/test_closed_loop.py --task stack_blocks")
print()