"""
CLR-VLMTDP Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# 读取requirements
requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8").splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="clr-vlmtdp",
    version="1.0.0",
    author="CLR-VLMTDP Team",
    description="Closed-loop Robust Vision-Language Model guided Trajectory Diffusion Policy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "clr-train=scripts.train_flow_tdp:main",
            "clr-test=scripts.test_closed_loop:main",
        ],
    },
)