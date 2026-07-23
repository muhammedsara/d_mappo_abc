"""
D-MAPPO-ABC: Distributed Multi-Agent Proximal Policy Optimization
Enhanced with Artificial Bee Colony for Edge Computing Task Scheduling
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="d_mappo_abc",
    version="1.0.0",
    author="Muhammed Şara, Koray Özdemir, Süleyman Eken, Adem Tuncer",
    author_email="mmuhammedsaraa@gmail.com",
    description="Distributed MAPPO-ABC for Edge Computing Task Scheduling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muhammedsara/d_mappo_abc",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    package_data={
        "": ["configs/*.yaml"],
    },
    include_package_data=True,
)
