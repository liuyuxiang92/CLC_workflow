from setuptools import setup, find_packages

setup(
    name="clc-workflow",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Yuxiang Liu",
    description="Screening workflow for perovskites applied in chemical looping beyond combustion",
    install_requires=[
        "numpy",
        "pandas",
    ],
)
