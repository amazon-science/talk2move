from setuptools import setup, find_packages

setup(
    name="flow-grpo",
    version="0.0.2",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai",
    ],
    extras_require={
    }
)
