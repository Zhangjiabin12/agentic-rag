#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
agentic-rag 项目安装配置
"""
from setuptools import setup, find_packages
import os

# 读取README.md作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 依赖项列表
requirements = [
    "openai>=1.6.0",
    "pydantic>=2.0.0",
    "httpx>=0.24.0",
    "agents>=0.2.0",
    "asyncio>=3.4.3",
    "typing_extensions>=4.5.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "python-dotenv>=1.0.0",
    "loguru>=0.7.0",
    "uvicorn>=0.23.0",
    "fastapi>=0.100.0",
]

setup(
    name="agentic-rag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="智能代理驱动的RAG系统实战",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agentic-rag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "agentic-rag=chat_llm.test_cacse_agent:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/agentic-rag/issues",
        "Source": "https://github.com/yourusername/agentic-rag",
        "Documentation": "https://agentic-rag.readthedocs.io/",
    },
) 