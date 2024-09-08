from setuptools import find_packages, setup

setup(
    name="apropos-ai",
    version="0.1.21",
    packages=find_packages(),
    install_requires=[
        "networkx>=3.3,<4.0",
        "pydantic>=2.8.2,<3.0",
        "regex>=2024.7.24,<2025.0",
        "loguru>=0.7.2,<0.8",
        "ruff>=0.5.5,<0.6",
        "backoff>=2.2.1,<3.0",
        "openai>=1.37.1,<2.0",
        "diskcache>=5.6.3,<6.0",
        "instructor>=1.3.7,<2.0",
        "datasets>=2.20.0,<3.0",
        "groq>=0.9.0,<1.0",
        "docker>=6.0.0,<7.0",
    ],
    author="Josh Purtell",
    author_email="jmvpurtell@gmail.com",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JoshuaPurtell/Apropos",
    license="MIT",
)
