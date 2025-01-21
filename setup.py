from setuptools import setup, find_packages

setup(
    name="common",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "huggingface_hub"
    ],
    author="Zack Ankner",
    author_email="zackankner@gmail.com",
    description="Common utilities I use",
    url="https://github.com/zankner/common",  # Replace with your project's URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
