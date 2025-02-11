from setuptools import setup, find_packages

setup(
    name="spider-cache",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'redis>=5.0.0',
        'redis-py-cluster>=2.1.0',
        'hnswlib>=0.6.0',
        'numpy>=1.19.0',
        'Pillow>=8.0.0',
        'wandb>=0.12.0'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A distributed caching system with importance sampling for deep learning",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spider-cache",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
) 