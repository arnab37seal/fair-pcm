from setuptools import setup, find_packages

setup(
    name="fair-pcm",
    version="0.1.0",
    description="Fair Possibilistic C-Means (Fair-PCM) clustering",
    author="",
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "scikit-learn>=1.1",
        "pandas>=1.5",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "matplotlib>=3.6"],
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
