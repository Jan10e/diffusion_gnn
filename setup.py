from setuptools import setup, find_packages

setup(
    name="moldiff",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "deepchem>=2.7.0",
        "rdkit-pypi>=2021.9.4",
        "torch-geometric>=2.0.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "tensorboard>=2.7.0",
    ],
    python_requires=">=3.8",
)