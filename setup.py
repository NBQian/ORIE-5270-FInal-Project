from setuptools import setup, find_packages
setup(
    name="crypto_momentum_lab",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "notebooks"]),
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.5",
        "pyarrow>=10",
        "python-binance>=1.0",
        "matplotlib>=3.5",
    ],
    python_requires=">=3.8",
)
