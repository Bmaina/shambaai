from setuptools import setup, find_packages

setup(
    name="shambaai",
    version="1.0.0",
    description="Predictive crop disease intelligence for East African smallholders",
    author="Benson M. Gachaga",
    author_email="maina.anu@gmail.com",
    url="https://github.com/Bmaina/shambaai",
    packages=find_packages(exclude=["tests*", "demo*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.0",
        "joblib>=1.3.0",
        "rasterio>=1.3.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "python-multipart>=0.0.6",
        ],
        "gee": [
            "earthengine-api>=0.1.370",
        ],
        "export": [
            "onnx>=1.14.0",
            "onnx-tf>=1.10.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
