"""Setup script for the package."""
from setuptools import setup, find_packages

setup(
    name="sagemaker-mlmodel-example",
    version="0.1.0",
    description="Example SageMaker ML pipeline with multiple models",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "sagemaker>=2.200.0",
        "boto3>=1.34.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "cdk": [
            "aws-cdk-lib>=2.100.0",
            "constructs>=10.0.0",
        ],
    },
)
