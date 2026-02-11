# SageMaker ML Model Example

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

A production-ready example of building an end-to-end machine learning pipeline with AWS SageMaker. This project demonstrates best practices for training, comparing, and deploying multiple ML models using SageMaker's built-in algorithms and infrastructure as code.

## 🎯 What This Project Does

- **Multi-Model Training**: Train and compare XGBoost, KNN, and scikit-learn models
- **Automated Hyperparameter Tuning**: Find optimal parameters using SageMaker's tuning jobs
- **Model Comparison**: Automatically select the best model based on performance metrics
- **Batch Inference**: Deploy models for scalable batch predictions
- **Infrastructure as Code**: Provision all AWS resources using AWS CDK
- **Local Development**: Run and test the entire pipeline locally without AWS costs

## 🚀 Quick Start

### Running Locally (No AWS Required)

```bash
# Clone and install
git clone https://github.com/bojanderson/sagemaker-mlmodel-example.git
cd sagemaker-mlmodel-example
pip install -r requirements.txt

# Run the complete pipeline in mock mode
python src/run_pipeline.py --bucket demo-bucket --mock
```

This runs the entire pipeline locally with mock training results. Perfect for:
- Understanding the pipeline flow
- Testing changes without AWS costs
- CI/CD integration
- Development and debugging

### Running on AWS SageMaker

```bash
# Set up AWS credentials
export AWS_DEFAULT_REGION=us-east-1

# Run training on SageMaker
python src/run_pipeline.py --bucket my-sagemaker-bucket --no-mock

# With hyperparameter tuning
python src/run_pipeline.py --bucket my-sagemaker-bucket --no-mock --tune

# Full pipeline with deployment
python src/run_pipeline.py --bucket my-sagemaker-bucket --no-mock --deploy
```

**💡 Tip**: Deploy infrastructure with CDK first (see [Deploying Infrastructure](#deploying-infrastructure-with-cdk))

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Testing](#testing)
- [Cost Considerations](#cost-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🏗️ Architecture Overview

The pipeline follows a linear workflow with five main stages:

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Data            │────▶│ S3 Upload    │────▶│ Model Training  │
│ Preparation     │     │              │     │ (XGB/KNN/GBM)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────────────────────────────┐
│ Batch Transform │◀────│ Model Comparison & Selection         │
│ Deployment      │     │ (Select best based on RMSE/MSE/MAE) │
└─────────────────┘     └──────────────────────────────────────┘
```

**Key Components**:
- **Data Layer**: California housing dataset → preprocessing → S3 storage
- **Training Layer**: Parallel training of 3 models using SageMaker containers
- **Comparison Layer**: Metric-based model selection (RMSE, MSE, MAE)
- **Deployment Layer**: Batch Transform for scalable inference

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## ✨ Features

This repository demonstrates a production-grade ML pipeline with:
- **XGBoost** - Gradient boosting algorithm
- **KNN** - K-Nearest Neighbors
- **scikit-learn GBM** - Gradient Boosting Machine

The pipeline includes:
1. **Data Preparation**: Uses the California housing dataset as an example
2. **Training**: Train all three models with configurable hyperparameters
3. **Hyperparameter Tuning**: Automatically find the best hyperparameters for each model
4. **Model Comparison**: Compare models based on performance metrics (RMSE, MSE, MAE)
5. **Deployment**: Deploy the best model for batch inference using SageMaker Batch Transform
6. **Infrastructure**: AWS CDK code to provision S3 buckets, IAM roles, and SageMaker resources

## Project Structure

```
.
├── src/
│   ├── pipeline/
│   │   ├── data_preparation.py      # Data generation and preprocessing
│   │   ├── training.py               # Model training and tuning
│   │   ├── model_comparison.py       # Model comparison logic
│   │   └── batch_deployment.py       # Batch inference deployment
│   ├── cdk/
│   │   ├── app.py                    # CDK app entry point
│   │   ├── sagemaker_stack.py        # Infrastructure stack
│   │   └── cdk.json                  # CDK configuration
│   └── run_pipeline.py               # Main pipeline orchestration
├── scripts/
│   └── sklearn_gbm_script.py         # scikit-learn training script
├── tests/
│   ├── unit/                         # Unit tests
│   └── integration/                  # Integration tests
├── requirements.txt                  # Python dependencies
└── requirements-cdk.txt              # CDK dependencies
```

## Prerequisites

- Python 3.8+
- AWS Account with SageMaker access
- AWS CLI configured with appropriate credentials
- Node.js (for CDK)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bojanderson/sagemaker-mlmodel-example.git
cd sagemaker-mlmodel-example
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install CDK dependencies (optional, for infrastructure deployment):
```bash
pip install -r requirements-cdk.txt
npm install -g aws-cdk
```

## Usage

### Running the Pipeline Locally (Mock Mode)

Run the pipeline with mock training results (no AWS credentials required):

```bash
python src/run_pipeline.py --bucket my-bucket --mock
```

This will:
- Generate and preprocess the California housing dataset
- Set up training configurations for all three models
- Use mock training results for demonstration
- Compare models and select the best one
- Save results to `model_comparison.json`

### Running the Pipeline on SageMaker

To actually run training on SageMaker:

```bash
python src/run_pipeline.py --bucket my-sagemaker-bucket --no-mock
```

With hyperparameter tuning:

```bash
python src/run_pipeline.py --bucket my-sagemaker-bucket --no-mock --tune
```

With deployment:

```bash
python src/run_pipeline.py --bucket my-sagemaker-bucket --no-mock --deploy
```

### Deploying Infrastructure with CDK

1. Bootstrap CDK (first time only):
```bash
cd src/cdk
cdk bootstrap
```

2. Deploy the stack:
```bash
cdk deploy
```

This creates:
- S3 bucket for data and models
- IAM roles for SageMaker execution
- CloudWatch log groups
- (Optional) SageMaker Notebook instance

3. Get the outputs:
```bash
cdk outputs
```

The stack outputs include:
- `DataBucketName`: S3 bucket name
- `SageMakerRoleArn`: IAM role for SageMaker
- `NotebookRoleArn`: IAM role for notebooks

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run unit tests only:
```bash
pytest tests/unit/ -v
```

Run integration tests:
```bash
pytest tests/integration/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Pipeline Components

### 1. Data Preparation

The `data_preparation.py` module:
- Loads the California housing dataset
- Splits into train/test sets
- Preprocesses data for SageMaker format (target column first, no headers)

### 2. Model Training

The `training.py` module sets up three models:

**XGBoost**:
- Uses SageMaker built-in XGBoost container
- Tunable hyperparameters: max_depth, eta, subsample, colsample_bytree, num_round

**KNN**:
- Uses SageMaker built-in KNN container
- Tunable hyperparameters: k, sample_size

**scikit-learn GBM**:
- Uses SageMaker scikit-learn container with custom script
- Tunable hyperparameters: n_estimators, max_depth, learning_rate

### 3. Model Comparison

The `model_comparison.py` module:
- Retrieves metrics from completed training jobs
- Compares models based on error metrics (RMSE, MSE, MAE)
- Selects the best performing model
- Saves comparison results to JSON

### 4. Batch Deployment

The `batch_deployment.py` module:
- Creates a SageMaker model from the best training job
- Sets up a Batch Transform job
- Processes test data and saves predictions to S3

## Example Output

```
============================================================
SageMaker ML Pipeline
============================================================

[Step 1/5] Generating and preparing data...
Train data shape: (16512, 9)
Test data shape: (4128, 9)
✓ Data preparation complete

[Step 2/5] Uploading data to S3...
  Uploaded train_processed.csv to s3://my-bucket/sagemaker-housing/data/train/train_processed.csv
  Uploaded test_processed.csv to s3://my-bucket/sagemaker-housing/data/validation/test_processed.csv
✓ Data upload complete

[Step 3/5] Training models: xgboost, knn, sklearn-gbm...
Using mock training results for demonstration...
✓ Mock training complete

[Step 4/5] Comparing models and selecting best...

=== Model Comparison ===
xgboost: RMSE = 0.5200
knn: MSE = 0.3100
sklearn-gbm: RMSE = 0.5000

✓ Best model: sklearn-gbm (RMSE = 0.5000)
Comparison results saved to model_comparison.json
✓ Model comparison complete

[Step 5/5] Skipping deployment (use --deploy flag to enable)

============================================================
Pipeline execution complete!
============================================================

Best model: sklearn-gbm
Results saved to: model_comparison.json
```

## Configuration

### Environment Variables

- `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)
- `CDK_DEFAULT_ACCOUNT`: AWS account ID for CDK

### Command Line Options

```bash
python src/run_pipeline.py --help
```

Options:
- `--bucket`: S3 bucket name (required)
- `--prefix`: S3 prefix (default: sagemaker-housing)
- `--region`: AWS region (default: us-east-1)
- `--models`: Models to train (default: xgboost knn sklearn-gbm)
- `--tune`: Enable hyperparameter tuning
- `--deploy`: Deploy for batch inference
- `--no-mock`: Actually run training (requires SageMaker)

## Cost Considerations

Running this pipeline on AWS SageMaker will incur costs:
- **Training**: Charges for ml.m5.xlarge instances during training
- **Hyperparameter Tuning**: Additional charges for multiple training jobs
- **Batch Transform**: Charges for ml.m5.xlarge instances during inference
- **S3**: Storage costs for data and models
- **CloudWatch**: Minimal charges for logs

Estimated cost for one complete run: $5-10 USD (depending on region and tuning configuration)

Use mock mode (`--mock`) for testing without AWS charges.

## Testing

The repository includes comprehensive tests:

- **Unit Tests**: Test individual components (data preparation, model comparison, training setup)
- **Integration Tests**: Test the complete pipeline end-to-end

All tests can run without AWS credentials by using mock data.

## Troubleshooting

For common issues and solutions, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

**Quick fixes**:
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
- **AWS credentials**: Configure with `aws configure` or use `--mock` mode
- **SageMaker permissions**: Ensure your IAM role has SageMaker execution permissions
- **Tests failing**: Run `pytest tests/ -v` to see detailed error messages

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
