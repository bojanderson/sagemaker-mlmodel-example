# API Documentation

This document provides comprehensive API reference for all pipeline modules and command-line interfaces.

## Table of Contents

- [Command-Line Interface](#command-line-interface)
- [Pipeline Modules](#pipeline-modules)
  - [Data Preparation](#data-preparation-module)
  - [Model Training](#model-training-module)
  - [Model Comparison](#model-comparison-module)
  - [Batch Deployment](#batch-deployment-module)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)

---

## Command-Line Interface

### Main Pipeline Script: `run_pipeline.py`

The main entry point for running the complete ML pipeline.

#### Usage

```bash
python src/run_pipeline.py [OPTIONS]
```

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--bucket` | string | S3 bucket name for storing data and models |

#### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prefix` | string | `sagemaker-housing` | S3 prefix for organizing files |
| `--region` | string | `us-east-1` | AWS region for SageMaker resources |
| `--models` | list | `xgboost knn sklearn-gbm` | Space-separated list of models to train |
| `--tune` | flag | `False` | Enable hyperparameter tuning |
| `--deploy` | flag | `False` | Deploy best model for batch inference |
| `--mock` | flag | `True` | Use mock training results (no AWS required) |
| `--no-mock` | flag | N/A | Actually run training on SageMaker |

#### Examples

**Run in mock mode (no AWS required):**
```bash
python src/run_pipeline.py --bucket demo-bucket --mock
```

**Train models on SageMaker:**
```bash
python src/run_pipeline.py --bucket my-bucket --no-mock
```

**Full pipeline with tuning and deployment:**
```bash
python src/run_pipeline.py --bucket my-bucket --no-mock --tune --deploy
```

**Train only specific models:**
```bash
python src/run_pipeline.py --bucket my-bucket --models xgboost sklearn-gbm --no-mock
```

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (missing arguments, AWS errors, etc.) |
| 2 | Data preparation failed |
| 3 | Training failed |
| 4 | Model comparison failed |
| 5 | Deployment failed |

---

## Pipeline Modules

### Data Preparation Module

**Module:** `src/pipeline/data_preparation.py`

Handles data loading, preprocessing, and formatting for SageMaker.

#### Functions

##### `generate_sample_data(output_dir="data", use_synthetic=False)`

Generate or load California housing dataset.

**Parameters:**
- `output_dir` (str): Directory to save data files. Default: `"data"`
- `use_synthetic` (bool): Generate synthetic data instead of fetching real data. Default: `False`

**Returns:**
- `tuple`: `(train_df, test_df, feature_names, target_name)`
  - `train_df` (pd.DataFrame): Training dataset
  - `test_df` (pd.DataFrame): Test dataset
  - `feature_names` (list): List of feature column names
  - `target_name` (str): Name of target column

**Example:**
```python
from src.pipeline.data_preparation import generate_sample_data

train_df, test_df, features, target = generate_sample_data(
    output_dir="data",
    use_synthetic=False
)

print(f"Train shape: {train_df.shape}")
print(f"Features: {features}")
```

##### `generate_and_prepare_data(synthetic=False)`

Main entry point for data preparation.

**Parameters:**
- `synthetic` (bool): Use synthetic data. Default: `False`

**Returns:**
- `tuple`: `(X_train, X_test, y_train, y_test)`
  - All returns are numpy arrays

**Example:**
```python
from src.pipeline.data_preparation import generate_and_prepare_data

X_train, X_test, y_train, y_test = generate_and_prepare_data(synthetic=True)
```

##### `save_data_for_sagemaker(X_train, X_test, y_train, y_test, output_dir=".")`

Format and save data for SageMaker training.

**Parameters:**
- `X_train` (np.ndarray): Training features
- `X_test` (np.ndarray): Test features
- `y_train` (np.ndarray): Training target
- `y_test` (np.ndarray): Test target
- `output_dir` (str): Directory to save files. Default: `"."`

**Returns:**
- `tuple`: `(train_file, test_file)`
  - Paths to saved CSV files

**Data Format:**
SageMaker requires CSV format with:
- Target column first
- No headers
- No index column

**Example:**
```python
from src.pipeline.data_preparation import save_data_for_sagemaker

train_file, test_file = save_data_for_sagemaker(
    X_train, X_test, y_train, y_test,
    output_dir="data"
)
# Creates: data/train_processed.csv, data/test_processed.csv
```

---

### Model Training Module

**Module:** `src/pipeline/training.py`

Configure and launch SageMaker training jobs.

#### Class: `ModelTrainer`

Main class for training multiple SageMaker models.

##### Constructor

```python
ModelTrainer(bucket, prefix="sagemaker-housing", region="us-east-1")
```

**Parameters:**
- `bucket` (str): S3 bucket name
- `prefix` (str): S3 prefix for organizing files. Default: `"sagemaker-housing"`
- `region` (str): AWS region. Default: `"us-east-1"`

**Example:**
```python
from src.pipeline.training import ModelTrainer

trainer = ModelTrainer(
    bucket="my-sagemaker-bucket",
    prefix="housing-project",
    region="us-west-2"
)
```

##### Methods

###### `get_data_paths()`

Get S3 paths for training and validation data.

**Returns:**
- `tuple`: `(train_path, validation_path)`
  - Both are S3 URIs (e.g., `s3://bucket/prefix/data/train`)

**Example:**
```python
train_path, val_path = trainer.get_data_paths()
print(f"Training data: {train_path}")
```

###### `train_xgboost(instance_type="ml.m5.xlarge", tune=False)`

Train XGBoost model using SageMaker built-in algorithm.

**Parameters:**
- `instance_type` (str): EC2 instance type. Default: `"ml.m5.xlarge"`
- `tune` (bool): Run hyperparameter tuning. Default: `False`

**Returns:**
- `sagemaker.tuner.HyperparameterTuner` or `sagemaker.estimator.Estimator`

**Hyperparameters:**
- `max_depth`: Maximum tree depth (default: 5)
- `eta`: Learning rate (default: 0.2)
- `subsample`: Subsample ratio (default: 0.7)
- `colsample_bytree`: Column sample ratio (default: 0.7)
- `num_round`: Number of boosting rounds (default: 100)
- `objective`: "reg:squarederror"

**Tuning Ranges:**
- `max_depth`: 3-10
- `eta`: 0.05-0.5
- `subsample`: 0.5-1.0
- `colsample_bytree`: 0.5-1.0
- `num_round`: 50-200

**Example:**
```python
# Training without tuning
xgb_estimator = trainer.train_xgboost(instance_type="ml.m5.xlarge")

# Training with hyperparameter tuning
xgb_tuner = trainer.train_xgboost(tune=True)
```

###### `train_knn(instance_type="ml.m5.xlarge", tune=False)`

Train K-Nearest Neighbors model.

**Parameters:**
- `instance_type` (str): EC2 instance type. Default: `"ml.m5.xlarge"`
- `tune` (bool): Run hyperparameter tuning. Default: `False`

**Returns:**
- `sagemaker.tuner.HyperparameterTuner` or `sagemaker.estimator.Estimator`

**Hyperparameters:**
- `k`: Number of neighbors (default: 10)
- `sample_size`: Training sample size (default: 10000)
- `predictor_type`: "regressor"

**Tuning Ranges:**
- `k`: 3-50
- `sample_size`: 5000-20000

**Example:**
```python
knn_estimator = trainer.train_knn(tune=True)
```

###### `train_sklearn_gbm(instance_type="ml.m5.xlarge", tune=False)`

Train scikit-learn Gradient Boosting Machine with custom script.

**Parameters:**
- `instance_type` (str): EC2 instance type. Default: `"ml.m5.xlarge"`
- `tune` (bool): Run hyperparameter tuning. Default: `False`

**Returns:**
- `sagemaker.tuner.HyperparameterTuner` or `sagemaker.sklearn.SKLearn`

**Hyperparameters:**
- `n_estimators`: Number of boosting stages (default: 100)
- `max_depth`: Maximum tree depth (default: 5)
- `learning_rate`: Learning rate (default: 0.1)

**Tuning Ranges:**
- `n_estimators`: 50-200
- `max_depth`: 3-10
- `learning_rate`: 0.01-0.3

**Example:**
```python
gbm_estimator = trainer.train_sklearn_gbm(
    instance_type="ml.m5.2xlarge",
    tune=True
)
```

#### Convenience Functions

##### `setup_training_jobs(sagemaker_session, role, models, bucket, prefix, region)`

Set up estimators for multiple models.

**Parameters:**
- `sagemaker_session`: SageMaker session object
- `role` (str): IAM role ARN for SageMaker
- `models` (list): List of model names (e.g., `["xgboost", "knn"]`)
- `bucket` (str): S3 bucket name
- `prefix` (str): S3 prefix
- `region` (str): AWS region

**Returns:**
- `dict`: Dictionary mapping model names to estimator objects

**Example:**
```python
estimators = setup_training_jobs(
    sagemaker_session=sess,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    models=["xgboost", "knn", "sklearn-gbm"],
    bucket="my-bucket",
    prefix="housing",
    region="us-east-1"
)
```

##### `train_models(estimators, train_data, validation_data, wait=True, mock=False)`

Train all configured models.

**Parameters:**
- `estimators` (dict): Dictionary of model estimators
- `train_data` (str): S3 path to training data
- `validation_data` (str): S3 path to validation data
- `wait` (bool): Wait for training to complete. Default: `True`
- `mock` (bool): Return mock results. Default: `False`

**Returns:**
- `dict`: Dictionary mapping model names to training job names

**Example:**
```python
job_names = train_models(
    estimators=estimators,
    train_data="s3://bucket/data/train",
    validation_data="s3://bucket/data/validation",
    wait=True,
    mock=False
)
print(f"Training jobs: {job_names}")
```

---

### Model Comparison Module

**Module:** `src/pipeline/model_comparison.py`

Compare trained models and select the best performer.

#### Functions

##### `compare_models(job_names, sagemaker_client, mock=False)`

Compare models based on training metrics.

**Parameters:**
- `job_names` (dict): Dictionary mapping model names to job names
- `sagemaker_client`: Boto3 SageMaker client
- `mock` (bool): Use mock metrics. Default: `False`

**Returns:**
- `dict`: Comparison results with structure:
```python
{
    "best_model": "xgboost",
    "best_job": "xgboost-2024-01-15-10-30-45",
    "best_metrics": {
        "rmse": 0.5000,
        "mse": 0.2500,
        "mae": 0.4000
    },
    "all_results": {
        "xgboost": {
            "job_name": "xgboost-2024-01-15-10-30-45",
            "metrics": {...}
        },
        ...
    }
}
```

**Metric Priority:**
1. RMSE (Root Mean Squared Error) - Primary
2. MSE (Mean Squared Error) - Secondary
3. MAE (Mean Absolute Error) - Tertiary

**Example:**
```python
import boto3
from src.pipeline.model_comparison import compare_models

sagemaker_client = boto3.client('sagemaker')

results = compare_models(
    job_names={"xgboost": "xgb-job-123", "knn": "knn-job-456"},
    sagemaker_client=sagemaker_client,
    mock=False
)

print(f"Best model: {results['best_model']}")
print(f"RMSE: {results['best_metrics']['rmse']}")
```

##### `save_comparison_results(results, output_file="model_comparison.json")`

Save comparison results to JSON file.

**Parameters:**
- `results` (dict): Comparison results from `compare_models()`
- `output_file` (str): Path to output JSON file. Default: `"model_comparison.json"`

**Returns:**
- `str`: Path to saved file

**Example:**
```python
from src.pipeline.model_comparison import save_comparison_results

output_path = save_comparison_results(
    results=results,
    output_file="results/comparison.json"
)
print(f"Results saved to: {output_path}")
```

##### `extract_metrics(job_description)`

Extract metrics from SageMaker job description.

**Parameters:**
- `job_description` (dict): SageMaker DescribeTrainingJob response

**Returns:**
- `dict`: Extracted metrics (RMSE, MSE, MAE, R²)

**Example:**
```python
from src.pipeline.model_comparison import extract_metrics

job_desc = sagemaker_client.describe_training_job(
    TrainingJobName="xgboost-job-123"
)
metrics = extract_metrics(job_desc)
print(f"Metrics: {metrics}")
```

---

### Batch Deployment Module

**Module:** `src/pipeline/batch_deployment.py`

Deploy models for batch inference using SageMaker Batch Transform.

#### Functions

##### `deploy_batch_transform(model_name, job_name, test_data_s3, output_s3, sagemaker_session, role, instance_type="ml.m5.xlarge")`

Deploy model for batch inference.

**Parameters:**
- `model_name` (str): Display name for the model
- `job_name` (str): Training job name to deploy
- `test_data_s3` (str): S3 URI to test data
- `output_s3` (str): S3 URI for predictions output
- `sagemaker_session`: SageMaker session object
- `role` (str): IAM role ARN
- `instance_type` (str): Instance type for transform job. Default: `"ml.m5.xlarge"`

**Returns:**
- `dict`: Deployment results with structure:
```python
{
    "model_name": "best-model",
    "transform_job_name": "transform-job-123",
    "output_path": "s3://bucket/output/",
    "status": "Completed"
}
```

**Example:**
```python
from src.pipeline.batch_deployment import deploy_batch_transform
import sagemaker

sess = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/SageMakerRole"

result = deploy_batch_transform(
    model_name="xgboost-best",
    job_name="xgboost-2024-01-15-10-30-45",
    test_data_s3="s3://bucket/data/test",
    output_s3="s3://bucket/predictions/",
    sagemaker_session=sess,
    role=role,
    instance_type="ml.m5.2xlarge"
)

print(f"Predictions saved to: {result['output_path']}")
```

##### `create_model(model_name, job_name, sagemaker_client, role)`

Create SageMaker model from training job.

**Parameters:**
- `model_name` (str): Name for the model
- `job_name` (str): Training job name
- `sagemaker_client`: Boto3 SageMaker client
- `role` (str): IAM role ARN

**Returns:**
- `str`: Model ARN

**Example:**
```python
import boto3

sagemaker_client = boto3.client('sagemaker')
model_arn = create_model(
    model_name="my-model",
    job_name="training-job-123",
    sagemaker_client=sagemaker_client,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

##### `monitor_transform_job(transform_job_name, sagemaker_client)`

Monitor batch transform job progress.

**Parameters:**
- `transform_job_name` (str): Transform job name
- `sagemaker_client`: Boto3 SageMaker client

**Returns:**
- `str`: Final job status ("Completed", "Failed", "Stopped")

**Example:**
```python
status = monitor_transform_job(
    transform_job_name="transform-job-123",
    sagemaker_client=sagemaker_client
)
print(f"Job status: {status}")
```

---

## Configuration

### Environment Variables

The pipeline respects the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` |
| `AWS_ACCESS_KEY_ID` | AWS access key | (from AWS config) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | (from AWS config) |
| `SAGEMAKER_ROLE` | IAM role ARN for SageMaker | (must be provided) |

### AWS Credentials

The pipeline uses standard AWS credential resolution:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM role (if running on EC2/ECS/Lambda)

### S3 Structure

Expected S3 structure:

```
s3://bucket/
└── prefix/
    ├── data/
    │   ├── train/
    │   │   └── train_processed.csv
    │   └── validation/
    │       └── test_processed.csv
    ├── models/
    │   └── model-name-timestamp/
    │       └── model.tar.gz
    └── predictions/
        └── test_processed.csv.out
```

---

## Usage Examples

### Example 1: Complete Pipeline with Python API

```python
import boto3
import sagemaker
from src.pipeline import data_preparation, training, model_comparison, batch_deployment

# Step 1: Prepare data
X_train, X_test, y_train, y_test = data_preparation.generate_and_prepare_data()
train_file, test_file = data_preparation.save_data_for_sagemaker(
    X_train, X_test, y_train, y_test
)

# Step 2: Upload to S3
sess = sagemaker.Session()
bucket = "my-sagemaker-bucket"
train_s3 = sess.upload_data(train_file, bucket=bucket, key_prefix="data/train")
test_s3 = sess.upload_data(test_file, bucket=bucket, key_prefix="data/validation")

# Step 3: Train models
trainer = training.ModelTrainer(bucket=bucket)
trainer.role = "arn:aws:iam::123456789012:role/SageMakerRole"

xgb = trainer.train_xgboost(tune=False)
xgb.fit({"train": train_s3, "validation": test_s3})

# Step 4: Compare models
sagemaker_client = boto3.client('sagemaker')
results = model_comparison.compare_models(
    job_names={"xgboost": xgb.latest_training_job.name},
    sagemaker_client=sagemaker_client
)

# Step 5: Deploy best model
deployment = batch_deployment.deploy_batch_transform(
    model_name=results['best_model'],
    job_name=results['best_job'],
    test_data_s3=test_s3,
    output_s3=f"s3://{bucket}/predictions/",
    sagemaker_session=sess,
    role=trainer.role
)

print(f"Predictions: {deployment['output_path']}")
```

### Example 2: Mock Mode for Testing

```python
from src.pipeline import data_preparation, training, model_comparison

# Prepare data (uses synthetic data)
X_train, X_test, y_train, y_test = data_preparation.generate_and_prepare_data(
    synthetic=True
)

# Train in mock mode (no AWS required)
mock_results = training.train_models(
    estimators={},
    train_data="",
    validation_data="",
    mock=True
)

# Compare with mock metrics
results = model_comparison.compare_models(
    job_names=mock_results,
    sagemaker_client=None,
    mock=True
)

print(f"Best model: {results['best_model']}")
```

### Example 3: Hyperparameter Tuning

```python
from src.pipeline.training import ModelTrainer

trainer = ModelTrainer(bucket="my-bucket")
trainer.role = "arn:aws:iam::123456789012:role/SageMakerRole"

# Run tuning for XGBoost
xgb_tuner = trainer.train_xgboost(
    instance_type="ml.m5.xlarge",
    tune=True
)

# Configure tuning job
train_path, val_path = trainer.get_data_paths()
xgb_tuner.fit(
    inputs={
        "train": train_path,
        "validation": val_path
    },
    wait=True
)

# Get best training job
best_job = xgb_tuner.best_training_job()
print(f"Best training job: {best_job}")
```

### Example 4: Custom Model Configuration

```python
from sagemaker.estimator import Estimator
import sagemaker

sess = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/SageMakerRole"

# Get XGBoost container
container = sagemaker.image_uris.retrieve(
    "xgboost",
    sess.boto_region_name,
    version="1.5-1"
)

# Custom estimator configuration
custom_xgb = Estimator(
    container,
    role=role,
    instance_count=1,
    instance_type="ml.c5.2xlarge",  # Compute-optimized
    volume_size=30,
    max_run=3600,
    use_spot_instances=True,  # Use spot instances for cost savings
    max_wait=7200,
    sagemaker_session=sess
)

# Set custom hyperparameters
custom_xgb.set_hyperparameters(
    objective="reg:squarederror",
    num_round=200,
    max_depth=8,
    eta=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train
custom_xgb.fit({"train": train_s3, "validation": val_s3})
```

---

## Error Handling

### Common Exceptions

#### `RuntimeError`

Raised when SageMaker SDK is not available or required resources are missing.

```python
try:
    trainer = ModelTrainer(bucket="my-bucket")
    trainer.train_xgboost()
except RuntimeError as e:
    print(f"Error: {e}")
    # Use mock mode or install SageMaker SDK
```

#### `botocore.exceptions.ClientError`

Raised for AWS API errors (permissions, resource not found, etc.).

```python
import botocore

try:
    results = compare_models(job_names, sagemaker_client)
except botocore.exceptions.ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'ResourceNotFound':
        print("Training job not found")
    elif error_code == 'AccessDenied':
        print("Check IAM permissions")
```

#### `ValueError`

Raised for invalid parameters or configurations.

```python
try:
    trainer.train_xgboost(instance_type="invalid-type")
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

---

## API Versioning

This API follows semantic versioning:

- **Current Version**: 1.0.0
- **Compatibility**: SageMaker Python SDK >= 2.0.0
- **Python**: 3.8+

### Deprecation Policy

- Deprecated features will be marked with warnings for 2 minor versions
- Breaking changes only occur in major version updates
- All deprecations are documented in CHANGELOG.md

---

## Additional Resources

- [SageMaker Python SDK Documentation](https://sagemaker.readthedocs.io/)
- [Boto3 SageMaker Reference](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html)
- [Architecture Documentation](ARCHITECTURE.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
