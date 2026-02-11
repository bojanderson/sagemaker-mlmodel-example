# Architecture Documentation

## Overview

This project implements an end-to-end machine learning pipeline on AWS SageMaker that trains multiple models, compares their performance, and deploys the best model for batch inference. The architecture follows a modular design with clear separation of concerns across five main components.

## System Architecture

### High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         User / Developer                           │
└───────────────────┬───────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                      run_pipeline.py (Orchestrator)                │
│  - Argument parsing                                                │
│  - Step coordination                                               │
│  - Error handling                                                  │
└───────────┬───────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────┐
│                        Pipeline Modules                            │
├───────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ Data            │  │ Training     │  │ Model Comparison    │  │
│  │ Preparation     │→ │ Module       │→ │ Module              │  │
│  └─────────────────┘  └──────────────┘  └─────────────────────┘  │
│           │                   │                      │             │
│           ▼                   ▼                      ▼             │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │               Batch Deployment Module                       │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────┬───────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────┐
│                         AWS SageMaker                              │
├───────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐  │
│  │ Training    │  │ Tuning      │  │ Batch Transform          │  │
│  │ Jobs        │  │ Jobs        │  │ Jobs                     │  │
│  └─────────────┘  └─────────────┘  └──────────────────────────┘  │
└───────────┬───────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────┐
│                         AWS Services                               │
├───────────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐                  │
│  │ S3      │    │ IAM      │    │ CloudWatch  │                  │
│  │ Storage │    │ Roles    │    │ Logs        │                  │
│  └─────────┘    └──────────┘    └─────────────┘                  │
└───────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Pipeline Orchestrator (`run_pipeline.py`)

**Purpose**: Main entry point that coordinates all pipeline steps.

**Responsibilities**:
- Parse command-line arguments
- Initialize AWS session and SageMaker client
- Execute pipeline steps in sequence
- Handle errors and provide user feedback
- Support both mock and real execution modes

**Key Features**:
- **Mock Mode**: Runs without AWS credentials for testing
- **Flexible Configuration**: Command-line arguments for all major settings
- **Error Handling**: Comprehensive exception handling with user-friendly messages
- **Progress Tracking**: Clear step-by-step execution feedback

### 2. Data Preparation Module (`data_preparation.py`)

**Purpose**: Load, preprocess, and prepare training/test datasets.

**Data Flow**:
```
California Housing Dataset
    │
    ├─► Load from sklearn
    │
    ├─► Split (80% train / 20% test)
    │
    ├─► Format for SageMaker
    │   - Target column first
    │   - No headers
    │   - CSV format
    │
    └─► Save to local files
         │
         └─► Upload to S3
```

**Key Functions**:
- `generate_and_prepare_data()`: Main entry point
  - Loads California housing dataset
  - Splits into train/test sets
  - Returns feature matrix and target arrays

- `save_data_for_sagemaker()`: Format and save data
  - Formats data for SageMaker (target first, no headers)
  - Saves to CSV files
  - Returns file paths

**Data Format**:
```
target,feature1,feature2,...,feature8
2.5,8.32,41.0,6.98,1.02,322.0,2.56,37.88,-122.23
...
```

### 3. Training Module (`training.py`)

**Purpose**: Configure and launch SageMaker training jobs for multiple models.

**Supported Models**:

| Model | Algorithm | Container | Hyperparameters |
|-------|-----------|-----------|----------------|
| XGBoost | Gradient Boosting | SageMaker Built-in | max_depth, eta, subsample, colsample_bytree, num_round |
| KNN | K-Nearest Neighbors | SageMaker Built-in | k, sample_size |
| scikit-learn GBM | Gradient Boosting | SageMaker sklearn | n_estimators, max_depth, learning_rate |

**Training Flow**:
```
┌────────────────────┐
│ Initialize Models  │
│ - Get containers   │
│ - Set hyperparams  │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│ For each model:    │
│                    │
│ ┌───────────────┐  │
│ │ Create        │  │
│ │ Estimator     │  │
│ └───────┬───────┘  │
│         │          │
│         ▼          │
│ ┌───────────────┐  │
│ │ Fit with      │  │
│ │ train data    │  │
│ └───────┬───────┘  │
│         │          │
│         ▼          │
│ ┌───────────────┐  │
│ │ Wait for      │  │
│ │ completion    │  │
│ └───────────────┘  │
└────────────────────┘
```

**Key Functions**:
- `setup_training_jobs()`: Initialize all model estimators
- `train_models()`: Execute training (mock or real)
- `tune_hyperparameters()`: Run hyperparameter optimization

**Training Modes**:
1. **Mock Mode**: Returns simulated training results instantly
2. **Real Mode**: Launches actual SageMaker training jobs
3. **Tuning Mode**: Runs hyperparameter tuning jobs with multiple trials

### 4. Model Comparison Module (`model_comparison.py`)

**Purpose**: Compare trained models and select the best performer.

**Comparison Flow**:
```
┌──────────────────────┐
│ Retrieve Metrics     │
│ from Training Jobs   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Extract Metrics:     │
│ - RMSE               │
│ - MSE                │
│ - MAE                │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Rank Models by:      │
│ 1. RMSE (primary)    │
│ 2. MSE (secondary)   │
│ 3. MAE (tertiary)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Select Best Model    │
│ Save Comparison JSON │
└──────────────────────┘
```

**Metric Priority**:
1. **RMSE** (Root Mean Squared Error) - Primary metric
2. **MSE** (Mean Squared Error) - Secondary metric
3. **MAE** (Mean Absolute Error) - Tertiary metric

**Key Functions**:
- `compare_models()`: Main comparison logic
  - Retrieves metrics from SageMaker
  - Ranks models by performance
  - Returns best model information

- `save_comparison_results()`: Save results to JSON
  - Model rankings
  - Metrics for each model
  - Best model selection reasoning

**Output Format** (`model_comparison.json`):
```json
{
  "models": [
    {
      "name": "sklearn-gbm",
      "job_name": "sklearn-gbm-2024-01-15-12-30-45",
      "metrics": {
        "rmse": 0.5000,
        "mse": 0.2500,
        "mae": 0.4000
      },
      "rank": 1
    },
    ...
  ],
  "best_model": "sklearn-gbm",
  "selection_criteria": "Lowest RMSE"
}
```

### 5. Batch Deployment Module (`batch_deployment.py`)

**Purpose**: Deploy the best model for batch inference using SageMaker Batch Transform.

**Deployment Flow**:
```
┌────────────────────┐
│ Create Model       │
│ - Best job name    │
│ - Container image  │
│ - IAM role         │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│ Create Transformer │
│ - Instance type    │
│ - Instance count   │
│ - Strategy         │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│ Run Transform Job  │
│ - Input: S3 test   │
│ - Output: S3 pred  │
└──────────┬─────────┘
           │
           ▼
┌────────────────────┐
│ Wait & Monitor     │
│ - Job status       │
│ - CloudWatch logs  │
└────────────────────┘
```

**Key Functions**:
- `deploy_batch_transform()`: Main deployment function
  - Creates SageMaker model
  - Sets up Batch Transform job
  - Monitors completion
  - Returns prediction S3 path

**Configuration**:
- **Instance Type**: ml.m5.xlarge (configurable)
- **Instance Count**: 1 (can scale horizontally)
- **Strategy**: MultiRecord (batch multiple records per request)

## Data Flow

### Complete Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. Data Preparation                          │
├─────────────────────────────────────────────────────────────────┤
│  sklearn dataset → train/test split → format → save locally     │
│                                                   │              │
│                                                   ▼              │
│                              ┌─────────────────────────────┐    │
│                              │  train_processed.csv        │    │
│                              │  test_processed.csv         │    │
│                              └─────────────┬───────────────┘    │
└────────────────────────────────────────────┼────────────────────┘
                                             │
┌────────────────────────────────────────────┼────────────────────┐
│                     2. S3 Upload            ▼                    │
├─────────────────────────────────────────────────────────────────┤
│                    s3://bucket/prefix/data/                      │
│                    ├── train/train_processed.csv                │
│                    └── validation/test_processed.csv            │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                     3. Training ▼                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ XGBoost      │  │ KNN          │  │ scikit-learn GBM     │  │
│  │ Training Job │  │ Training Job │  │ Training Job         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────────────┘  │
│         │                 │                  │                  │
│         ▼                 ▼                  ▼                  │
│  s3://.../model.tar.gz (for each model)                         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                4. Model Compare ▼                                │
├─────────────────────────────────────────────────────────────────┤
│  Retrieve metrics → Compare → Select best → Save JSON           │
│                                              │                   │
│                                              ▼                   │
│                                   model_comparison.json          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│                  5. Deployment  ▼                                │
├─────────────────────────────────────────────────────────────────┤
│  Create model → Batch Transform → Predictions to S3             │
│                                              │                   │
│                                              ▼                   │
│                    s3://bucket/prefix/predictions/               │
└─────────────────────────────────────────────────────────────────┘
```

## AWS Resource Architecture

### Infrastructure Components (CDK Stack)

The project uses AWS CDK to provision all required infrastructure:

```
┌─────────────────────────────────────────────────────────────────┐
│                      CDK Stack (Python)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐                                         │
│  │  S3 Bucket         │                                         │
│  │  - Data storage    │                                         │
│  │  - Model storage   │                                         │
│  │  - Predictions     │                                         │
│  └────────────────────┘                                         │
│                                                                  │
│  ┌────────────────────┐       ┌────────────────────┐           │
│  │  IAM Roles         │       │  CloudWatch        │           │
│  │  - SageMaker       │       │  - Log groups      │           │
│  │  - Notebook        │       │  - Metrics         │           │
│  └────────────────────┘       └────────────────────┘           │
│                                                                  │
│  ┌────────────────────┐                                         │
│  │  SageMaker         │                                         │
│  │  - Notebook (opt)  │                                         │
│  └────────────────────┘                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### S3 Structure

```
s3://my-sagemaker-bucket/
├── sagemaker-housing/              # Default prefix
│   ├── data/
│   │   ├── train/
│   │   │   └── train_processed.csv
│   │   └── validation/
│   │       └── test_processed.csv
│   ├── models/
│   │   ├── xgboost-2024-01-15-10-30-45/
│   │   │   └── model.tar.gz
│   │   ├── knn-2024-01-15-10-35-20/
│   │   │   └── model.tar.gz
│   │   └── sklearn-gbm-2024-01-15-10-40-10/
│   │       ├── model.tar.gz
│   │       └── code/
│   │           └── sklearn_gbm_script.py
│   └── predictions/
│       └── test_processed.csv.out
└── model_comparison.json
```

## Design Decisions

### 1. Mock Mode for Testing

**Decision**: Support running the entire pipeline without AWS credentials.

**Rationale**:
- Enables local development and testing
- Reduces costs during development
- Allows CI/CD integration without AWS setup
- Faster feedback loop for code changes

**Implementation**: Each module checks for `mock=True` flag and returns simulated data.

### 2. Modular Architecture

**Decision**: Separate concerns into five distinct modules.

**Rationale**:
- **Maintainability**: Each module has a single responsibility
- **Testability**: Modules can be tested independently
- **Reusability**: Components can be reused in other pipelines
- **Clarity**: Clear separation makes the codebase easier to understand

### 3. SageMaker Built-in Algorithms

**Decision**: Use SageMaker's built-in containers rather than custom containers.

**Rationale**:
- **Optimized Performance**: AWS-optimized implementations
- **Managed Updates**: AWS handles security patches and updates
- **Reduced Complexity**: No need to manage Docker images
- **Cost Effective**: Pre-built containers are included in SageMaker pricing

### 4. Metric-Based Model Selection

**Decision**: Automatically select best model using RMSE as primary metric.

**Rationale**:
- **Objectivity**: Removes human bias from model selection
- **Reproducibility**: Same metrics always yield same choice
- **Transparency**: Clear criteria for model selection
- **Extensibility**: Easy to add more metrics or change priority

### 5. Batch Transform for Deployment

**Decision**: Use Batch Transform instead of real-time endpoints.

**Rationale**:
- **Cost Efficiency**: No always-on infrastructure
- **Scalability**: Handles large datasets efficiently
- **Simplicity**: Easier to manage than endpoints
- **Use Case Fit**: California housing predictions don't require real-time inference

## Extension Points

### Adding New Models

To add a new model to the pipeline:

1. **Update `training.py`**:
   ```python
   def setup_training_jobs(sagemaker_session, role, models, ...):
       # Add new model configuration
       if 'new-model' in models:
           estimators['new-model'] = NewModelEstimator(...)
   ```

2. **Add hyperparameters** in the model configuration dictionary

3. **Update mock data** in `train_models()` if using mock mode

### Custom Metrics

To add new evaluation metrics:

1. **Update `model_comparison.py`**:
   ```python
   def compare_models(job_names, sagemaker_client, mock=False):
       # Add metric extraction
       metrics['new_metric'] = extract_new_metric(job)
   ```

2. **Update ranking logic** to include new metric

### Alternative Datasets

To use a different dataset:

1. **Update `data_preparation.py`**:
   ```python
   def generate_and_prepare_data():
       # Load your custom dataset
       X, y = load_custom_dataset()
       return X_train, X_test, y_train, y_test
   ```

2. **Ensure data format** matches SageMaker requirements (target first, no headers)

## Performance Considerations

### Training Performance

- **Instance Types**: Current default is `ml.m5.xlarge`
  - For larger datasets, consider `ml.m5.2xlarge` or GPU instances
  - For production workloads, use `ml.c5.xlarge` for cost optimization

- **Parallel Training**: Models train independently and can run in parallel
  - Current implementation: Sequential execution
  - Enhancement: Use threading or async to parallelize

### Batch Transform Performance

- **Instance Count**: Scale horizontally for large datasets
  ```python
  transformer = model.transformer(
      instance_count=4,  # Process data in parallel across 4 instances
      ...
  )
  ```

- **Batch Size**: Tune `max_payload` for optimal throughput
  - Larger payloads = fewer requests = better performance
  - Must fit in instance memory

### Cost Optimization

1. **Use Spot Instances** for training (70% cost reduction)
   ```python
   estimator = sagemaker.estimator.Estimator(
       ...
       use_spot_instances=True,
       max_wait=3600,
   )
   ```

2. **Right-size instances** based on dataset
   - Small datasets (<1GB): ml.m5.large
   - Medium datasets (1-10GB): ml.m5.xlarge
   - Large datasets (>10GB): ml.m5.2xlarge+

3. **Clean up resources** after pipeline completes
   - Delete unused models
   - Archive old training data to Glacier

## Security

### IAM Permissions

Minimum required permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateModel",
        "sagemaker:CreateTransformJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:DescribeModel",
        "sagemaker:DescribeTransformJob"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/*"
      ]
    }
  ]
}
```

### Data Security

- **Encryption**: All S3 data is encrypted at rest (SSE-S3)
- **VPC**: Consider deploying SageMaker in VPC for network isolation
- **Access Logging**: Enable S3 access logs for audit trails

## Monitoring

### CloudWatch Metrics

Key metrics to monitor:

- **Training Jobs**:
  - `TrainingTime`: Duration of training
  - `TrainingDataSize`: Input data size
  - Algorithm-specific metrics (loss, accuracy, etc.)

- **Batch Transform**:
  - `ModelLatency`: Time per prediction
  - `Invocations`: Number of predictions made

### Logging

- **Application Logs**: Python logging to stdout (captured by CloudWatch)
- **SageMaker Logs**: Automatic logging to CloudWatch Logs
- **Custom Metrics**: Use CloudWatch custom metrics for business KPIs

## Testing Strategy

### Unit Tests

Test individual modules in isolation:
- Data preparation logic
- Model configuration
- Metric extraction
- Comparison logic

### Integration Tests

Test the complete pipeline:
- End-to-end mock execution
- S3 upload/download
- SageMaker API interactions (mocked)

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest tests/ -v --cov=src

- name: Test pipeline in mock mode
  run: |
    python src/run_pipeline.py --bucket test-bucket --mock
```

## Future Enhancements

### Planned Improvements

1. **Real-time Endpoints**: Add support for real-time inference
2. **MLOps Integration**: Add MLflow or SageMaker Experiments tracking
3. **Model Registry**: Integrate with SageMaker Model Registry
4. **A/B Testing**: Support deploying multiple model versions
5. **AutoML Integration**: Add SageMaker Autopilot support
6. **Feature Store**: Integrate with SageMaker Feature Store
7. **Pipeline Automation**: Convert to SageMaker Pipelines

### Community Contributions

We welcome contributions in these areas:
- Additional model types (Linear Learner, Neural Networks)
- Real-time deployment options
- Advanced hyperparameter tuning strategies
- Cost optimization techniques
- Performance benchmarking

## References

- [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [AWS CDK Python Reference](https://docs.aws.amazon.com/cdk/api/v2/python/)
- [SageMaker Built-in Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
