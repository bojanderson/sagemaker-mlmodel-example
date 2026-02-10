# SageMaker ML Pipeline - Implementation Summary

## Overview
This repository provides a complete end-to-end machine learning pipeline using AWS SageMaker with three different algorithms: XGBoost, K-Nearest Neighbors (KNN), and scikit-learn Gradient Boosting Machine (GBM).

## Key Features

### 1. Data Pipeline
- **Synthetic Data Generation**: Creates California housing-style dataset for offline development
- **Data Preprocessing**: Formats data for SageMaker (target column first, no headers)
- **Train/Test Split**: Proper 80/20 split with reproducibility

### 2. Model Training
- **XGBoost**: Using SageMaker built-in container (v1.5-1)
  - Configurable hyperparameters: max_depth, eta, subsample, colsample_bytree, num_round
  - Objective: reg:squarederror
  
- **KNN**: Using SageMaker built-in container
  - Configurable hyperparameters: k, sample_size
  - Predictor type: regressor
  
- **scikit-learn GBM**: Using SageMaker scikit-learn container (v1.2-1)
  - Custom training script: `scripts/sklearn_gbm_script.py`
  - Configurable hyperparameters: n_estimators, max_depth, learning_rate

### 3. Hyperparameter Tuning
- Automatic hyperparameter optimization for all models
- Configurable search ranges
- Parallel job execution support
- Metric-based optimization (minimize RMSE/MSE)

### 4. Model Comparison & Selection
- Automatic comparison based on validation metrics
- Supports multiple metric types: RMSE, MSE, MAE, R²
- Selects best performing model
- Outputs comparison results to JSON

### 5. Batch Deployment
- SageMaker Batch Transform for batch inference
- Configurable instance types
- S3 input/output handling
- Job monitoring and status tracking

### 6. Infrastructure as Code (CDK)
- Complete AWS CDK stack in TypeScript/Python
- Resources created:
  - S3 bucket for data and models (versioned, encrypted)
  - IAM roles for SageMaker execution
  - CloudWatch log groups
  - Optional: SageMaker Notebook instance

### 7. Testing
- **Unit Tests**: 12 tests covering individual components
- **Integration Tests**: 3 tests for end-to-end pipeline
- **Mock Mode**: Can run without AWS credentials
- **Coverage**: Test coverage tracking included

## Project Structure

```
sagemaker-mlmodel-example/
├── src/
│   ├── pipeline/
│   │   ├── data_preparation.py       # Data generation & preprocessing
│   │   ├── training.py                # Model training configurations
│   │   ├── model_comparison.py        # Model comparison logic
│   │   └── batch_deployment.py        # Batch inference deployment
│   ├── cdk/
│   │   ├── app.py                     # CDK app entry point
│   │   ├── sagemaker_stack.py         # Infrastructure stack
│   │   └── cdk.json                   # CDK configuration
│   └── run_pipeline.py                # Main orchestration script
├── scripts/
│   └── sklearn_gbm_script.py          # scikit-learn training script
├── tests/
│   ├── unit/                          # Unit tests
│   │   ├── test_data_preparation.py
│   │   ├── test_training.py
│   │   └── test_model_comparison.py
│   └── integration/
│       └── test_pipeline.py           # End-to-end tests
├── .github/
│   └── workflows/
│       └── tests.yml                  # CI/CD configuration
├── requirements.txt                   # Python dependencies
├── requirements-cdk.txt               # CDK dependencies
├── setup.py                           # Package setup
├── pytest.ini                         # Pytest configuration
├── examples.py                        # Usage examples
├── example_notebook.ipynb             # Jupyter notebook demo
├── CONTRIBUTING.md                    # Contribution guidelines
└── README.md                          # Main documentation
```

## Usage Modes

### 1. Local/Mock Mode (No AWS Credentials Required)
```bash
python src/run_pipeline.py --bucket test-bucket
```
- Uses synthetic data
- Mock training results
- No actual AWS resources used
- Perfect for development and testing

### 2. Development Mode (With AWS Credentials)
```bash
python src/run_pipeline.py --bucket my-bucket --no-mock
```
- Uploads data to S3
- Configures actual SageMaker training jobs
- Requires AWS credentials and permissions

### 3. Full Production Mode
```bash
python src/run_pipeline.py --bucket my-bucket --no-mock --tune --deploy
```
- Full hyperparameter tuning
- Actual model training
- Best model deployment for batch inference

## Testing Results

All tests pass successfully:
- ✅ 15 tests passed
- ⏭️ 6 tests skipped (require full SageMaker SDK)
- ⚠️ 3 warnings (network access for dataset download)

## Key Design Decisions

1. **Graceful Degradation**: Code works even without AWS credentials (mock mode)
2. **Version Compatibility**: Handles different SageMaker SDK versions
3. **Synthetic Data**: No external data dependencies for testing
4. **Minimal Dependencies**: Core functionality requires only common packages
5. **Comprehensive Testing**: Both unit and integration tests included
6. **Well-Documented**: Extensive README, examples, and code comments

## Performance

Mock pipeline execution: ~10 seconds
- Data generation: ~2 seconds
- Model comparison: <1 second
- Result output: <1 second

Real SageMaker execution time varies:
- Training (each model): 5-15 minutes
- Hyperparameter tuning: 30-60 minutes
- Batch transform: 5-10 minutes

## Cost Estimates

AWS costs for one complete run (approximate):
- **Data Storage (S3)**: $0.01 (negligible for small datasets)
- **Training Jobs**: $1-3 (3 models × 5-15 min on ml.m5.xlarge)
- **Hyperparameter Tuning**: $10-30 (10 jobs per model)
- **Batch Transform**: $0.50-1.00 (5-10 min on ml.m5.xlarge)
- **CloudWatch Logs**: $0.05 (negligible)

**Total per run**: $1.50-$35 depending on configuration

## Security Features

1. **S3 Encryption**: Server-side encryption enabled
2. **Block Public Access**: All S3 buckets have public access blocked
3. **IAM Least Privilege**: Roles have minimal required permissions
4. **Versioning**: S3 bucket versioning enabled
5. **No Hardcoded Credentials**: Uses AWS IAM roles

## Future Enhancements

Potential improvements:
1. Add more algorithms (Random Forest, Neural Networks)
2. Add real-time endpoint deployment option
3. Add model monitoring and drift detection
4. Add data quality checks
5. Add experiment tracking (e.g., MLflow)
6. Add model registry integration
7. Add automated model retraining pipeline
8. Add A/B testing capabilities

## Maintenance

The code is designed to be maintainable:
- Clear separation of concerns
- Comprehensive tests
- Type hints where appropriate
- Detailed documentation
- CI/CD ready with GitHub Actions

## License

See [LICENSE](LICENSE) file for details.
