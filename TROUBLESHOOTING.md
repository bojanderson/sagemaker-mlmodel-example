# Troubleshooting Guide

This guide helps resolve common issues when running the SageMaker ML Pipeline.

## Table of Contents

- [Installation Issues](#installation-issues)
- [AWS Credentials and Permissions](#aws-credentials-and-permissions)
- [Data Preparation Issues](#data-preparation-issues)
- [Training Issues](#training-issues)
- [Model Comparison Issues](#model-comparison-issues)
- [Deployment Issues](#deployment-issues)
- [Testing Issues](#testing-issues)
- [Performance Issues](#performance-issues)
- [CDK Deployment Issues](#cdk-deployment-issues)

---

## Installation Issues

### Issue: `ModuleNotFoundError: No module named 'sagemaker'`

**Problem:** SageMaker SDK not installed.

**Solution:**
```bash
pip install -r requirements.txt
```

If using CDK:
```bash
pip install -r requirements-cdk.txt
```

**Verification:**
```bash
python -c "import sagemaker; print(sagemaker.__version__)"
```

---

### Issue: `ImportError: cannot import name 'HyperparameterTuner'`

**Problem:** Incompatible SageMaker SDK version.

**Solution:**
```bash
pip install --upgrade sagemaker>=2.0.0
```

**Check version:**
```bash
pip show sagemaker
```

---

### Issue: `sklearn.datasets.fetch_california_housing` fails

**Problem:** No network connection or blocked by firewall.

**Solution:** Use synthetic data mode:
```bash
python src/run_pipeline.py --bucket demo-bucket --mock
```

Or generate synthetic data in Python:
```python
from src.pipeline.data_preparation import generate_and_prepare_data

# Uses synthetic data
X_train, X_test, y_train, y_test = generate_and_prepare_data(synthetic=True)
```

---

## AWS Credentials and Permissions

### Issue: `NoCredentialsError: Unable to locate credentials`

**Problem:** AWS credentials not configured.

**Solutions:**

**Option 1: Configure AWS CLI**
```bash
aws configure
```
Enter your:
- AWS Access Key ID
- AWS Secret Access Key
- Default region name (e.g., us-east-1)
- Default output format (json)

**Option 2: Use environment variables**
```bash
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**Option 3: Use mock mode (no credentials needed)**
```bash
python src/run_pipeline.py --bucket demo-bucket --mock
```

**Verification:**
```bash
aws sts get-caller-identity
```

---

### Issue: `AccessDeniedException: User is not authorized to perform`

**Problem:** IAM user/role lacks required permissions.

**Solution:** Attach this policy to your IAM user/role:

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
        "sagemaker:CreateHyperParameterTuningJob",
        "sagemaker:DescribeTrainingJob",
        "sagemaker:DescribeModel",
        "sagemaker:DescribeTransformJob",
        "sagemaker:DescribeHyperParameterTuningJob",
        "sagemaker:ListTrainingJobs",
        "sagemaker:StopTrainingJob",
        "sagemaker:StopTransformJob"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "iam:PassRole"
      ],
      "Resource": "arn:aws:iam::*:role/*SageMaker*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

**Verification:**
```bash
aws sagemaker list-training-jobs --max-results 1
```

---

### Issue: `ValidationException: The role is not valid`

**Problem:** SageMaker execution role doesn't exist or is incorrectly formatted.

**Solution:**

**1. Create SageMaker execution role:**
```bash
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'
```

**2. Attach managed policy:**
```bash
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

**3. Use the role ARN:**
```bash
export SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
```

**Verification:**
```bash
aws iam get-role --role-name SageMakerExecutionRole
```

---

## Data Preparation Issues

### Issue: `FileNotFoundError: [Errno 2] No such file or directory`

**Problem:** Output directory doesn't exist.

**Solution:**
```bash
mkdir -p data
python src/run_pipeline.py --bucket demo-bucket --mock
```

Or in Python:
```python
import os
os.makedirs("data", exist_ok=True)
```

---

### Issue: Data upload to S3 fails with `NoSuchBucket`

**Problem:** S3 bucket doesn't exist.

**Solution:**

**Create bucket:**
```bash
aws s3 mb s3://your-bucket-name --region us-east-1
```

**Or use CDK to create all infrastructure:**
```bash
cd src/cdk
cdk deploy
```

**Verification:**
```bash
aws s3 ls s3://your-bucket-name
```

---

### Issue: `ValueError: could not convert string to float`

**Problem:** Data contains invalid values or wrong format.

**Solution:**

**Check data format:**
```python
import pandas as pd

df = pd.read_csv("train_processed.csv", header=None)
print(df.head())
print(df.dtypes)
print(df.isnull().sum())
```

**Regenerate data:**
```python
from src.pipeline.data_preparation import generate_and_prepare_data

X_train, X_test, y_train, y_test = generate_and_prepare_data(synthetic=True)
```

---

## Training Issues

### Issue: Training job fails with `AlgorithmError`

**Problem:** Data format incompatible with SageMaker algorithm.

**Common causes:**
- Headers in CSV file (SageMaker expects no headers)
- Target column not first
- Non-numeric values

**Solution:**

**Verify data format:**
```bash
head -n 5 train_processed.csv
```

Expected format:
```
target,feature1,feature2,...
2.5,8.32,41.0,...
```

**Regenerate with correct format:**
```python
from src.pipeline.data_preparation import save_data_for_sagemaker

train_file, test_file = save_data_for_sagemaker(
    X_train, X_test, y_train, y_test
)
```

---

### Issue: Training job stuck in "InProgress" status

**Problem:** Instance type not available or resource limits reached.

**Solutions:**

**1. Check job status:**
```python
import boto3

client = boto3.client('sagemaker')
response = client.describe_training_job(TrainingJobName='job-name')
print(response['TrainingJobStatus'])
print(response.get('FailureReason', 'No failure'))
```

**2. Try different instance type:**
```bash
python src/run_pipeline.py --bucket my-bucket --no-mock
# Edit src/run_pipeline.py to use ml.m5.large instead of ml.m5.xlarge
```

**3. Check CloudWatch logs:**
```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

**4. Request service quota increase:**
```bash
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-E3100A6D \
  --desired-value 2
```

---

### Issue: `ResourceLimitExceeded: The account-level service limit`

**Problem:** SageMaker resource limits reached.

**Solution:**

**Check current limits:**
```bash
aws service-quotas list-service-quotas \
  --service-code sagemaker \
  --query 'Quotas[?QuotaName==`ml.m5.xlarge for training job usage`]'
```

**Request increase:**
1. Go to AWS Service Quotas console
2. Select Amazon SageMaker
3. Search for "ml.m5.xlarge for training job usage"
4. Request quota increase

**Or use smaller instance:**
```python
trainer.train_xgboost(instance_type="ml.m5.large")
```

---

### Issue: Training fails with `OutOfMemory` error

**Problem:** Instance memory insufficient for dataset size.

**Solution:**

**1. Use larger instance type:**
```python
trainer.train_xgboost(instance_type="ml.m5.2xlarge")
```

**2. Reduce dataset size:**
```python
from sklearn.model_selection import train_test_split

# Use 50% of data
X_train_small, _, y_train_small, _ = train_test_split(
    X_train, y_train, train_size=0.5, random_state=42
)
```

**3. Adjust model hyperparameters:**
```python
# For XGBoost, reduce max_depth
estimator.set_hyperparameters(max_depth=3)  # instead of 5
```

---

## Model Comparison Issues

### Issue: `KeyError: 'FinalMetricDataList'`

**Problem:** Training job didn't emit metrics.

**Solution:**

**Check training job metrics:**
```python
import boto3

client = boto3.client('sagemaker')
response = client.describe_training_job(TrainingJobName='job-name')

if 'FinalMetricDataList' in response:
    print(response['FinalMetricDataList'])
else:
    print("No metrics available")
    print(f"Job status: {response['TrainingJobStatus']}")
```

**Ensure training completed:**
```python
from src.pipeline.model_comparison import compare_models

results = compare_models(
    job_names=job_names,
    sagemaker_client=client,
    mock=False  # Use real metrics
)
```

---

### Issue: All models have same metrics

**Problem:** Running in mock mode.

**Solution:**
```bash
# Run with real training
python src/run_pipeline.py --bucket my-bucket --no-mock
```

---

### Issue: `model_comparison.json` not created

**Problem:** Permission issues or invalid path.

**Solution:**

**Check write permissions:**
```bash
ls -la model_comparison.json
chmod 644 model_comparison.json  # if exists
```

**Specify full path:**
```python
from src.pipeline.model_comparison import save_comparison_results

save_comparison_results(
    results=results,
    output_file="/full/path/to/model_comparison.json"
)
```

---

## Deployment Issues

### Issue: Batch Transform job fails with `ClientError`

**Problem:** Model artifacts not found or invalid configuration.

**Solution:**

**1. Verify model exists:**
```bash
aws sagemaker describe-model --model-name your-model-name
```

**2. Check S3 model artifacts:**
```bash
aws s3 ls s3://bucket/prefix/models/ --recursive
```

**3. Verify test data path:**
```bash
aws s3 ls s3://bucket/prefix/data/validation/
```

**4. Ensure role has S3 access:**
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject", "s3:PutObject"],
  "Resource": [
    "arn:aws:s3:::bucket/*"
  ]
}
```

---

### Issue: Predictions output format incorrect

**Problem:** Content type or output format misconfigured.

**Solution:**

**Check output in S3:**
```bash
aws s3 cp s3://bucket/predictions/test_processed.csv.out - | head -n 5
```

**Specify content type:**
```python
transformer = model.transformer(
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=output_s3,
    accept='text/csv',  # Specify output format
    assemble_with='Line'  # Assemble by line
)
```

---

### Issue: Batch Transform job is slow

**Problem:** Instance type or batch size not optimized.

**Solutions:**

**1. Use more powerful instance:**
```python
transformer = model.transformer(
    instance_type='ml.c5.2xlarge',  # Compute-optimized
    ...
)
```

**2. Increase instance count:**
```python
transformer = model.transformer(
    instance_count=4,  # Parallel processing
    ...
)
```

**3. Tune batch size:**
```python
transformer = model.transformer(
    max_payload=100,  # MB per request
    max_concurrent_transforms=8,  # Concurrent requests
    ...
)
```

---

## Testing Issues

### Issue: `pytest: command not found`

**Problem:** Pytest not installed.

**Solution:**
```bash
pip install pytest pytest-cov
```

**Verification:**
```bash
pytest --version
```

---

### Issue: Tests fail with import errors

**Problem:** Source code not in Python path.

**Solution:**

**Option 1: Install in development mode**
```bash
pip install -e .
```

**Option 2: Set PYTHONPATH**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
pytest tests/
```

**Option 3: Run from project root**
```bash
cd /path/to/sagemaker-mlmodel-example
pytest tests/
```

---

### Issue: Tests skipped with "requires sagemaker"

**Problem:** Optional dependencies not installed.

**Expected behavior:** Tests that require SageMaker SDK are automatically skipped if not available.

**To run all tests:**
```bash
pip install sagemaker boto3
pytest tests/ -v
```

**To see skipped tests:**
```bash
pytest tests/ -v -rs
```

---

### Issue: Integration tests fail with AWS errors

**Problem:** AWS credentials required for integration tests.

**Solution:**

**Run only unit tests:**
```bash
pytest tests/unit/ -v
```

**Or use mock mode:**
```bash
pytest tests/ -v -k "not integration"
```

**Configure AWS for integration tests:**
```bash
export AWS_DEFAULT_REGION=us-east-1
aws configure
pytest tests/integration/ -v
```

---

## Performance Issues

### Issue: Pipeline execution is slow

**Problem:** Sequential execution or slow instance types.

**Solutions:**

**1. Use faster instance types:**
```python
# For training
trainer.train_xgboost(instance_type="ml.c5.2xlarge")

# For batch transform
transformer = model.transformer(instance_type="ml.c5.2xlarge")
```

**2. Enable spot instances for cost savings:**
```python
estimator = Estimator(
    ...,
    use_spot_instances=True,
    max_wait=3600
)
```

**3. Reduce dataset size for testing:**
```python
# Use smaller sample
X_train_small = X_train[:1000]
y_train_small = y_train[:1000]
```

---

### Issue: High costs

**Problem:** Running expensive instances for extended periods.

**Solutions:**

**1. Use mock mode for development:**
```bash
python src/run_pipeline.py --bucket demo-bucket --mock
```

**2. Use spot instances (70% savings):**
```python
estimator.use_spot_instances = True
estimator.max_wait = 3600
```

**3. Right-size instances:**
- Small datasets: `ml.m5.large`
- Medium datasets: `ml.m5.xlarge`
- Large datasets: `ml.m5.2xlarge`

**4. Clean up resources:**
```bash
# Delete models
aws sagemaker delete-model --model-name model-name

# Delete old data
aws s3 rm s3://bucket/prefix/old-data/ --recursive
```

**5. Monitor costs:**
```bash
# Check SageMaker costs
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --filter file://filter.json
```

---

## CDK Deployment Issues

### Issue: `cdk: command not found`

**Problem:** AWS CDK not installed.

**Solution:**
```bash
npm install -g aws-cdk
```

**Verification:**
```bash
cdk --version
```

---

### Issue: `This stack uses assets, so the toolkit stack must be deployed`

**Problem:** CDK not bootstrapped.

**Solution:**
```bash
cd src/cdk
cdk bootstrap aws://ACCOUNT-ID/REGION
```

Example:
```bash
cdk bootstrap aws://123456789012/us-east-1
```

---

### Issue: CDK deployment fails with CloudFormation errors

**Problem:** Resource conflicts or invalid configurations.

**Solutions:**

**1. View detailed error:**
```bash
cdk deploy --verbose
```

**2. Check CloudFormation events:**
```bash
aws cloudformation describe-stack-events \
  --stack-name SageMakerStack \
  --max-items 10
```

**3. Destroy and redeploy:**
```bash
cdk destroy
cdk deploy
```

**4. Check for resource conflicts:**
```bash
# Check if bucket already exists
aws s3 ls s3://your-bucket-name

# Check if role already exists
aws iam get-role --role-name SageMakerExecutionRole
```

---

## Common Error Messages

### `"Training job failed with status: Failed"`

**Diagnosis:**
```python
import boto3

client = boto3.client('sagemaker')
response = client.describe_training_job(TrainingJobName='job-name')
print(response.get('FailureReason'))
```

**Common causes:**
- Data format issues
- Insufficient memory
- Invalid hyperparameters
- Network connectivity issues

**Solution:** Check CloudWatch logs for detailed error messages.

---

### `"Could not find model data at s3://..."`

**Problem:** Model artifacts not uploaded or wrong path.

**Solution:**

**Check training job outputs:**
```python
response = client.describe_training_job(TrainingJobName='job-name')
print(response['ModelArtifacts']['S3ModelArtifacts'])
```

**Verify S3 path:**
```bash
aws s3 ls s3://bucket/path/to/model/
```

---

### `"Rate exceeded" or throttling errors`

**Problem:** Too many API calls.

**Solution:**

**Add exponential backoff:**
```python
import time
from botocore.exceptions import ClientError

def describe_with_retry(client, job_name, max_retries=5):
    for i in range(max_retries):
        try:
            return client.describe_training_job(TrainingJobName=job_name)
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                wait_time = 2 ** i
                time.sleep(wait_time)
            else:
                raise
```

---

## Getting Additional Help

### Check CloudWatch Logs

**View training job logs:**
```bash
aws logs tail /aws/sagemaker/TrainingJobs --follow --filter-pattern "job-name"
```

**View transform job logs:**
```bash
aws logs tail /aws/sagemaker/TransformJobs --follow
```

### Enable Debug Mode

**Python logging:**
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

**SageMaker verbose output:**
```python
estimator.fit(data, wait=True, logs='All')
```

### Useful Commands

**Check SageMaker service health:**
```bash
aws health describe-events --filter services=SAGEMAKER
```

**List recent training jobs:**
```bash
aws sagemaker list-training-jobs \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 5
```

**Get training job details:**
```bash
aws sagemaker describe-training-job \
  --training-job-name job-name \
  --output table
```

---

## Still Need Help?

If you're still experiencing issues:

1. **Check the logs:** CloudWatch Logs often contain detailed error messages
2. **Review documentation:** [AWS SageMaker Docs](https://docs.aws.amazon.com/sagemaker/)
3. **Search issues:** [GitHub Issues](https://github.com/bojanderson/sagemaker-mlmodel-example/issues)
4. **Report a bug:** Create a new GitHub issue with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version, SageMaker SDK version)
   - Relevant logs

**Provide this information when reporting issues:**

```bash
# System info
python --version
pip list | grep sagemaker
aws --version

# Error details
cat model_comparison.json
tail -n 50 /path/to/logfile

# AWS region
echo $AWS_DEFAULT_REGION
```
