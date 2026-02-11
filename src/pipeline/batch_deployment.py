"""
Batch inference deployment using SageMaker Batch Transform.
"""

import boto3
import time

try:
    import sagemaker
    from sagemaker.model import Model

    SAGEMAKER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SAGEMAKER_AVAILABLE = False
    sagemaker = None


class BatchDeployer:
    """Deploy model for batch inference using SageMaker Batch Transform."""

    def __init__(self, role, bucket, prefix="sagemaker-housing", region="us-east-1"):
        """
        Initialize BatchDeployer.

        Args:
            role: IAM role ARN for SageMaker
            bucket: S3 bucket name
            prefix: S3 prefix
            region: AWS region
        """
        self.role = role
        self.bucket = bucket
        self.prefix = prefix
        self.region = region

        if SAGEMAKER_AVAILABLE:
            self.sess = sagemaker.Session()
        else:
            self.sess = None

        self.sagemaker_client = boto3.client("sagemaker", region_name=region)

    def create_model(
        self, model_name, model_data, image_uri, model_framework="xgboost"
    ):
        """
        Create a SageMaker model.

        Args:
            model_name: Name for the model
            model_data: S3 path to model artifacts (model.tar.gz)
            image_uri: Container image URI
            model_framework: Framework name (for naming)

        Returns:
            Model object
        """
        print(f"Creating model: {model_name}")

        model = Model(
            image_uri=image_uri,
            model_data=model_data,
            role=self.role,
            name=model_name,
            sagemaker_session=self.sess,
        )

        return model

    def create_batch_transform_job(
        self,
        model_name,
        input_path,
        output_path,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        content_type="text/csv",
        split_type="Line",
        accept="text/csv",
    ):
        """
        Create a batch transform job.

        Args:
            model_name: Name of the model to use
            input_path: S3 path to input data
            output_path: S3 path for output predictions
            instance_type: EC2 instance type
            instance_count: Number of instances
            content_type: Content type of input data
            split_type: How to split input data
            accept: Content type for output

        Returns:
            Transform job name
        """
        transform_job_name = f"{model_name}-batch-{int(time.time())}"

        print(f"Creating batch transform job: {transform_job_name}")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")

        try:
            self.sagemaker_client.create_transform_job(
                TransformJobName=transform_job_name,
                ModelName=model_name,
                TransformInput={
                    "DataSource": {
                        "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_path}
                    },
                    "ContentType": content_type,
                    "SplitType": split_type,
                },
                TransformOutput={
                    "S3OutputPath": output_path,
                    "Accept": accept,
                    "AssembleWith": "Line",
                },
                TransformResources={
                    "InstanceType": instance_type,
                    "InstanceCount": instance_count,
                },
            )

            print(f"Transform job created: {transform_job_name}")
            return transform_job_name

        except Exception as e:
            print(f"Error creating transform job: {e}")
            raise

    def wait_for_transform_job(self, job_name, poll_interval=30):
        """
        Wait for a batch transform job to complete.

        Args:
            job_name: Name of the transform job
            poll_interval: Seconds between status checks
        """
        print(f"\nWaiting for transform job {job_name} to complete...")

        while True:
            response = self.sagemaker_client.describe_transform_job(
                TransformJobName=job_name
            )

            status = response["TransformJobStatus"]
            print(f"Status: {status}")

            if status in ["Completed", "Failed", "Stopped"]:
                if status == "Completed":
                    print("✓ Transform job completed successfully!")
                    print(
                        f"  Output location: {response['TransformOutput']['S3OutputPath']}"
                    )
                else:
                    print(f"✗ Transform job {status}")
                    if "FailureReason" in response:
                        print(f"  Reason: {response['FailureReason']}")
                break

            time.sleep(poll_interval)

    def deploy_best_model_for_batch(
        self, model_name, model_data, image_uri, test_data_path
    ):
        """
        Deploy the best model for batch inference.

        Args:
            model_name: Name for the model
            model_data: S3 path to model artifacts
            image_uri: Container image URI
            test_data_path: S3 path to test data

        Returns:
            Transform job name
        """
        # Create model
        model = self.create_model(model_name, model_data, image_uri)

        # Create model in SageMaker
        try:
            model._create_sagemaker_model(instance_type="ml.m5.xlarge")
        except Exception as e:
            print(f"Note: {e}")

        # Create batch transform job
        output_path = f"s3://{self.bucket}/{self.prefix}/batch-output"
        job_name = self.create_batch_transform_job(
            model_name=model_name,
            input_path=test_data_path,
            output_path=output_path,
        )

        return job_name


def get_image_uri_for_model(model_type, region="us-east-1"):
    """
    Get the appropriate container image URI for a model type.

    Args:
        model_type: Type of model ('xgboost', 'knn', 'sklearn-gbm')
        region: AWS region

    Returns:
        Image URI string
    """
    if not SAGEMAKER_AVAILABLE:
        # Return placeholder URIs for testing
        uris = {
            "xgboost": f"{region}.dkr.ecr.amazonaws.com/xgboost:latest",
            "knn": f"{region}.dkr.ecr.amazonaws.com/knn:latest",
            "sklearn-gbm": f"{region}.dkr.ecr.amazonaws.com/sklearn:latest",
        }
        return uris.get(model_type, f"{region}.dkr.ecr.amazonaws.com/unknown:latest")

    if model_type == "xgboost":
        return sagemaker.image_uris.retrieve("xgboost", region, version="1.5-1")
    elif model_type == "knn":
        return sagemaker.image_uris.retrieve("knn", region)
    elif model_type == "sklearn-gbm":
        return sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    print("Batch deployment module loaded successfully")
    print("Use BatchDeployer class to deploy models for batch inference")
