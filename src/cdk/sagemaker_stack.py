"""
CDK Stack for SageMaker ML Pipeline Infrastructure.
Defines S3 buckets, IAM roles, and SageMaker resources.
"""
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    RemovalPolicy,
    CfnOutput,
)
from constructs import Construct


class SageMakerMLStack(Stack):
    """CDK Stack for SageMaker ML Pipeline."""
    
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Create S3 bucket for data and models
        self.data_bucket = s3.Bucket(
            self, "MLDataBucket",
            bucket_name=f"sagemaker-ml-data-{self.account}-{self.region}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )
        
        # Create IAM role for SageMaker execution
        self.sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )
        
        # Grant S3 permissions to SageMaker role
        self.data_bucket.grant_read_write(self.sagemaker_role)
        
        # Add additional permissions for CloudWatch Logs
        self.sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=["*"],
            )
        )
        
        # Add ECR permissions for pulling SageMaker containers
        self.sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                resources=["*"],
            )
        )
        
        # Create SageMaker Notebook Instance (optional, for interactive work)
        notebook_role = iam.Role(
            self, "NotebookRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )
        
        self.data_bucket.grant_read_write(notebook_role)
        
        # Uncomment to create notebook instance
        # notebook_instance = sagemaker.CfnNotebookInstance(
        #     self, "MLNotebook",
        #     instance_type="ml.t3.medium",
        #     role_arn=notebook_role.role_arn,
        #     notebook_instance_name="sagemaker-ml-notebook",
        # )
        
        # Outputs
        CfnOutput(
            self, "DataBucketName",
            value=self.data_bucket.bucket_name,
            description="S3 bucket for ML data and models",
        )
        
        CfnOutput(
            self, "SageMakerRoleArn",
            value=self.sagemaker_role.role_arn,
            description="IAM role ARN for SageMaker execution",
        )
        
        CfnOutput(
            self, "NotebookRoleArn",
            value=notebook_role.role_arn,
            description="IAM role ARN for SageMaker notebooks",
        )
