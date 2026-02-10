#!/usr/bin/env python3
"""CDK app entry point."""
import os
import aws_cdk as cdk
from sagemaker_stack import SageMakerMLStack


app = cdk.App()

# Create the SageMaker ML stack
SageMakerMLStack(
    app,
    "SageMakerMLStack",
    description="SageMaker ML Pipeline Infrastructure",
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION', 'us-east-1')
    )
)

app.synth()
