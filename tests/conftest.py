"""Test configuration and fixtures."""

import os
import sys

# Set up mock AWS credentials before any imports
# This prevents SageMaker SDK from failing during import
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
