"""
Training and hyperparameter tuning for multiple SageMaker models.
Supports XGBoost, KNN, and scikit-learn GBM.
"""

import json

try:
    import sagemaker
    from sagemaker.estimator import Estimator
    from sagemaker.tuner import (
        HyperparameterTuner,
        IntegerParameter,
        ContinuousParameter,
    )
    from sagemaker.sklearn.estimator import SKLearn

    SAGEMAKER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # SageMaker not available or wrong version
    SAGEMAKER_AVAILABLE = False
    sagemaker = None
    print("Warning: SageMaker SDK not fully available. Some features may not work.")


class ModelTrainer:
    """Train and tune multiple SageMaker models."""

    def __init__(self, bucket, prefix="sagemaker-housing", region="us-east-1"):
        """
        Initialize ModelTrainer.

        Args:
            bucket: S3 bucket name for data and models
            prefix: S3 prefix for organizing files
            region: AWS region
        """
        self.bucket = bucket
        self.prefix = prefix
        self.region = region

        if SAGEMAKER_AVAILABLE:
            self.sess = sagemaker.Session()
        else:
            self.sess = None

        # Role will be set manually or from environment
        self.role = None

    def get_data_paths(self):
        """Get S3 paths for training and validation data."""
        train_path = f"s3://{self.bucket}/{self.prefix}/data/train"
        validation_path = f"s3://{self.bucket}/{self.prefix}/data/validation"
        return train_path, validation_path

    def train_xgboost(self, instance_type="ml.m5.xlarge", tune=False):
        """
        Train XGBoost model using SageMaker built-in algorithm.

        Args:
            instance_type: EC2 instance type for training
            tune: Whether to run hyperparameter tuning

        Returns:
            Tuner or Estimator object
        """
        if not SAGEMAKER_AVAILABLE:
            raise RuntimeError("SageMaker SDK not available")

        # Get XGBoost container
        container = sagemaker.image_uris.retrieve(
            "xgboost", self.region, version="1.5-1"
        )

        # Set up estimator
        xgb = Estimator(
            container,
            role=self.role,
            instance_count=1,
            instance_type=instance_type,
            output_path=f"s3://{self.bucket}/{self.prefix}/models/xgboost",
            sagemaker_session=self.sess,
        )

        # Set hyperparameters
        xgb.set_hyperparameters(
            objective="reg:squarederror",
            num_round=100,
            max_depth=5,
            eta=0.2,
            subsample=0.8,
            colsample_bytree=0.8,
        )

        if tune:
            # Define hyperparameter ranges
            hyperparameter_ranges = {
                "max_depth": IntegerParameter(3, 10),
                "eta": ContinuousParameter(0.01, 0.3),
                "subsample": ContinuousParameter(0.5, 1.0),
                "colsample_bytree": ContinuousParameter(0.5, 1.0),
                "num_round": IntegerParameter(50, 200),
            }

            # Create tuner
            tuner = HyperparameterTuner(
                xgb,
                objective_metric_name="validation:rmse",
                objective_type="Minimize",
                hyperparameter_ranges=hyperparameter_ranges,
                max_jobs=10,
                max_parallel_jobs=2,
            )

            return tuner

        return xgb

    def train_knn(self, instance_type="ml.m5.xlarge", tune=False):
        """
        Train KNN model using SageMaker built-in algorithm.

        Args:
            instance_type: EC2 instance type for training
            tune: Whether to run hyperparameter tuning

        Returns:
            Tuner or Estimator object
        """
        # Get KNN container
        container = sagemaker.image_uris.retrieve("knn", self.region)

        # Set up estimator
        knn = Estimator(
            container,
            role=self.role,
            instance_count=1,
            instance_type=instance_type,
            output_path=f"s3://{self.bucket}/{self.prefix}/models/knn",
            sagemaker_session=self.sess,
        )

        # Set hyperparameters
        knn.set_hyperparameters(
            predictor_type="regressor",
            k=5,
            sample_size=10000,
        )

        if tune:
            # Define hyperparameter ranges
            hyperparameter_ranges = {
                "k": IntegerParameter(3, 20),
                "sample_size": IntegerParameter(5000, 20000),
            }

            # Create tuner
            tuner = HyperparameterTuner(
                knn,
                objective_metric_name="test:mse",
                objective_type="Minimize",
                hyperparameter_ranges=hyperparameter_ranges,
                max_jobs=10,
                max_parallel_jobs=2,
            )

            return tuner

        return knn

    def train_sklearn_gbm(self, instance_type="ml.m5.xlarge", tune=False):
        """
        Train scikit-learn GradientBoostingRegressor using SageMaker.

        Args:
            instance_type: EC2 instance type for training
            tune: Whether to run hyperparameter tuning

        Returns:
            Tuner or Estimator object
        """
        # Set up SKLearn estimator
        sklearn = SKLearn(
            entry_point="sklearn_gbm_script.py",
            source_dir="scripts",
            role=self.role,
            instance_type=instance_type,
            framework_version="1.2-1",
            py_version="py3",
            output_path=f"s3://{self.bucket}/{self.prefix}/models/sklearn-gbm",
            sagemaker_session=self.sess,
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
            },
        )

        if tune:
            # Define hyperparameter ranges
            hyperparameter_ranges = {
                "n_estimators": IntegerParameter(50, 200),
                "max_depth": IntegerParameter(3, 10),
                "learning_rate": ContinuousParameter(0.01, 0.3),
            }

            # Create tuner
            tuner = HyperparameterTuner(
                sklearn,
                objective_metric_name="validation:rmse",
                objective_type="Minimize",
                hyperparameter_ranges=hyperparameter_ranges,
                max_jobs=10,
                max_parallel_jobs=2,
            )

            return tuner

        return sklearn

    def run_training_jobs(self, models=["xgboost", "knn", "sklearn-gbm"], tune=False):
        """
        Run training jobs for specified models.

        Args:
            models: List of model names to train
            tune: Whether to run hyperparameter tuning

        Returns:
            dict: Dictionary of model names to estimator/tuner objects
        """
        train_path, validation_path = self.get_data_paths()
        results = {}

        for model_name in models:
            print(f"\nStarting training for {model_name}...")

            if model_name == "xgboost":
                estimator = self.train_xgboost(tune=tune)
                data_channels = {"train": train_path, "validation": validation_path}
            elif model_name == "knn":
                estimator = self.train_knn(tune=tune)
                data_channels = {"train": train_path, "test": validation_path}
            elif model_name == "sklearn-gbm":
                estimator = self.train_sklearn_gbm(tune=tune)
                data_channels = {"train": train_path, "validation": validation_path}
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Note: In a real scenario, you would call estimator.fit(data_channels)
            # For this example, we return the configured estimators
            results[model_name] = {
                "estimator": estimator,
                "data_channels": data_channels,
            }

        return results


def save_training_config(config, output_file="training_config.json"):
    """Save training configuration to file."""
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Training configuration saved to {output_file}")
