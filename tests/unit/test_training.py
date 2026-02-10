"""Unit tests for training module."""

import pytest
from pipeline.training import ModelTrainer, SAGEMAKER_AVAILABLE


class TestModelTrainer:
    """Test suite for model training."""

    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(
            bucket="test-bucket", prefix="test-prefix", region="us-east-1"
        )

        assert trainer.bucket == "test-bucket"
        assert trainer.prefix == "test-prefix"
        assert trainer.region == "us-east-1"

    def test_get_data_paths(self):
        """Test getting S3 data paths."""
        trainer = ModelTrainer(bucket="my-bucket", prefix="my-prefix")

        train_path, validation_path = trainer.get_data_paths()

        assert train_path == "s3://my-bucket/my-prefix/data/train"
        assert validation_path == "s3://my-bucket/my-prefix/data/validation"

    @pytest.mark.skipif(not SAGEMAKER_AVAILABLE, reason="SageMaker SDK not available")
    def test_train_xgboost_returns_estimator(self):
        """Test that train_xgboost returns an estimator."""
        trainer = ModelTrainer(bucket="test-bucket")
        trainer.role = "arn:aws:iam::123456789012:role/test-role"

        estimator = trainer.train_xgboost(tune=False)

        assert estimator is not None
        # Check that hyperparameters are set
        assert hasattr(estimator, "hyperparameters")

    @pytest.mark.skipif(not SAGEMAKER_AVAILABLE, reason="SageMaker SDK not available")
    def test_train_xgboost_with_tuning_returns_tuner(self):
        """Test that train_xgboost with tuning returns a tuner."""
        trainer = ModelTrainer(bucket="test-bucket")
        trainer.role = "arn:aws:iam::123456789012:role/test-role"

        tuner = trainer.train_xgboost(tune=True)

        assert tuner is not None
        # Tuner should have different attributes than estimator
        assert hasattr(tuner, "estimator")

    @pytest.mark.skipif(not SAGEMAKER_AVAILABLE, reason="SageMaker SDK not available")
    def test_train_knn_returns_estimator(self):
        """Test that train_knn returns an estimator."""
        trainer = ModelTrainer(bucket="test-bucket")
        trainer.role = "arn:aws:iam::123456789012:role/test-role"

        estimator = trainer.train_knn(tune=False)

        assert estimator is not None
        assert hasattr(estimator, "hyperparameters")

    @pytest.mark.skipif(not SAGEMAKER_AVAILABLE, reason="SageMaker SDK not available")
    def test_train_sklearn_gbm_returns_estimator(self):
        """Test that train_sklearn_gbm returns an estimator."""
        trainer = ModelTrainer(bucket="test-bucket")
        trainer.role = "arn:aws:iam::123456789012:role/test-role"

        estimator = trainer.train_sklearn_gbm(tune=False)

        assert estimator is not None

    @pytest.mark.skipif(not SAGEMAKER_AVAILABLE, reason="SageMaker SDK not available")
    def test_run_training_jobs_all_models(self):
        """Test running training jobs for all models."""
        trainer = ModelTrainer(bucket="test-bucket")
        trainer.role = "arn:aws:iam::123456789012:role/test-role"

        results = trainer.run_training_jobs(
            models=["xgboost", "knn", "sklearn-gbm"], tune=False
        )

        # Check that all models are in results
        assert "xgboost" in results
        assert "knn" in results
        assert "sklearn-gbm" in results

        # Check that each has estimator and data channels
        for model_name, result in results.items():
            assert "estimator" in result
            assert "data_channels" in result
            assert result["estimator"] is not None

    @pytest.mark.skipif(not SAGEMAKER_AVAILABLE, reason="SageMaker SDK not available")
    def test_run_training_jobs_single_model(self):
        """Test running training job for single model."""
        trainer = ModelTrainer(bucket="test-bucket")
        trainer.role = "arn:aws:iam::123456789012:role/test-role"

        results = trainer.run_training_jobs(models=["xgboost"], tune=False)

        assert len(results) == 1
        assert "xgboost" in results

    def test_run_training_jobs_invalid_model_raises_error(self):
        """Test that invalid model name raises an error."""
        trainer = ModelTrainer(bucket="test-bucket")
        trainer.role = "arn:aws:iam::123456789012:role/test-role"

        with pytest.raises(ValueError, match="Unknown model"):
            trainer.run_training_jobs(models=["invalid-model"], tune=False)
