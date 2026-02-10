#!/usr/bin/env python
"""
Example script showing different ways to use the SageMaker ML pipeline.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline.data_preparation import generate_sample_data, preprocess_for_sagemaker
from pipeline.model_comparison import ModelComparator, create_mock_model_results
from run_pipeline import run_pipeline


def example_1_data_preparation():
    """Example 1: Generate and preprocess data only."""
    print("=" * 60)
    print("Example 1: Data Preparation")
    print("=" * 60)

    # Generate data
    train_df, test_df, features, target = generate_sample_data(use_synthetic=True)

    print(f"\nGenerated {len(train_df)} training samples")
    print(f"Generated {len(test_df)} test samples")
    print(f"Features: {features}")
    print(f"Target: {target}")

    # Show a few samples
    print("\nFirst 3 rows of training data:")
    print(train_df.head(3))


def example_2_model_comparison():
    """Example 2: Compare models using mock results."""
    print("\n" + "=" * 60)
    print("Example 2: Model Comparison")
    print("=" * 60)

    # Create mock results
    mock_results = create_mock_model_results()

    # Compare models
    comparator = ModelComparator()
    best_model, best_result = comparator.compare_models(mock_results)

    print(f"\nSelected best model: {best_model}")
    print(f"Performance: {best_result['metrics']}")


def example_3_full_pipeline():
    """Example 3: Run the complete pipeline."""
    print("\n" + "=" * 60)
    print("Example 3: Full Pipeline")
    print("=" * 60)

    # Run complete pipeline with mock training
    best_model, model_results = run_pipeline(
        bucket="example-bucket",
        prefix="example-prefix",
        models=["xgboost", "knn", "sklearn-gbm"],
        tune=False,
        deploy=False,
        mock_training=True,
    )

    print(f"\n\nPipeline complete!")
    print(f"Best model: {best_model}")
    print(f"Total models trained: {len(model_results)}")


def example_4_custom_models():
    """Example 4: Run pipeline with subset of models."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Model Selection")
    print("=" * 60)

    # Run with only XGBoost and sklearn-gbm
    best_model, model_results = run_pipeline(
        bucket="example-bucket",
        models=["xgboost", "sklearn-gbm"],  # Only two models
        mock_training=True,
    )

    print(f"\n\nSelected models: {list(model_results.keys())}")
    print(f"Best model: {best_model}")


if __name__ == "__main__":
    print("\nSageMaker ML Pipeline Examples\n")

    # Run all examples
    example_1_data_preparation()
    example_2_model_comparison()
    example_3_full_pipeline()
    example_4_custom_models()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review generated files: data/, model_comparison.json")
    print(
        "2. Try with real SageMaker: python src/run_pipeline.py --bucket <bucket> --no-mock"
    )
    print("3. Enable tuning: add --tune flag")
    print("4. Deploy model: add --deploy flag")
