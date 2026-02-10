"""
Scikit-learn Gradient Boosting training script for SageMaker.
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)

    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )

    return parser.parse_args()


def load_data(file_path):
    """Load CSV data (target in first column)."""
    df = pd.read_csv(file_path, header=None)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    return X, y


def train_model(X_train, y_train, X_val, y_val, args):
    """Train Gradient Boosting Regressor."""
    print("Training Gradient Boosting Regressor...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluate on training data
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    print(f"\nTraining metrics:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  R2: {train_r2:.4f}")

    # Evaluate on validation data
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)

    print(f"\nValidation metrics:")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  MAE: {val_mae:.4f}")
    print(f"  R2: {val_r2:.4f}")

    # Log metrics for SageMaker (CloudWatch)
    print(f"\nvalidation:rmse={val_rmse:.4f};")
    print(f"validation:mae={val_mae:.4f};")
    print(f"validation:r2={val_r2:.4f};")

    return model


def save_model(model, model_dir):
    """Save trained model."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    args = parse_args()

    # Load training data
    train_file = os.path.join(args.train, "train_processed.csv")
    X_train, y_train = load_data(train_file)

    # Load validation data
    val_file = os.path.join(args.validation, "test_processed.csv")
    X_val, y_val = load_data(val_file)

    # Train model
    model = train_model(X_train, y_train, X_val, y_val, args)

    # Save model
    save_model(model, args.model_dir)
