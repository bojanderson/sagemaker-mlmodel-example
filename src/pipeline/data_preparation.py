"""
Generate and prepare sample data for the ML pipeline.
Uses California housing dataset as a simple example.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_sample_data(output_dir='data', use_synthetic=False):
    """
    Generate sample dataset for ML pipeline.
    Uses California housing dataset or synthetic data if offline.
    
    Args:
        output_dir: Directory to save the data files
        use_synthetic: If True, generate synthetic data instead of fetching real data
    
    Returns:
        tuple: (train_df, test_df, feature_names, target_name)
    """
    if use_synthetic:
        # Generate synthetic housing data
        np.random.seed(42)
        n_samples = 20640
        
        # Generate features that mimic California housing dataset
        df = pd.DataFrame({
            'MedInc': np.random.uniform(0.5, 15, n_samples),
            'HouseAge': np.random.uniform(1, 52, n_samples),
            'AveRooms': np.random.uniform(1, 20, n_samples),
            'AveBedrms': np.random.uniform(1, 10, n_samples),
            'Population': np.random.uniform(3, 35682, n_samples),
            'AveOccup': np.random.uniform(1, 50, n_samples),
            'Latitude': np.random.uniform(32, 42, n_samples),
            'Longitude': np.random.uniform(-124, -114, n_samples),
        })
        
        # Generate target with some correlation to features
        df['MedHouseVal'] = (
            0.4 * df['MedInc'] + 
            0.01 * df['HouseAge'] + 
            0.02 * df['AveRooms'] +
            np.random.normal(0, 0.5, n_samples)
        ).clip(0.15, 5.0)
        
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
    else:
        # Try to load California housing dataset
        try:
            from sklearn.datasets import fetch_california_housing
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            feature_names = housing.feature_names
        except Exception as e:
            print(f"Could not fetch California housing data: {e}")
            print("Falling back to synthetic data...")
            return generate_sample_data(output_dir=output_dir, use_synthetic=True)
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Features: {feature_names}")
    print(f"Target: MedHouseVal")
    
    return train_df, test_df, feature_names, 'MedHouseVal'


def preprocess_for_sagemaker(input_file, output_file, target_column='MedHouseVal'):
    """
    Preprocess data for SageMaker training.
    Moves target column to the first position (required by some SageMaker algorithms).
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        target_column: Name of target column
    """
    df = pd.read_csv(input_file)
    
    # Move target column to first position
    cols = df.columns.tolist()
    cols.remove(target_column)
    cols = [target_column] + cols
    df = df[cols]
    
    # Save without header (SageMaker XGBoost and KNN prefer this format)
    df.to_csv(output_file, header=False, index=False)
    print(f"Preprocessed data saved to {output_file}")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    train_df, test_df, feature_names, target_name = generate_sample_data()
    
    # Preprocess for SageMaker
    preprocess_for_sagemaker('data/train.csv', 'data/train_processed.csv')
    preprocess_for_sagemaker('data/test.csv', 'data/test_processed.csv')
    
    print("\nData generation and preprocessing complete!")
    print("Files created:")
    print("  - data/train.csv (with headers)")
    print("  - data/test.csv (with headers)")
    print("  - data/train_processed.csv (headerless, target first)")
    print("  - data/test_processed.csv (headerless, target first)")
