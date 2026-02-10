"""Unit tests for data preparation module."""
import pytest
import os
import pandas as pd
import tempfile
import shutil
from pipeline.data_preparation import generate_sample_data, preprocess_for_sagemaker


class TestDataPreparation:
    """Test suite for data preparation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_generate_sample_data(self):
        """Test generating sample data."""
        train_df, test_df, feature_names, target_name = generate_sample_data(
            output_dir=self.temp_dir,
            use_synthetic=True  # Use synthetic data for testing
        )
        
        # Check that data frames are not empty
        assert len(train_df) > 0
        assert len(test_df) > 0
        
        # Check that target column exists
        assert target_name in train_df.columns
        assert target_name in test_df.columns
        
        # Check that feature names are present
        for feature in feature_names:
            assert feature in train_df.columns
            assert feature in test_df.columns
        
        # Check that files were created
        assert os.path.exists(os.path.join(self.temp_dir, 'train.csv'))
        assert os.path.exists(os.path.join(self.temp_dir, 'test.csv'))
        
        # Check train/test split ratio (approximately 80/20)
        total = len(train_df) + len(test_df)
        train_ratio = len(train_df) / total
        assert 0.75 < train_ratio < 0.85
    
    def test_preprocess_for_sagemaker(self):
        """Test preprocessing for SageMaker format."""
        # First generate some sample data
        generate_sample_data(output_dir=self.temp_dir, use_synthetic=True)
        
        input_file = os.path.join(self.temp_dir, 'train.csv')
        output_file = os.path.join(self.temp_dir, 'train_processed.csv')
        
        # Preprocess
        df = preprocess_for_sagemaker(input_file, output_file, target_column='MedHouseVal')
        
        # Check that output file was created
        assert os.path.exists(output_file)
        
        # Read the processed file
        processed_df = pd.read_csv(output_file, header=None)
        
        # Check that target is in first column
        # (We can't check exact values, but we can check structure)
        assert processed_df.shape[0] > 0
        assert processed_df.shape[1] == 9  # 8 features + 1 target
        
        # Check that first column is the target (should be continuous values)
        target_col = processed_df.iloc[:, 0]
        assert target_col.min() > 0  # Housing prices should be positive
        assert target_col.max() < 10  # California housing target is in range 0-5
    
    def test_preprocess_preserves_data_size(self):
        """Test that preprocessing preserves the number of rows."""
        # Generate sample data
        train_df, _, _, _ = generate_sample_data(output_dir=self.temp_dir, use_synthetic=True)
        
        input_file = os.path.join(self.temp_dir, 'train.csv')
        output_file = os.path.join(self.temp_dir, 'train_processed.csv')
        
        # Preprocess
        preprocess_for_sagemaker(input_file, output_file)
        
        # Check that row count is preserved
        processed_df = pd.read_csv(output_file, header=None)
        assert len(processed_df) == len(train_df)
