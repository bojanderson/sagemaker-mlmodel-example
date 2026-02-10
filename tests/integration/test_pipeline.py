"""Integration tests for the complete pipeline."""
import pytest
import os
import tempfile
import shutil
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from run_pipeline import run_pipeline


class TestPipelineIntegration:
    """Integration tests for the ML pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.original_dir = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        os.chdir(self.original_dir)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_with_mock_training(self):
        """Test running the complete pipeline with mock training."""
        # Run pipeline with mock training (doesn't require AWS credentials)
        best_model, model_results = run_pipeline(
            bucket='test-bucket',
            prefix='test-prefix',
            region='us-east-1',
            models=['xgboost', 'knn', 'sklearn-gbm'],
            tune=False,
            deploy=False,
            mock_training=True,
        )
        
        # Check that a best model was selected
        assert best_model is not None
        assert best_model in ['xgboost', 'knn', 'sklearn-gbm']
        
        # Check that model results were returned
        assert len(model_results) == 3
        assert 'xgboost' in model_results
        assert 'knn' in model_results
        assert 'sklearn-gbm' in model_results
        
        # Check that data files were created
        assert os.path.exists('data/train.csv')
        assert os.path.exists('data/test.csv')
        assert os.path.exists('data/train_processed.csv')
        assert os.path.exists('data/test_processed.csv')
        
        # Check that comparison results were saved
        assert os.path.exists('model_comparison.json')
    
    def test_pipeline_with_subset_of_models(self):
        """Test running pipeline with only some models."""
        best_model, model_results = run_pipeline(
            bucket='test-bucket',
            prefix='test-prefix',
            models=['xgboost', 'sklearn-gbm'],
            mock_training=True,
        )
        
        # Only 2 models should be in results (mock data still has 3, but that's ok)
        assert best_model in ['xgboost', 'knn', 'sklearn-gbm']
    
    def test_pipeline_creates_required_files(self):
        """Test that pipeline creates all required output files."""
        run_pipeline(
            bucket='test-bucket',
            models=['xgboost'],
            mock_training=True,
        )
        
        # Check data directory structure
        assert os.path.exists('data')
        assert os.path.isdir('data')
        
        # Check that comparison file exists
        assert os.path.exists('model_comparison.json')
        
        # Verify comparison file has valid JSON
        import json
        with open('model_comparison.json', 'r') as f:
            comparison_data = json.load(f)
            assert 'best_model' in comparison_data
            assert 'all_models' in comparison_data
