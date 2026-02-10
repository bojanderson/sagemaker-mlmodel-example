"""
Model comparison and selection logic.
Compares performance of different models and selects the best one.
"""
import json
import boto3
from typing import Dict, List, Tuple
import pandas as pd


class ModelComparator:
    """Compare and select the best model from training results."""
    
    def __init__(self, sagemaker_client=None):
        """
        Initialize ModelComparator.
        
        Args:
            sagemaker_client: Boto3 SageMaker client (optional)
        """
        if sagemaker_client is None:
            try:
                self.sagemaker_client = boto3.client('sagemaker')
            except Exception:
                # No credentials available, but that's OK for mock mode
                self.sagemaker_client = None
        else:
            self.sagemaker_client = sagemaker_client
    
    def get_training_job_metrics(self, job_name: str) -> Dict:
        """
        Get metrics from a training job.
        
        Args:
            job_name: Name of the training job
            
        Returns:
            dict: Metrics from the training job
        """
        try:
            response = self.sagemaker_client.describe_training_job(
                TrainingJobName=job_name
            )
            
            metrics = {}
            if 'FinalMetricDataList' in response:
                for metric in response['FinalMetricDataList']:
                    metrics[metric['MetricName']] = metric['Value']
            
            return {
                'job_name': job_name,
                'status': response['TrainingJobStatus'],
                'metrics': metrics,
                'model_artifacts': response.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
            }
        except Exception as e:
            print(f"Error getting metrics for {job_name}: {e}")
            return None
    
    def get_tuning_job_results(self, tuning_job_name: str) -> List[Dict]:
        """
        Get results from a hyperparameter tuning job.
        
        Args:
            tuning_job_name: Name of the tuning job
            
        Returns:
            list: List of training job results sorted by performance
        """
        try:
            response = self.sagemaker_client.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=tuning_job_name
            )
            
            # Get best training job
            best_job = response.get('BestTrainingJob', {})
            
            # List all training jobs
            jobs_response = self.sagemaker_client.list_training_jobs_for_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=tuning_job_name,
                SortBy='FinalObjectiveMetricValue',
                SortOrder='Ascending',
                MaxResults=100
            )
            
            results = []
            for job_summary in jobs_response.get('TrainingJobSummaries', []):
                job_result = {
                    'job_name': job_summary['TrainingJobName'],
                    'status': job_summary['TrainingJobStatus'],
                    'objective_metric': job_summary.get('FinalHyperParameterTuningJobObjectiveMetric', {}).get('Value'),
                    'tuned_hyperparameters': job_summary.get('TunedHyperParameters', {}),
                }
                results.append(job_result)
            
            return results
        except Exception as e:
            print(f"Error getting tuning results for {tuning_job_name}: {e}")
            return []
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Compare models and select the best one.
        
        Args:
            model_results: Dictionary mapping model names to their results
                          Format: {'xgboost': {'metrics': {...}, ...}, ...}
        
        Returns:
            tuple: (best_model_name, best_model_info)
        """
        print("\n=== Model Comparison ===")
        
        comparison = []
        for model_name, result in model_results.items():
            metrics = result.get('metrics', {})
            
            # Extract relevant metric (RMSE, MSE, etc.)
            # Different models may have different metric names
            metric_value = None
            metric_name = None
            
            if 'validation:rmse' in metrics:
                metric_value = metrics['validation:rmse']
                metric_name = 'RMSE'
            elif 'test:mse' in metrics:
                metric_value = metrics['test:mse']
                metric_name = 'MSE'
            elif 'validation:mae' in metrics:
                metric_value = metrics['validation:mae']
                metric_name = 'MAE'
            
            if metric_value is not None:
                comparison.append({
                    'model': model_name,
                    'metric_name': metric_name,
                    'metric_value': metric_value,
                    'result': result
                })
                print(f"{model_name}: {metric_name} = {metric_value:.4f}")
        
        if not comparison:
            raise ValueError("No valid metrics found for comparison")
        
        # Select best model (lowest error metric)
        best = min(comparison, key=lambda x: x['metric_value'])
        
        print(f"\n✓ Best model: {best['model']} ({best['metric_name']} = {best['metric_value']:.4f})")
        
        return best['model'], best['result']
    
    def save_comparison_results(self, comparison_data: Dict, output_file: str):
        """
        Save comparison results to file.
        
        Args:
            comparison_data: Dictionary with comparison results
            output_file: Path to output JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        print(f"\nComparison results saved to {output_file}")


def create_mock_model_results():
    """Create mock model results for testing."""
    return {
        'xgboost': {
            'job_name': 'xgboost-2024-01-01-00-00-00-000',
            'status': 'Completed',
            'metrics': {
                'validation:rmse': 0.52,
                'train:rmse': 0.48,
            },
            'model_artifacts': 's3://bucket/models/xgboost/model.tar.gz'
        },
        'knn': {
            'job_name': 'knn-2024-01-01-00-00-00-000',
            'status': 'Completed',
            'metrics': {
                'test:mse': 0.31,  # ~0.56 RMSE
                'train:mse': 0.28,
            },
            'model_artifacts': 's3://bucket/models/knn/model.tar.gz'
        },
        'sklearn-gbm': {
            'job_name': 'sklearn-gbm-2024-01-01-00-00-00-000',
            'status': 'Completed',
            'metrics': {
                'validation:rmse': 0.50,
                'validation:mae': 0.35,
                'validation:r2': 0.82,
            },
            'model_artifacts': 's3://bucket/models/sklearn-gbm/model.tar.gz'
        },
    }


if __name__ == "__main__":
    # Example usage with mock data
    comparator = ModelComparator()
    mock_results = create_mock_model_results()
    
    best_model, best_result = comparator.compare_models(mock_results)
    
    comparison_data = {
        'best_model': best_model,
        'best_model_info': best_result,
        'all_models': mock_results,
    }
    
    comparator.save_comparison_results(comparison_data, 'model_comparison.json')
