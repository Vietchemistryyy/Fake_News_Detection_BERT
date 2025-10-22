"""
Fake News Detection - Source Package
"""

__version__ = "1.0.0"
__author__ = "Fake News Detection Team"

# Import main classes and functions for easy access
try:
    from .config import DataConfig, ModelConfig, TrainingConfig
    from .model import BaselineModel
    from .train import train_baseline_model
    from .evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve
    from .dataset import create_dataset_from_dataframe
    from .preprocessing import preprocess_pipeline
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")