"""
Fake News Detection - Source Package
Centralized imports for easy access to all modules
"""

__version__ = "1.0.0"
__author__ = "Fake News Detection Team"

# Import main classes and functions for easy access
try:
    # Configuration
    from .config import (
        DataConfig, 
        ModelConfig, 
        TrainingConfig,
        BaselineConfig,
        EvaluationConfig,
        APIConfig
    )
    
    # Models
    from .model import (
        BaselineModel,
        BertClassifier,
        BertForSequenceClassification,
        create_baseline_model,
        create_bert_model,
        create_model
    )
    
    # Training
    from .train import (
        train_baseline_model,
        train_bert_model,
        BertTrainer,
        save_training_results
    )
    
    # Evaluation
    from .evaluate import (
        evaluate_model,
        compute_metrics,
        compute_extended_metrics,
        plot_confusion_matrix,
        plot_roc_curve,
        plot_precision_recall_curve,
        compare_models,
        save_evaluation_results,
        create_evaluation_report
    )
    
    # Dataset
    from .dataset import (
        FakeNewsDataset,
        create_dataset_from_dataframe,
        create_data_loaders,
        get_sample_batch
    )
    
    # Preprocessing
    from .preprocessing import (
        preprocess_pipeline,
        TextPreprocessor,
        load_raw_data,
        split_data,
        check_data_quality
    )
    
    # Utils
    from .utils import (
        setup_logger,
        Timer,
        set_seed,
        save_json,
        load_json,
        save_csv,
        load_csv,
        get_data_statistics,
        print_statistics
    )
    
    print("✅ Fake News Detection package loaded successfully")
    
except ImportError as e:
    print(f"⚠️  Warning: Some imports failed: {e}")
    print("💡 Some features may not be available")


# Define what gets exported when using "from src import *"
__all__ = [
    # Config
    'DataConfig',
    'ModelConfig', 
    'TrainingConfig',
    'BaselineConfig',
    
    # Models
    'BaselineModel',
    'BertClassifier',
    'BertForSequenceClassification',
    'create_baseline_model',
    'create_bert_model',
    'create_model',
    
    # Training
    'train_baseline_model',
    'train_bert_model',
    'BertTrainer',
    
    # Evaluation
    'evaluate_model',
    'compute_metrics',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'compare_models',
    
    # Dataset
    'FakeNewsDataset',
    'create_dataset_from_dataframe',
    'create_data_loaders',
    
    # Preprocessing
    'preprocess_pipeline',
    'TextPreprocessor',
    
    # Utils
    'setup_logger',
    'Timer',
    'set_seed',
]