"""
Configuration file for Fake News Detection project
Centralized settings for data paths, model parameters, and training configs
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory
ROOT_DIR = Path(__file__).parent.parent
PROJECT_NAME = "Fake_News_Detection"

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Model paths
MODELS_DIR = ROOT_DIR / "models"
TOKENIZER_DIR = MODELS_DIR / "tokenizer"

# Results paths
RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"

# Notebooks
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

class DataConfig:
    """Data processing configuration"""

    # File names
    RAW_DATA_FILE = "data.csv"
    TRAIN_FILE = "train.csv"
    VAL_FILE = "val.csv"
    TEST_FILE = "test.csv"
    SAMPLE_FILE = "sample_1000.csv"

    # Full paths
    RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILE
    TRAIN_PATH = PROCESSED_DATA_DIR / TRAIN_FILE
    VAL_PATH = PROCESSED_DATA_DIR / VAL_FILE
    TEST_PATH = PROCESSED_DATA_DIR / TEST_FILE
    SAMPLE_PATH = SAMPLE_DATA_DIR / SAMPLE_FILE

    # Column names
    TEXT_COLUMN = "content"
    LABEL_COLUMN = "label"
    CLEANED_TEXT_COLUMN = "cleaned_content"

    # Split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Random seed for reproducibility
    RANDOM_SEED = 42

    # Preprocessing options
    REMOVE_URLS = True
    REMOVE_MENTIONS = True
    REMOVE_HASHTAGS = True
    LOWERCASE = True
    REMOVE_EXTRA_SPACES = True


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

class ModelConfig:
    """Model architecture and training configuration"""

    # Model selection - UPDATED FOR ROBERTA
    MODEL_NAME = "roberta-base"  # Changed from bert-base-uncased to roberta-base
    NUM_LABELS = 2  # Binary classification (Real=0, Fake=1)

    # Tokenization
    MAX_LENGTH = 256  # Maximum sequence length (256 for speed, 512 for accuracy)
    PADDING = "max_length"
    TRUNCATION = True

    # Training parameters
    BATCH_SIZE = 16  # Reduced from 32 for RoBERTa (uses more memory)
    LEARNING_RATE = 2e-5  # Standard for RoBERTa fine-tuning
    NUM_EPOCHS = 3  # Usually 3-5 epochs is sufficient
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Optimizer
    OPTIMIZER = "adamw"
    EPSILON = 1e-8

    # Learning rate scheduler
    SCHEDULER = "linear"  # Options: linear, cosine

    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Model saving
    SAVE_STRATEGY = "epoch"  # Save after each epoch
    SAVE_TOTAL_LIMIT = 3  # Keep only best 3 checkpoints
    LOAD_BEST_MODEL_AT_END = True

    # Evaluation
    EVALUATION_STRATEGY = "epoch"  # Evaluate after each epoch
    METRIC_FOR_BEST_MODEL = "f1"  # Use F1-score to select best model

    # Model file names
    BEST_MODEL_NAME = "best_model.pt"
    BASELINE_MODEL_NAME = "baseline_model.pkl"
    FINAL_MODEL_NAME = "final_model.pt"


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training environment configuration"""

    # Device
    USE_CUDA = True  # Set to False to use CPU only
    
    # Mixed precision training (faster on modern GPUs)
    USE_FP16 = True

    # Gradient accumulation (simulate larger batch size)
    GRADIENT_ACCUMULATION_STEPS = 2  # Increased for RoBERTa

    # Gradient clipping (prevent exploding gradients)
    MAX_GRAD_NORM = 1.0

    # Logging
    LOGGING_STEPS = 100  # Log every N steps
    LOGGING_DIR = RESULTS_DIR / "logs"

    # Reproducibility
    SEED = 42
    DETERMINISTIC = True

    # DataLoader
    NUM_WORKERS = 4  # Number of workers for data loading
    PIN_MEMORY = True  # Faster data transfer to GPU


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

class EvaluationConfig:
    """Evaluation and metrics configuration"""

    # Metrics to compute
    METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Confusion matrix
    SAVE_CONFUSION_MATRIX = True

    # Classification report
    SAVE_CLASSIFICATION_REPORT = True

    # Predictions
    SAVE_PREDICTIONS = True
    PREDICTIONS_FILE = "predictions.csv"

    # Metrics file
    METRICS_FILE = "metrics.json"

    # ROC curve
    SAVE_ROC_CURVE = True
    ROC_CURVE_FILE = "roc_curve.png"


# ============================================================================
# API CONFIGURATION (for Phase 3)
# ============================================================================

class APIConfig:
    """FastAPI configuration"""

    # API settings
    API_TITLE = "Fake News Detection API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "API for detecting fake news using RoBERTa"

    # Server
    HOST = "0.0.0.0"
    PORT = 8000
    RELOAD = True  # Auto-reload on code changes (dev only)

    # CORS
    ALLOW_ORIGINS = ["http://localhost:3000"]  # Next.js frontend
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["*"]
    ALLOW_HEADERS = ["*"]

    # MongoDB
    MONGODB_URL = "mongodb://localhost:27017"
    DATABASE_NAME = "fake_news_db"
    COLLECTION_NAME = "predictions"

    # Model loading
    MODEL_PATH = MODELS_DIR / "roberta"  # Updated path
    TOKENIZER_PATH = MODEL_PATH


# ============================================================================
# BASELINE MODEL CONFIGURATION
# ============================================================================

class BaselineConfig:
    """Baseline model (TF-IDF + Logistic Regression) configuration"""

    # TF-IDF parameters
    MAX_FEATURES = 10000
    MIN_DF = 5
    MAX_DF = 0.8
    NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

    # Logistic Regression
    SOLVER = "lbfgs"
    MAX_ITER = 1000
    C = 1.0  # Regularization strength

    # Model file
    TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.pkl"
    BASELINE_MODEL_FILE = "baseline_logistic_regression.pkl"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        MODELS_DIR,
        MODELS_DIR / "roberta",  # Add RoBERTa model directory
        MODELS_DIR / "baseline",  # Add baseline model directory
        TOKENIZER_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        METRICS_DIR / "roberta",  # Add RoBERTa metrics directory
        METRICS_DIR / "baseline",  # Add baseline metrics directory
        PREDICTIONS_DIR,
        PREDICTIONS_DIR / "roberta",  # Add RoBERTa predictions directory
        PREDICTIONS_DIR / "baseline",  # Add baseline predictions directory
        VISUALIZATIONS_DIR,
        VISUALIZATIONS_DIR / "roberta",  # Add RoBERTa visualizations directory
        VISUALIZATIONS_DIR / "baseline",  # Add baseline visualizations directory
        TrainingConfig.LOGGING_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("All directories created successfully.")


def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\nProject Root: {ROOT_DIR}")
    print(f"\nData Configuration:")
    print(f"   - Raw Data: {DataConfig.RAW_DATA_PATH}")
    print(f"   - Train/Val/Test: {DataConfig.TRAIN_RATIO}/{DataConfig.VAL_RATIO}/{DataConfig.TEST_RATIO}")
    print(f"\nModel Configuration:")
    print(f"   - Model: {ModelConfig.MODEL_NAME}")
    print(f"   - Max Length: {ModelConfig.MAX_LENGTH}")
    print(f"   - Batch Size: {ModelConfig.BATCH_SIZE}")
    print(f"   - Learning Rate: {ModelConfig.LEARNING_RATE}")
    print(f"   - Epochs: {ModelConfig.NUM_EPOCHS}")
    print(f"\nTraining Configuration:")
    print(f"   - FP16: {TrainingConfig.USE_FP16}")
    print(f"   - Gradient Accumulation: {TrainingConfig.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   - Seed: {TrainingConfig.SEED}")
    print("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create directories
    create_directories()

    # Print configuration
    print_config()