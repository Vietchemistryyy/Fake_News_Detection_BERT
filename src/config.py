"""
Configuration file for Fake News Detection project
Centralized settings for data paths, model parameters, and training configs
UPDATED: Optimized for DeBERTa-v3-base with enhanced training settings
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory
ROOT_DIR = Path(__file__).parent.parent
PROJECT_NAME = "Fake_News_Detection_DeBERTa"

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Model paths
MODELS_DIR = ROOT_DIR / "models"
TOKENIZER_DIR = MODELS_DIR / "tokenizer"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

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
# MODEL CONFIGURATION - OPTIMIZED FOR DEBERTA-V3
# ============================================================================

class ModelConfig:
    """Model architecture and training configuration"""

    # ✨ NEW: Model selection - DeBERTa-v3-base
    MODEL_NAME = "microsoft/deberta-v3-base"  # CHANGED from roberta-base
    
    # Alternative models (comment/uncomment to switch):
    # MODEL_NAME = "microsoft/deberta-v3-large"  # Needs Colab Pro
    # MODEL_NAME = "roberta-large"  # Alternative
    # MODEL_NAME = "google/electra-base-discriminator"  # Fast & efficient
    
    NUM_LABELS = 2  # Binary classification (Real=0, Fake=1)

    # ✨ OPTIMIZED: Tokenization for DeBERTa
    MAX_LENGTH = 384  # Increased from 256 (DeBERTa handles longer sequences better)
    PADDING = "max_length"
    TRUNCATION = True

    # ✨ OPTIMIZED: Training parameters for DeBERTa-v3
    BATCH_SIZE = 16  # Reduced for DeBERTa (larger than RoBERTa)
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 32
    LEARNING_RATE = 2e-05  # Increased slightly (DeBERTa is more stable)
    NUM_EPOCHS = 10  # Increased from 5 (DeBERTa benefits from more epochs)
    WARMUP_RATIO = 0.1  # 10% of training for warmup
    WEIGHT_DECAY = 0.01  # Standard for DeBERTa
    DROPOUT_RATE = 0.1  # DeBERTa's default dropout
    
    # ✨ NEW: Label smoothing (reduces overfitting)
    LABEL_SMOOTHING_FACTOR = 0.0
    
    # Optimizer
    OPTIMIZER = "adamw"
    EPSILON = 1e-8
    MAX_GRAD_NORM = 1.0  # Gradient clipping

    # ✨ IMPROVED: Learning rate scheduler
    SCHEDULER = "cosine_with_restarts"  # Better than linear
    NUM_CYCLES = 2  # For cosine with restarts

    # ✨ IMPROVED: Early stopping
    EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for 3 epochs
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Model saving
    SAVE_STRATEGY = "epoch"
    SAVE_TOTAL_LIMIT = 3  # Keep only best 3 checkpoints
    LOAD_BEST_MODEL_AT_END = True

    # Evaluation
    EVALUATION_STRATEGY = "epoch"
    METRIC_FOR_BEST_MODEL = "f1"  # Use F1-score
    GREATER_IS_BETTER = True

    # Model file names
    BEST_MODEL_NAME = "best_deberta_model"
    BASELINE_MODEL_NAME = "baseline_model.pkl"
    FINAL_MODEL_NAME = "final_deberta_model"


# ============================================================================
# TRAINING CONFIGURATION - OPTIMIZED FOR COLAB
# ============================================================================

class TrainingConfig:
    """Training environment configuration"""

    # Device
    USE_CUDA = True
    DEVICE = "cuda" if USE_CUDA else "cpu"

    # ✨ OPTIMIZED: Mixed precision training (essential for DeBERTa)
    USE_FP16 = True  # Reduces memory by ~50%
    FP16_OPT_LEVEL = "O1"  # Automatic mixed precision

    # ✨ IMPROVED: Gradient accumulation
    GRADIENT_ACCUMULATION_STEPS = 2  # Simulate batch size of 32

    # Gradient clipping
    MAX_GRAD_NORM = 1.0

    # ✨ IMPROVED: Logging
    LOGGING_STEPS = 50  # Log every 50 steps (more frequent)
    LOGGING_DIR = RESULTS_DIR / "logs"
    LOGGING_FIRST_STEP = True

    # Reproducibility
    SEED = 42
    DETERMINISTIC = True

    # ✨ OPTIMIZED: DataLoader for Colab
    NUM_WORKERS = 2  # Colab works best with 2 workers
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2  # Prefetch 2 batches

    # ✨ NEW: Memory optimization
    GRADIENT_CHECKPOINTING = False  # Disable gradient checkpointing for stability
    OPTIM = "adamw_torch"  # Use PyTorch's AdamW (faster on GPU)


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
    
    # ✨ NEW: Per-epoch evaluation
    EVAL_ACCUMULATION_STEPS = 10  # Accumulate predictions (saves memory)


# ============================================================================
# BASELINE MODEL CONFIGURATION
# ============================================================================

class BaselineConfig:
    """Baseline model (TF-IDF + Logistic Regression) configuration"""

    # TF-IDF parameters
    MAX_FEATURES = 10000
    MIN_DF = 5
    MAX_DF = 0.8
    NGRAM_RANGE = (1, 2)

    # Logistic Regression
    SOLVER = "lbfgs"
    MAX_ITER = 1000
    C = 1.0

    # Model files
    TFIDF_VECTORIZER_FILE = "tfidf_vectorizer.pkl"
    BASELINE_MODEL_FILE = "baseline_logistic_regression.pkl"


# ============================================================================
# COLAB-SPECIFIC CONFIGURATION
# ============================================================================

class ColabConfig:
    """Google Colab specific settings"""
    
    # ✨ NEW: Colab environment detection
    IS_COLAB = 'COLAB_GPU' in os.environ or 'google.colab' in str(globals())
    
    # ✨ NEW: Memory management
    CLEAR_CACHE_EVERY_N_STEPS = 100  # Clear CUDA cache periodically
    
    # ✨ NEW: Checkpointing
    SAVE_CHECKPOINT_TO_DRIVE = True  # Save to Google Drive
    DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/fake_news_checkpoints"
    
    # ✨ NEW: Tensorboard
    USE_TENSORBOARD = True
    TENSORBOARD_DIR = "/content/drive/MyDrive/fake_news_logs"
    
    # ✨ NEW: Resource monitoring
    LOG_GPU_MEMORY = True
    LOG_EVERY_N_STEPS = 50


# ============================================================================
# ADVANCED TRAINING CONFIGURATION
# ============================================================================

class AdvancedConfig:
    """Advanced training techniques"""
    
    # ✨ NEW: Data augmentation
    USE_DATA_AUGMENTATION = False  # Set True to enable
    AUGMENTATION_RATIO = 0.2  # Augment 20% of training data
    
    # ✨ NEW: Ensemble settings
    USE_ENSEMBLE = False
    ENSEMBLE_MODELS = [
        "microsoft/deberta-v3-base",
        "roberta-large",
    ]
    ENSEMBLE_WEIGHTS = [0.6, 0.4]
    
    # ✨ NEW: Learning rate finder
    USE_LR_FINDER = False  # Run LR finder before training
    LR_FINDER_STEPS = 100
    
    # ✨ NEW: Stochastic Weight Averaging
    USE_SWA = False  # Improves generalization
    SWA_START_EPOCH = 7
    SWA_LR = 5e-6


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
        TOKENIZER_DIR,
        CHECKPOINT_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        VISUALIZATIONS_DIR,
        TrainingConfig.LOGGING_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("✅ All directories created successfully.")


def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\n🚀 Project: {PROJECT_NAME}")
    print(f"📁 Root: {ROOT_DIR}")
    
    print(f"\n📊 Data Configuration:")
    print(f"   - Raw Data: {DataConfig.RAW_DATA_PATH}")
    print(f"   - Splits: {DataConfig.TRAIN_RATIO}/{DataConfig.VAL_RATIO}/{DataConfig.TEST_RATIO}")
    
    print(f"\n🤖 Model Configuration:")
    print(f"   - Model: {ModelConfig.MODEL_NAME}")
    print(f"   - Max Length: {ModelConfig.MAX_LENGTH}")
    print(f"   - Batch Size: {ModelConfig.BATCH_SIZE}")
    print(f"   - Gradient Accumulation: {ModelConfig.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   - Effective Batch Size: {ModelConfig.BATCH_SIZE * ModelConfig.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   - Learning Rate: {ModelConfig.LEARNING_RATE}")
    print(f"   - Epochs: {ModelConfig.NUM_EPOCHS}")
    print(f"   - Warmup Ratio: {ModelConfig.WARMUP_RATIO}")
    print(f"   - Label Smoothing: {ModelConfig.LABEL_SMOOTHING_FACTOR}")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   - Device: {TrainingConfig.DEVICE}")
    print(f"   - FP16: {TrainingConfig.USE_FP16}")
    print(f"   - Gradient Checkpointing: {TrainingConfig.GRADIENT_CHECKPOINTING}")
    print(f"   - Seed: {TrainingConfig.SEED}")
    
    if ColabConfig.IS_COLAB:
        print(f"\n☁️  Colab Configuration:")
        print(f"   - Environment: Google Colab")
        print(f"   - Save to Drive: {ColabConfig.SAVE_CHECKPOINT_TO_DRIVE}")
        print(f"   - TensorBoard: {ColabConfig.USE_TENSORBOARD}")
    
    print("=" * 80)


def get_model_info():
    """Get information about the selected model"""
    model_info = {
        "microsoft/deberta-v3-base": {
            "params": "184M",
            "memory": "~3GB",
            "speed": "Medium",
            "accuracy": "High (95-96%)",
            "recommended_batch": 16
        },
        "microsoft/deberta-v3-large": {
            "params": "435M",
            "memory": "~6GB",
            "speed": "Slow",
            "accuracy": "Very High (96-97%)",
            "recommended_batch": 8
        },
        "roberta-base": {
            "params": "125M",
            "memory": "~2GB",
            "speed": "Fast",
            "accuracy": "Good (93-94%)",
            "recommended_batch": 32
        },
        "roberta-large": {
            "params": "355M",
            "memory": "~5GB",
            "speed": "Medium",
            "accuracy": "High (94-95%)",
            "recommended_batch": 16
        }
    }
    
    return model_info.get(ModelConfig.MODEL_NAME, {})


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create directories
    create_directories()

    # Print configuration
    print_config()
    
    # Print model info
    print("\n📊 Model Information:")
    info = get_model_info()
    for key, value in info.items():
        print(f"   - {key}: {value}")