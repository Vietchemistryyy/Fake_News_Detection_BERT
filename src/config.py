"""
Configuration for training and fine-tuning
"""

import os

# ==================== Model Configuration ====================
MODEL_NAME = "roberta-base"  # For English
PHOBERT_MODEL_NAME = "vinai/phobert-base"  # For Vietnamese

# ==================== Training Configuration ====================
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# ==================== Data Configuration ====================
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE = os.path.join(DATA_DIR, "val.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# ==================== Model Saving ====================
MODEL_SAVE_DIR = "models"
BERT_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, "BERT")
PHOBERT_MODEL_DIR = os.path.join(MODEL_SAVE_DIR, "PhoBERT")

# ==================== Labels ====================
LABEL_MAP = {
    "real": 0,
    "fake": 1
}
NUM_LABELS = 2

# ==================== Training Settings ====================
SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 100
SAVE_STEPS = 500
EVAL_STEPS = 500

# ==================== Device ====================
DEVICE = "cuda"  # Will auto-detect in code

# ==================== Dropout ====================
DROPOUT_RATE = 0.1
ATTENTION_DROPOUT = 0.1

# ==================== Early Stopping ====================
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_DELTA = 0.001
