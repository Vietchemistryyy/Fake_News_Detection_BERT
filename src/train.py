"""
Training functions for Fake News Detection models
UPDATED: Enhanced training for DeBERTa-v3 with advanced techniques
Optimized for Google Colab
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import time
import gc
import os

# Try to import transformers
try:
    import transformers
    from transformers import (
        TrainingArguments,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_scheduler
    )
    
    try:
        from transformers import Trainer
    except ImportError:
        from transformers.trainer import Trainer
    
    try:
        from transformers import EarlyStoppingCallback
        from transformers import TrainerCallback  # THÊM DÒNG NÀY
        EARLY_STOPPING_AVAILABLE = True
    except ImportError:
        try:
            from transformers.trainer_callback import EarlyStoppingCallback
            EARLY_STOPPING_AVAILABLE = True
        except ImportError:
            EarlyStoppingCallback = None
            TrainerCallback = None  
            EARLY_STOPPING_AVAILABLE = False
    
    TRANSFORMERS_AVAILABLE = True
    print(f"✅ Transformers {transformers.__version__} loaded successfully")
except ImportError as e:
    print(f"⚠️  Warning: Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False
    EARLY_STOPPING_AVAILABLE = False

try:
    from .config import ModelConfig, TrainingConfig, ColabConfig, METRICS_DIR
    from .model import BaselineModel
    from .evaluate import compute_metrics
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import ModelConfig, TrainingConfig, ColabConfig, METRICS_DIR
    from src.model import BaselineModel
    from src.evaluate import compute_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.debug("🧹 GPU memory cleared")


def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"💾 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


class MemoryCallback(TrainerCallback):
    """Callback to manage memory during training"""
    def __init__(self, clear_every_n_steps=100):
        super().__init__()
        self.clear_every_n_steps = clear_every_n_steps
        self.step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if self.step % self.clear_every_n_steps == 0:
            clear_gpu_memory()
            if ColabConfig.LOG_GPU_MEMORY:
                log_gpu_memory()
        return control


# ============================================================================
# BASELINE MODEL TRAINING
# ============================================================================

def train_baseline_model(
    train_df,
    val_df,
    test_df,
    save_path: str = None
) -> Dict[str, Any]:
    """
    Train baseline model (TF-IDF + Logistic Regression)
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        save_path: Path to save the model
        
    Returns:
        Dictionary containing training results
    """
    logger.info("="*80)
    logger.info("TRAINING BASELINE MODEL")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Extract text and labels
    X_train = train_df['cleaned_content'].tolist()
    y_train = train_df['label'].tolist()
    X_val = val_df['cleaned_content'].tolist()
    y_val = val_df['label'].tolist()
    X_test = test_df['cleaned_content'].tolist()
    y_test = test_df['label'].tolist()
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Create and train baseline model
    baseline_model = BaselineModel()
    baseline_model.train(X_train, y_train, X_val, y_val)
    
    # Make predictions
    train_preds = baseline_model.predict(X_train)
    val_preds = baseline_model.predict(X_val)
    test_preds = baseline_model.predict(X_test)
    
    train_proba = baseline_model.predict_proba(X_train)
    val_proba = baseline_model.predict_proba(X_val)
    test_proba = baseline_model.predict_proba(X_test)
    
    # Calculate metrics
    train_metrics = {
        'accuracy': float(accuracy_score(y_train, train_preds)),
        'precision': float(precision_recall_fscore_support(y_train, train_preds, average='weighted', zero_division=0)[0]),
        'recall': float(precision_recall_fscore_support(y_train, train_preds, average='weighted', zero_division=0)[1]),
        'f1': float(precision_recall_fscore_support(y_train, train_preds, average='weighted', zero_division=0)[2]),
        'roc_auc': float(roc_auc_score(y_train, train_proba[:, 1]))
    }
    
    val_metrics = {
        'accuracy': float(accuracy_score(y_val, val_preds)),
        'precision': float(precision_recall_fscore_support(y_val, val_preds, average='weighted', zero_division=0)[0]),
        'recall': float(precision_recall_fscore_support(y_val, val_preds, average='weighted', zero_division=0)[1]),
        'f1': float(precision_recall_fscore_support(y_val, val_preds, average='weighted', zero_division=0)[2]),
        'roc_auc': float(roc_auc_score(y_val, val_proba[:, 1]))
    }
    
    test_metrics = {
        'accuracy': float(accuracy_score(y_test, test_preds)),
        'precision': float(precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)[0]),
        'recall': float(precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)[1]),
        'f1': float(precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)[2]),
        'roc_auc': float(roc_auc_score(y_test, test_proba[:, 1]))
    }
    
    training_time = time.time() - start_time
    
    # Save model if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        baseline_model.save(save_path)
    
    # Prepare results
    results = {
        'model_type': 'baseline_tfidf_lr',
        'model_name': 'Baseline (TF-IDF + Logistic Regression)',
        'training_time': float(training_time),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    logger.info("✅ Baseline model training completed!")
    logger.info(f"⏱️  Training time: {training_time:.2f} seconds")
    logger.info(f"📊 Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"📊 Test F1-score: {test_metrics['f1']:.4f}")
    
    return results


# ============================================================================
# ENHANCED BERT/DEBERTA TRAINER CLASS
# ============================================================================

class EnhancedBertTrainer:
    """
    Enhanced BERT/DeBERTa trainer with advanced techniques
    Optimized for Google Colab with memory management
    """
    
    def __init__(
        self,
        model_name: str = ModelConfig.MODEL_NAME,
        output_dir: str = None
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library not available. "
                "Please install: pip install transformers accelerate"
            )
        
        self.model_name = model_name
        self.output_dir = output_dir or "results/models/deberta"
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Detect Colab environment
        self.is_colab = ColabConfig.IS_COLAB
        
        logger.info(f"🤖 EnhancedBertTrainer initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Output: {self.output_dir}")
        logger.info(f"   Environment: {'Google Colab' if self.is_colab else 'Local'}")
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with optimizations"""
        logger.info(f"📥 Loading {self.model_name}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with optimizations
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=ModelConfig.NUM_LABELS,
                hidden_dropout_prob=ModelConfig.DROPOUT_RATE,
                attention_probs_dropout_prob=ModelConfig.DROPOUT_RATE,
            )
            
            # Enable gradient checkpointing for memory efficiency
            if TrainingConfig.GRADIENT_CHECKPOINTING:
                self.model.gradient_checkpointing_enable()
                logger.info("✅ Gradient checkpointing enabled")
            
            logger.info("✅ Model and tokenizer loaded successfully")
            log_gpu_memory()
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def train(
        self,
        train_dataset,
        val_dataset,
        num_epochs: int = ModelConfig.NUM_EPOCHS,
        batch_size: int = ModelConfig.BATCH_SIZE,
        learning_rate: float = ModelConfig.LEARNING_RATE,
        warmup_ratio: float = ModelConfig.WARMUP_RATIO,
        weight_decay: float = ModelConfig.WEIGHT_DECAY
    ) -> Dict[str, Any]:
        """
        Train the model with enhanced settings
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Batch size per device
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio
            weight_decay: Weight decay
            
        Returns:
            Dictionary containing training results
        """
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()
        
        logger.info("=" * 80)
        logger.info(f"🚀 TRAINING {self.model_name.upper()}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Device: {device}")
        
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            log_gpu_memory()
        
        logger.info(f"\n📊 Dataset Information:")
        logger.info(f"   Training samples: {len(train_dataset)}")
        logger.info(f"   Validation samples: {len(val_dataset)}")
        
        logger.info(f"\n⚙️  Training Hyperparameters:")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Gradient accumulation: {ModelConfig.GRADIENT_ACCUMULATION_STEPS}")
        logger.info(f"   Effective batch size: {batch_size * ModelConfig.GRADIENT_ACCUMULATION_STEPS}")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Epochs: {num_epochs}")
        logger.info(f"   Warmup ratio: {warmup_ratio}")
        logger.info(f"   Weight decay: {weight_decay}")
        logger.info(f"   Label smoothing: {ModelConfig.LABEL_SMOOTHING_FACTOR}")
        
        # Training arguments with advanced settings
        training_args = TrainingArguments(
            # Output
            output_dir=self.output_dir,
            
            # Training
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=ModelConfig.GRADIENT_ACCUMULATION_STEPS,
            
            # Optimization
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            max_grad_norm=TrainingConfig.MAX_GRAD_NORM,
            optim=TrainingConfig.OPTIM,
            
            # Label smoothing
            label_smoothing_factor=ModelConfig.LABEL_SMOOTHING_FACTOR,
            
            # Learning rate scheduler
            lr_scheduler_type=ModelConfig.SCHEDULER,
            
            # Mixed precision
            fp16=TrainingConfig.USE_FP16 and torch.cuda.is_available(),
            fp16_opt_level=TrainingConfig.FP16_OPT_LEVEL if TrainingConfig.USE_FP16 else "O0",
            
            # Evaluation
            eval_strategy="epoch",
            eval_accumulation_steps=10,  # Save memory during evaluation
            
            # Saving
            save_strategy="epoch",
            save_total_limit=ModelConfig.SAVE_TOTAL_LIMIT,
            load_best_model_at_end=True,
            metric_for_best_model=ModelConfig.METRIC_FOR_BEST_MODEL,
            greater_is_better=ModelConfig.GREATER_IS_BETTER,
            
            # Logging
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=TrainingConfig.LOGGING_STEPS,
            logging_first_step=TrainingConfig.LOGGING_FIRST_STEP,
            report_to="none",  # Disable wandb
            
            # DataLoader
            dataloader_num_workers=TrainingConfig.NUM_WORKERS,
            dataloader_pin_memory=TrainingConfig.PIN_MEMORY,
            dataloader_prefetch_factor=TrainingConfig.PREFETCH_FACTOR if TrainingConfig.NUM_WORKERS > 0 else None,
            
            # Misc
            seed=TrainingConfig.SEED,
            remove_unused_columns=False,
            disable_tqdm=False,
            push_to_hub=False,
        )
        
        # Prepare callbacks
        callbacks = []
        
        # Early stopping
        if EARLY_STOPPING_AVAILABLE:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=ModelConfig.EARLY_STOPPING_PATIENCE
                )
            )
            logger.info(f"✅ Early stopping enabled (patience={ModelConfig.EARLY_STOPPING_PATIENCE})")
        
        # Memory management callback
        if ColabConfig.CLEAR_CACHE_EVERY_N_STEPS:
            callbacks.append(MemoryCallback(ColabConfig.CLEAR_CACHE_EVERY_N_STEPS))
            logger.info(f"✅ Memory management enabled (clear every {ColabConfig.CLEAR_CACHE_EVERY_N_STEPS} steps)")
        
        # Create trainer
        try:
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=callbacks if callbacks else None
            )
            logger.info("✅ Trainer created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create trainer: {e}")
            raise
        
        # Clear memory before training
        clear_gpu_memory()
        
        # Train the model
        logger.info("\n" + "=" * 80)
        logger.info("🚀 STARTING TRAINING")
        logger.info("=" * 80 + "\n")
        
        try:
            train_result = self.trainer.train()
            logger.info("\n✅ Training completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
        training_time = time.time() - start_time
        
        # Evaluate on validation set
        logger.info("\n📊 Evaluating on validation set...")
        eval_result = self.trainer.evaluate()
        
        # Clear memory after training
        clear_gpu_memory()
        
        # Print detailed results
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"\n⏱️  Training Time:")
        logger.info(f"   Total: {training_time:.2f}s ({training_time/60:.2f}m)")
        logger.info(f"   Per epoch: {training_time/num_epochs:.2f}s")
        
        logger.info(f"\n📊 Final Validation Metrics:")
        for k, v in eval_result.items():
            if isinstance(v, (int, float)) and not k.startswith('epoch'):
                logger.info(f"   {k}: {v:.4f}")
        
        # Prepare comprehensive results
        results = {
            'model_type': 'transformer',
            'model_name': self.model_name,
            'training_time': float(training_time),
            'training_time_per_epoch': float(training_time / num_epochs),
            'train_metrics': {
                k: float(v) if isinstance(v, (int, float)) else v 
                for k, v in train_result.metrics.items()
            },
            'eval_metrics': {
                k: float(v) if isinstance(v, (int, float)) else v 
                for k, v in eval_result.items()
            },
            'hyperparameters': {
                'batch_size': batch_size,
                'effective_batch_size': batch_size * ModelConfig.GRADIENT_ACCUMULATION_STEPS,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'warmup_ratio': warmup_ratio,
                'weight_decay': weight_decay,
                'label_smoothing': ModelConfig.LABEL_SMOOTHING_FACTOR,
                'scheduler': ModelConfig.SCHEDULER,
                'fp16': TrainingConfig.USE_FP16,
                'gradient_checkpointing': TrainingConfig.GRADIENT_CHECKPOINTING
            },
            'output_dir': self.output_dir,
            'device': str(device)
        }
        
        if torch.cuda.is_available():
            results['gpu_info'] = {
                'name': torch.cuda.get_device_name(0),
                'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3
            }
        
        logger.info("=" * 80 + "\n")
        
        return results
    
    def save_model(self, save_path: str = None):
        """
        Save the trained model and tokenizer
        
        Args:
            save_path: Path to save the model (default: self.output_dir)
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before saving")
        
        save_path = save_path or self.output_dir
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        try:
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"✅ Model and tokenizer saved to: {save_path}")
            
            # Save to Google Drive if in Colab
            if self.is_colab and ColabConfig.SAVE_CHECKPOINT_TO_DRIVE:
                try:
                    drive_path = ColabConfig.DRIVE_CHECKPOINT_DIR
                    Path(drive_path).mkdir(parents=True, exist_ok=True)
                    self.trainer.save_model(drive_path)
                    self.tokenizer.save_pretrained(drive_path)
                    logger.info(f"✅ Model also saved to Google Drive: {drive_path}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not save to Drive: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def evaluate(self, test_dataset) -> Dict[str, Any]:
        """
        Evaluate the model on test dataset
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation results dictionary
        """
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 EVALUATING ON TEST SET")
        logger.info("=" * 80)
        
        try:
            test_results = self.trainer.evaluate(test_dataset)
            
            logger.info("\n✅ Test evaluation completed!")
            logger.info("\n📊 Test Metrics:")
            for k, v in test_results.items():
                if isinstance(v, (int, float)) and not k.startswith('epoch'):
                    logger.info(f"   {k}: {v:.4f}")
            
            logger.info("=" * 80 + "\n")
            
            return {
                k: float(v) if isinstance(v, (int, float)) else v 
                for k, v in test_results.items()
            }
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_bert_model(
    train_dataset,
    val_dataset,
    test_dataset,
    model_name: str = ModelConfig.MODEL_NAME,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Complete BERT/DeBERTa training pipeline
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        model_name: Name of the transformer model
        output_dir: Output directory for saving
        
    Returns:
        Complete training and evaluation results
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Transformers library not available. "
            "Please install: pip install transformers accelerate"
        )
    
    logger.info("=" * 80)
    logger.info("🚀 TRANSFORMER MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info("=" * 80 + "\n")
    
    # Create trainer
    bert_trainer = EnhancedBertTrainer(model_name=model_name, output_dir=output_dir)
    
    # Train the model
    train_results = bert_trainer.train(train_dataset, val_dataset)
    
    # Evaluate on test set
    test_results = bert_trainer.evaluate(test_dataset)
    
    # Save the model
    bert_trainer.save_model()
    
    # Combine results
    complete_results = {
        **train_results,
        'test_metrics': test_results
    }
    
    logger.info("=" * 80)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\n🎯 Final Results:")
    logger.info(f"   Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    logger.info(f"   Test F1: {test_results.get('eval_f1', 0):.4f}")
    logger.info(f"   Test Precision: {test_results.get('eval_precision', 0):.4f}")
    logger.info(f"   Test Recall: {test_results.get('eval_recall', 0):.4f}")
    logger.info("=" * 80 + "\n")
    
    return complete_results


def save_training_results(results: Dict[str, Any], filepath: str):
    """
    Save training results to JSON file with proper type conversion
    
    Args:
        results: Training results dictionary
        filepath: Path to save the results
    """
    def convert_to_serializable(obj):
        """Convert numpy/torch types to Python native types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        logger.info(f"✅ Training results saved to: {filepath}")
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        raise


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TRAIN.PY MODULE - ENHANCED FOR DEBERTA-V3")
    print("=" * 80)
    print(f"✅ TRANSFORMERS_AVAILABLE: {TRANSFORMERS_AVAILABLE}")
    print(f"✅ EARLY_STOPPING_AVAILABLE: {EARLY_STOPPING_AVAILABLE}")
    
    if TRANSFORMERS_AVAILABLE:
        print(f"✅ Transformers version: {transformers.__version__}")
        print(f"✅ Default model: {ModelConfig.MODEL_NAME}")
        print("\n💡 Ready to train transformer models!")
        
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            log_gpu_memory()
        else:
            print("⚠️  No GPU detected, training will use CPU")
    else:
        print("\n⚠️  Only baseline model training available")
        print("💡 Install: pip install transformers accelerate")
    
    print("=" * 80)