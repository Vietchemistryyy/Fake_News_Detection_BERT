"""
Training functions for Fake News Detection models
Includes training for both baseline and BERT models
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

# Try to import transformers, handle import error gracefully
try:
    import transformers
    from transformers import (
        TrainingArguments,
        AutoTokenizer,
        AutoModelForSequenceClassification
    )
    
    # Try to import Trainer with fallback
    try:
        from transformers import Trainer
    except ImportError:
        from transformers.trainer import Trainer
    
    # Try to import EarlyStoppingCallback with fallback - MAKE IT OPTIONAL
    try:
        from transformers import EarlyStoppingCallback
        EARLY_STOPPING_AVAILABLE = True
    except ImportError:
        try:
            from transformers.trainer_callback import EarlyStoppingCallback
            EARLY_STOPPING_AVAILABLE = True
        except ImportError:
            EarlyStoppingCallback = None
            EARLY_STOPPING_AVAILABLE = False
    
    TRANSFORMERS_AVAILABLE = True
    print(f"✅ Transformers {transformers.__version__} loaded successfully")
    if EARLY_STOPPING_AVAILABLE:
        print(f"✅ EarlyStoppingCallback available")
    else:
        print(f"⚠️  EarlyStoppingCallback not available (will train without early stopping)")
except ImportError as e:
    print(f"⚠️  Warning: Transformers not available: {e}")
    print("💡 Only baseline model training will be available.")
    TRANSFORMERS_AVAILABLE = False
    EARLY_STOPPING_AVAILABLE = False

try:
    from .config import ModelConfig, TrainingConfig, METRICS_DIR
    from .model import BaselineModel
    from .evaluate import compute_metrics
except ImportError:
    # Fallback for when running the file directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import ModelConfig, TrainingConfig, METRICS_DIR
    from src.model import BaselineModel
    from src.evaluate import compute_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        'test_metrics': test_metrics,
        'train_predictions': train_preds.tolist(),
        'val_predictions': val_preds.tolist(),
        'test_predictions': test_preds.tolist(),
        'train_probabilities': train_proba.tolist(),
        'val_probabilities': val_proba.tolist(),
        'test_probabilities': test_proba.tolist()
    }
    
    logger.info("✅ Baseline model training completed!")
    logger.info(f"⏱️  Training time: {training_time:.2f} seconds")
    logger.info(f"📊 Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"📊 Test F1-score: {test_metrics['f1']:.4f}")
    
    return results


class BertTrainer:
    """
    Custom BERT/RoBERTa trainer class with unified interface
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
        self.output_dir = output_dir or "results/models/bert"
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        logger.info(f"🤖 BertTrainer initialized for: {model_name}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
    def setup_model_and_tokenizer(self):
        """
        Setup model and tokenizer
        """
        logger.info(f"📥 Loading {self.model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=ModelConfig.NUM_LABELS
            )
            logger.info("✅ Model and tokenizer loaded successfully")
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
        warmup_steps: int = ModelConfig.WARMUP_STEPS,
        weight_decay: float = ModelConfig.WEIGHT_DECAY
    ) -> Dict[str, Any]:
        """
        Train the BERT/RoBERTa model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay
            
        Returns:
            Training results dictionary
        """
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()
        
        logger.info("="*80)
        logger.info(f"TRAINING {self.model_name.upper()} MODEL")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Device: {device}")
        logger.info(f"📊 Training samples: {len(train_dataset)}")
        logger.info(f"📊 Validation samples: {len(val_dataset)}")
        logger.info(f"⚙️  Batch size: {batch_size}")
        logger.info(f"⚙️  Learning rate: {learning_rate}")
        logger.info(f"⚙️  Epochs: {num_epochs}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            eval_strategy="epoch",  # Changed from evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
            fp16=TrainingConfig.USE_FP16 and torch.cuda.is_available(),
            dataloader_num_workers=TrainingConfig.NUM_WORKERS,
            remove_unused_columns=False,
            report_to="none",  # Disable wandb/tensorboard
            push_to_hub=False,
            disable_tqdm=False,  # Show progress bar
        )
        
        # Prepare callbacks
        callbacks = []
        if EARLY_STOPPING_AVAILABLE:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
            logger.info("✅ Early stopping enabled (patience=3)")
        else:
            logger.warning("⚠️  Early stopping not available - training all epochs")
        
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
        
        # Train the model
        logger.info("🚀 Starting training...")
        try:
            train_result = self.trainer.train()
            logger.info("✅ Training completed successfully")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
        training_time = time.time() - start_time
        
        # Evaluate on validation set
        logger.info("📊 Evaluating on validation set...")
        eval_result = self.trainer.evaluate()
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED ✅")
        logger.info("="*80)
        logger.info(f"⏱️  Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        logger.info(f"📊 Final validation metrics:")
        for k, v in eval_result.items():
            if isinstance(v, (int, float)):
                logger.info(f"   {k}: {v:.4f}")
        
        # Prepare results with consistent structure
        results = {
            'model_type': 'bert',
            'model_name': self.model_name,
            'training_time': float(training_time),
            'train_metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in train_result.metrics.items()},
            'eval_metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in eval_result.items()},
            'output_dir': self.output_dir,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': str(device)
        }
        
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
        
        # Ensure directory exists
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        try:
            self.trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"✅ Model and tokenizer saved to: {save_path}")
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
        
        logger.info("📊 Evaluating on test set...")
        
        try:
            test_results = self.trainer.evaluate(test_dataset)
            
            logger.info("✅ Test evaluation completed!")
            logger.info("📊 Test metrics:")
            for k, v in test_results.items():
                if isinstance(v, (int, float)):
                    logger.info(f"   {k}: {v:.4f}")
            
            # Convert to consistent format
            return {k: float(v) if isinstance(v, (int, float)) else v 
                   for k, v in test_results.items()}
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise


def train_bert_model(
    train_dataset,
    val_dataset,
    test_dataset,
    model_name: str = ModelConfig.MODEL_NAME,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Complete BERT/RoBERTa training pipeline
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        model_name: Name of the BERT model
        output_dir: Output directory for saving
        
    Returns:
        Complete training and evaluation results
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Transformers library not available. "
            "Please install: pip install transformers accelerate"
        )
    
    logger.info("="*80)
    logger.info("BERT/ROBERTA TRAINING PIPELINE")
    logger.info("="*80)
    
    # Create trainer
    bert_trainer = BertTrainer(model_name=model_name, output_dir=output_dir)
    
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
    
    logger.info("="*80)
    logger.info("PIPELINE COMPLETED ✅")
    logger.info("="*80)
    
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
    
    # Ensure parent directory exists
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
    print("="*80)
    print("TRAIN.PY MODULE")
    print("="*80)
    print(f"✅ TRANSFORMERS_AVAILABLE: {TRANSFORMERS_AVAILABLE}")
    print(f"✅ EARLY_STOPPING_AVAILABLE: {EARLY_STOPPING_AVAILABLE}")
    
    if TRANSFORMERS_AVAILABLE:
        print(f"✅ Transformers version: {transformers.__version__}")
        print("\n💡 Ready to train BERT/RoBERTa models")
    else:
        print("\n⚠️  Only baseline model training available")
        print("💡 Install transformers: pip install transformers accelerate")
    
    print("="*80)