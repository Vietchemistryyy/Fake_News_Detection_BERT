"""
Training pipeline for BERT Fake News Classifier
Handles training loop, validation, and model checkpointing
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

from .config import ModelConfig, TrainingConfig, MODELS_DIR
from .model_claude import BERTFakeNewsClassifier, save_model
from .utils import setup_logger, set_seed, save_json

logger = setup_logger('training')


class Trainer:
    """
    Trainer class for BERT fine-tuning
    """
    
    def __init__(self,
                 model: BERTFakeNewsClassifier,
                 train_loader,
                 val_loader,
                 device=None,
                 learning_rate: float = ModelConfig.LEARNING_RATE,
                 epochs: int = ModelConfig.NUM_EPOCHS):
        """
        Initialize trainer
        
        Args:
            model: BERT classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            epochs: Number of training epochs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=ModelConfig.EPSILON,
            weight_decay=ModelConfig.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=ModelConfig.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        logger.info("Trainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Learning rate: {self.learning_rate}")
        logger.info(f"   Epochs: {self.epochs}")
        logger.info(f"   Training batches: {len(self.train_loader)}")
        logger.info(f"   Validation batches: {len(self.val_loader)}")
    
    def train_epoch(self) -> tuple:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                TrainingConfig.MAX_GRAD_NORM
            )
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Update progress bar
            current_loss = total_loss / (progress_bar.n + 1)
            current_acc = correct_predictions / total_samples
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate(self) -> tuple:
        """
        Validate model
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                # Accumulate loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self, save_best: bool = True, save_dir: Path = None):
        """
        Complete training loop
        
        Args:
            save_best: Whether to save best model
            save_dir: Directory to save models
        """
        if save_dir is None:
            save_dir = MODELS_DIR
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("TRAINING STARTED")
        logger.info("="*80)
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            logger.info("-" * 80)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Log results
            logger.info(f"\nEpoch {epoch + 1} Results:")
            logger.info(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            logger.info(f"   Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                best_model_path = save_dir / ModelConfig.BEST_MODEL_NAME
                save_model(
                    self.model,
                    best_model_path,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    metrics={
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }
                )
                logger.info(f"   🌟 New best model saved! Val Loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= ModelConfig.EARLY_STOPPING_PATIENCE:
                logger.info(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                logger.info(f"   No improvement for {ModelConfig.EARLY_STOPPING_PATIENCE} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total training time: {total_time/60:.2f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save training history
        history_path = save_dir / "training_history.json"
        save_json(self.history, history_path)
        logger.info(f"Training history saved to: {history_path}")
        
        return self.history


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_model(train_loader,
               val_loader,
               model_name: str = ModelConfig.MODEL_NAME,
               epochs: int = ModelConfig.NUM_EPOCHS,
               learning_rate: float = ModelConfig.LEARNING_RATE,
               save_dir: Path = None):
    """
    Complete training pipeline
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model_name: Name of pretrained model
        epochs: Number of epochs
        learning_rate: Learning rate
        save_dir: Directory to save models
        
    Returns:
        Tuple of (trained_model, history)
    """
    # Set random seed
    set_seed(TrainingConfig.SEED)
    
    # Initialize model
    from model_claude import load_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_name=model_name, device=device)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        epochs=epochs
    )
    
    # Train
    history = trainer.train(save_best=True, save_dir=save_dir)
    
    return trainer.model, history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    from dataset_claude import load_data_loaders
    
    logger.info("Loading data loaders...")
    train_loader, val_loader, test_loader = load_data_loaders()
    
    logger.info("\nStarting training...")
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=ModelConfig.NUM_EPOCHS
    )
    
    logger.info("\n✅ Training pipeline completed successfully!")