"""
Training utilities
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import time
from typing import Dict, Any

from .utils import format_time, calculate_metrics, print_metrics, EarlyStopping


class Trainer:
    """Trainer for BERT-based models"""
    
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 3
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_dataloader)
        metrics = calculate_metrics(np.array(predictions), np.array(true_labels))
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_dataloader)
        metrics = calculate_metrics(np.array(predictions), np.array(true_labels))
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """Full training loop"""
        print(f"\n{'='*70}")
        print(f"Starting Training - {self.num_epochs} epochs")
        print(f"{'='*70}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 70)
            
            # Train
            start_time = time.time()
            train_metrics = self.train_epoch()
            train_time = time.time() - start_time
            
            print(f"\nTraining completed in {format_time(train_time)}")
            print_metrics(train_metrics, "Training")
            
            # Validate
            start_time = time.time()
            val_metrics = self.validate()
            val_time = time.time() - start_time
            
            print(f"\nValidation completed in {format_time(val_time)}")
            print_metrics(val_metrics, "Validation")
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                print(f"\n✓ New best validation loss: {best_val_loss:.4f}")
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"\n⚠ Early stopping triggered after epoch {epoch + 1}")
                break
        
        print(f"\n{'='*70}")
        print("Training completed!")
        print(f"{'='*70}\n")
        
        return self.history
