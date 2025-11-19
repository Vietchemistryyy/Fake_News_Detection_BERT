"""
Model evaluation utilities
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluate model on a dataset"""
    
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'loss': total_loss / len(dataloader)
    }
    
    return predictions, true_labels, metrics


def print_evaluation_report(predictions: np.ndarray, true_labels: np.ndarray):
    """Print detailed evaluation report"""
    
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        predictions,
        target_names=['Real', 'Fake'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Real  Fake")
    print(f"Actual Real   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Fake   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    print("="*70 + "\n")


def plot_confusion_matrix(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    save_path: str = None
):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_training_history(history: Dict, save_path: str = None):
    """Plot training history"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to: {save_path}")
    
    plt.show()


def predict_single(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 256
) -> Dict[str, any]:
    """Predict on a single text"""
    
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
    
    return {
        'prediction': 'fake' if prediction == 1 else 'real',
        'confidence': probs[0][prediction].item(),
        'probabilities': {
            'real': probs[0][0].item(),
            'fake': probs[0][1].item()
        }
    }
