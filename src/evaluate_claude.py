"""
Evaluation pipeline for Fake News Detection
Calculates metrics, confusion matrix, and generates reports
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, Tuple, List

from .config import EvaluationConfig, METRICS_DIR, PREDICTIONS_DIR
from .utils import setup_logger, save_json
from .model_claude import BERTFakeNewsClassifier

logger = setup_logger('evaluation')


class Evaluator:
    """
    Evaluator class for model evaluation
    """
    
    def __init__(self, model: BERTFakeNewsClassifier, device=None):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Evaluator initialized on device: {self.device}")
    
    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions for entire dataset
        
        Args:
            data_loader: DataLoader for dataset
            
        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Get probabilities and predictions
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1)
                
                # Move to CPU and convert to numpy
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return (
            np.array(all_predictions),
            np.array(all_probabilities),
            np.array(all_labels)
        )
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         y_prob: np.ndarray = None) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional, for ROC-AUC)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        # ROC-AUC (if probabilities provided)
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        metrics['precision_real'] = precision_per_class[0]
        metrics['precision_fake'] = precision_per_class[1]
        metrics['recall_real'] = recall_per_class[0]
        metrics['recall_fake'] = recall_per_class[1]
        metrics['f1_real'] = f1_per_class[0]
        metrics['f1_fake'] = f1_per_class[1]
        
        return metrics
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            save_path: str = None):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real (0)', 'Fake (1)'],
                   yticklabels=['Real (0)', 'Fake (1)'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def evaluate(self, data_loader, save_results: bool = True) -> Dict:
        """
        Complete evaluation pipeline
        """
        logger.info("="*80)
        logger.info("EVALUATION STARTED")
        logger.info("="*80)
        
        # Get predictions
        logger.info("\nGenerating predictions...")
        y_pred, y_prob, y_true = self.predict(data_loader)
        
        # Calculate metrics
        logger.info("\nCalculating metrics...")
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # Print metrics
        logger.info("\n" + "="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"\n📊 Overall Metrics:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics


def evaluate_model(model: BERTFakeNewsClassifier,
                  test_loader,
                  device=None) -> Dict:
    """Evaluate model on test set"""
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(test_loader, save_results=True)
    return metrics