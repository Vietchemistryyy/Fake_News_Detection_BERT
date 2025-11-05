"""
Evaluation functions for Fake News Detection models
Includes metrics computation, visualization, and model comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from pathlib import Path
import json

try:
    from .config import VISUALIZATIONS_DIR, METRICS_DIR
except ImportError:
    # Fallback for when running the file directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import VISUALIZATIONS_DIR, METRICS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for Hugging Face Trainer
    
    Args:
        eval_pred: Evaluation predictions from Trainer
        
    Returns:
        Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    
    # Get predicted labels
    predictions = np.argmax(predictions, axis=1)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute extended evaluation metrics including per-class metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary containing extended metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report
    }
    
    # Add ROC AUC and Average Precision if probabilities provided
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                # Binary classification with probabilities for both classes
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                avg_precision = average_precision_score(y_true, y_proba[:, 1])
                results['roc_auc'] = float(roc_auc)
                results['average_precision'] = float(avg_precision)
            else:
                # Single probability array
                roc_auc = roc_auc_score(y_true, y_proba)
                avg_precision = average_precision_score(y_true, y_proba)
                results['roc_auc'] = float(roc_auc)
                results['average_precision'] = float(avg_precision)
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC or Average Precision: {e}")
    
    return results


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        model_name: Name of the model for logging
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Use compute_extended_metrics for comprehensive evaluation
    results = compute_extended_metrics(y_true, y_pred, y_proba)
    results['model_name'] = model_name
    
    logger.info(f"✅ {model_name} evaluation completed:")
    logger.info(f"   Accuracy: {results['accuracy']:.4f}")
    logger.info(f"   Precision: {results['precision']:.4f}")
    logger.info(f"   Recall: {results['recall']:.4f}")
    logger.info(f"   F1-score: {results['f1']:.4f}")
    if 'roc_auc' in results:
        logger.info(f"   ROC AUC: {results['roc_auc']:.4f}")
    
    return results


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: str = None
):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake']
    )
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add accuracy text
    accuracy = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}', 
             transform=plt.gca().transAxes, ha='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: str = None
):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        model_name: Name of the model
        save_path: Path to save the plot
    """
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_scores = y_proba[:, 1]
    else:
        y_scores = y_proba
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to: {save_path}")
    
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: str = None
):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        model_name: Name of the model
        save_path: Path to save the plot
    """
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_scores = y_proba[:, 1]
    else:
        y_scores = y_proba
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'AP = {avg_precision:.4f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to: {save_path}")
    
    plt.show()


def compare_models(
    results_list: List[Dict[str, Any]],
    save_path: str = None
):
    """
    Compare multiple models' performance
    
    Args:
        results_list: List of evaluation results dictionaries
        save_path: Path to save the comparison plot
    """
    # Extract metrics for comparison
    model_names = [r['model_name'] for r in results_list]
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Create comparison DataFrame
    comparison_data = []
    for result in results_list:
        row = {'Model': result['model_name']}
        for metric in metrics:
            if metric in result:
                row[metric] = result[metric]
            else:
                row[metric] = None
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            bars = ax.bar(df_comparison['Model'], df_comparison[metric], 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_title(f'{metric.capitalize()}', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.capitalize())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if not pd.isna(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom')
    
    # Remove empty subplot
    if len(metrics) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison saved to: {save_path}")
    
    plt.show()
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(df_comparison.round(4).to_string(index=False))
    
    return df_comparison


def save_evaluation_results(
    results: Dict[str, Any],
    filepath: str
):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Evaluation results dictionary
        filepath: Path to save the results
    """
    # Ensure parent directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {filepath}")


def create_evaluation_report(
    baseline_results: Dict[str, Any],
    bert_results: Dict[str, Any],
    save_dir: str = None
) -> Dict[str, Any]:
    """
    Create comprehensive evaluation report comparing baseline and BERT models
    
    Args:
        baseline_results: Baseline model evaluation results
        bert_results: BERT model evaluation results
        save_dir: Directory to save the report
        
    Returns:
        Comprehensive evaluation report
    """
    logger.info("Creating comprehensive evaluation report...")
    
    # Create comparison
    comparison_df = compare_models([baseline_results, bert_results])
    
    # Create report
    report = {
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'baseline_model': baseline_results,
        'bert_model': bert_results,
        'comparison': comparison_df.to_dict('records'),
        'summary': {
            'best_accuracy': max(baseline_results['accuracy'], bert_results['accuracy']),
            'best_f1': max(baseline_results['f1'], bert_results['f1']),
            'best_roc_auc': max(baseline_results.get('roc_auc', 0), bert_results.get('roc_auc', 0)),
            'accuracy_improvement': bert_results['accuracy'] - baseline_results['accuracy'],
            'f1_improvement': bert_results['f1'] - baseline_results['f1']
        }
    }
    
    # Save report if directory provided
    if save_dir:
        save_path = Path(save_dir) / "evaluation_report.json"
        save_evaluation_results(report, save_path)
    
    logger.info("✅ Evaluation report created!")
    
    return report


def analyze_misclassifications(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: List[str],
    model_name: str = "Model",
    top_n: int = 10
) -> pd.DataFrame:
    """
    Analyze misclassified samples
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        texts: List of text samples
        model_name: Name of the model
        top_n: Number of misclassifications to show
        
    Returns:
        DataFrame containing misclassified samples
    """
    # Find misclassified samples
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        logger.info(f"No misclassifications found for {model_name}")
        return pd.DataFrame()
    
    # Create DataFrame of misclassifications
    misclassified_data = []
    for idx in misclassified_indices[:top_n]:
        misclassified_data.append({
            'index': idx,
            'true_label': y_true[idx],
            'predicted_label': y_pred[idx],
            'text': texts[idx][:200] + "..." if len(texts[idx]) > 200 else texts[idx]
        })
    
    df_misclassified = pd.DataFrame(misclassified_data)
    
    logger.info(f"Found {len(misclassified_indices)} misclassifications for {model_name}")
    logger.info(f"Showing top {min(top_n, len(misclassified_indices))} misclassifications:")
    
    return df_misclassified