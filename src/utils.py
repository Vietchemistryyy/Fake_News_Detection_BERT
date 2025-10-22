"""
Utility functions for Fake News Detection project
Helper functions for logging, visualization, and common operations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Any

# Set style for plots
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Setup logger with console and file handlers

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        logging.Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def save_json(data: Dict, filepath: Path):
    """Save dictionary to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved JSON to: {filepath}")


def load_json(filepath: Path) -> Dict:
    """Load JSON file to dictionary"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Loaded JSON from: {filepath}")
    return data


def save_csv(df: pd.DataFrame, filepath: Path):
    """Save DataFrame to CSV"""
    df.to_csv(filepath, index=False)
    print(f"✅ Saved CSV to: {filepath}")


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load CSV to DataFrame"""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded CSV from: {filepath}")
    return df


# ============================================================================
# DATA STATISTICS
# ============================================================================

def get_data_statistics(df: pd.DataFrame, text_column: str = 'content',
                        label_column: str = 'label') -> Dict:
    """
    Calculate comprehensive statistics for the dataset

    Args:
        df: DataFrame with text and labels
        text_column: Name of text column
        label_column: Name of label column

    Returns:
        Dictionary with statistics
    """
    stats = {}

    # Basic info
    stats['total_records'] = len(df)
    stats['n_features'] = len(df.columns)

    # Label distribution
    label_counts = df[label_column].value_counts().to_dict()
    stats['label_distribution'] = label_counts
    stats['fake_count'] = label_counts.get(1, 0)
    stats['real_count'] = label_counts.get(0, 0)
    stats['fake_percentage'] = (stats['fake_count'] / stats['total_records']) * 100
    stats['imbalance_ratio'] = stats['fake_count'] / max(stats['real_count'], 1)

    # Text statistics
    df['text_length'] = df[text_column].astype(str).str.len()
    df['word_count'] = df[text_column].astype(str).str.split().str.len()

    stats['avg_text_length'] = float(df['text_length'].mean())
    stats['median_text_length'] = float(df['text_length'].median())
    stats['min_text_length'] = int(df['text_length'].min())
    stats['max_text_length'] = int(df['text_length'].max())

    stats['avg_word_count'] = float(df['word_count'].mean())
    stats['median_word_count'] = float(df['word_count'].median())
    stats['min_word_count'] = int(df['word_count'].min())
    stats['max_word_count'] = int(df['word_count'].max())

    # Missing values
    stats['missing_values'] = df.isnull().sum().to_dict()

    return stats


def print_statistics(stats: Dict):
    """Pretty print statistics"""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"\n📊 Total Records: {stats['total_records']:,}")
    print(f"\n🏷️  Label Distribution:")
    print(f"   - Real News (0): {stats['real_count']:,} ({100 - stats['fake_percentage']:.2f}%)")
    print(f"   - Fake News (1): {stats['fake_count']:,} ({stats['fake_percentage']:.2f}%)")
    print(f"   - Imbalance Ratio: {stats['imbalance_ratio']:.3f}")
    print(f"\n📝 Text Statistics:")
    print(f"   - Avg Length: {stats['avg_text_length']:.1f} characters")
    print(f"   - Median Length: {stats['median_text_length']:.1f} characters")
    print(f"   - Range: {stats['min_text_length']} - {stats['max_text_length']} characters")
    print(f"   - Avg Words: {stats['avg_word_count']:.1f}")
    print("=" * 80)


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_label_distribution(df: pd.DataFrame, label_column: str = 'label',
                            save_path: Path = None):
    """Plot label distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    label_counts = df[label_column].value_counts()
    axes[0].bar(['Real (0)', 'Fake (1)'],
                [label_counts.get(0, 0), label_counts.get(1, 0)],
                color=['#10b981', '#ef4444'])
    axes[0].set_title('Label Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)

    # Add counts on bars
    for i, v in enumerate([label_counts.get(0, 0), label_counts.get(1, 0)]):
        axes[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')

    # Pie chart
    label_pct = df[label_column].value_counts(normalize=True) * 100
    axes[1].pie([label_pct.get(0, 0), label_pct.get(1, 0)],
                labels=['Real (0)', 'Fake (1)'],
                autopct='%1.1f%%',
                colors=['#10b981', '#ef4444'],
                startangle=90)
    axes[1].set_title('Label Distribution (%)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot to: {save_path}")

    plt.show()


def plot_text_length_distribution(df: pd.DataFrame, text_column: str = 'content',
                                  label_column: str = 'label', save_path: Path = None):
    """Plot text length distributions"""
    df['text_length'] = df[text_column].astype(str).str.len()
    df['word_count'] = df[text_column].astype(str).str.split().str.len()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Text length histogram
    axes[0, 0].hist(df['text_length'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['text_length'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['text_length'].mean():.0f}")
    axes[0, 0].set_title('Text Length Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Word count histogram
    axes[0, 1].hist(df['word_count'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['word_count'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['word_count'].mean():.0f}")
    axes[0, 1].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Text length by label
    for label in df[label_column].unique():
        data = df[df[label_column] == label]['text_length']
        axes[1, 0].hist(data, bins=30, alpha=0.6,
                        label=f"{'Fake' if label == 1 else 'Real'} ({label})")
    axes[1, 0].set_title('Text Length by Label', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Length (characters)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Word count by label (box plot)
    df.boxplot(column='word_count', by=label_column, ax=axes[1, 1])
    axes[1, 1].set_title('Word Count by Label', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Label')
    axes[1, 1].set_ylabel('Word Count')
    plt.sca(axes[1, 1])
    plt.xticks([1, 2], ['Real (0)', 'Fake (1)'])

    plt.suptitle('Text Statistics Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot to: {save_path}")

    plt.show()


# ============================================================================
# TIMER UTILITY
# ============================================================================

class Timer:
    """Context manager for timing code blocks"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        print(f"⏱️  {self.name} started...")
        return self

    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"✅ {self.name} completed in {elapsed:.2f} seconds")


# ============================================================================
# PROGRESS UTILITIES
# ============================================================================

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()


# ============================================================================
# SEED SETTING FOR REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✅ Random seed set to: {seed}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    print(f"Available functions: {[func for func in dir() if not func.startswith('_')]}")