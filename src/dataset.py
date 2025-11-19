"""
Dataset classes for training
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict
import pandas as pd


class FakeNewsDataset(Dataset):
    """Dataset for fake news detection"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file"""
    df = pd.read_csv(filepath)
    
    # Ensure required columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    
    return df


def prepare_datasets(
    train_file: str,
    val_file: str,
    test_file: str,
    tokenizer,
    max_length: int = 256
) -> tuple:
    """Prepare train, validation, and test datasets"""
    
    # Load data
    train_df = load_data(train_file)
    val_df = load_data(val_file)
    test_df = load_data(test_file)
    
    # Create datasets
    train_dataset = FakeNewsDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = FakeNewsDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = FakeNewsDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return train_dataset, val_dataset, test_dataset
