"""
PyTorch Dataset for Fake News Detection
Custom dataset class for RoBERTa/BERT training and evaluation
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

# Try to import transformers, handle import error gracefully
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Transformers not available: {e}")
    print("Only baseline model training will be available.")
    TRANSFORMERS_AVAILABLE = False

try:
    from .config import ModelConfig, DataConfig, TrainingConfig
except ImportError:
    # Fallback for when running the file directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import ModelConfig, DataConfig, TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FakeNewsDataset(Dataset):
    """
    PyTorch Dataset for fake news detection
    Compatible with RoBERTa, BERT, DistilBERT, etc.
    
    Args:
        texts: List of text strings
        labels: List of corresponding labels (0=real, 1=fake)
        tokenizer: Hugging Face tokenizer (AutoTokenizer)
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = ModelConfig.MAX_LENGTH
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Dataset initialized with {len(texts)} samples")
        logger.info(f"Max length: {max_length}")
        logger.info(f"Tokenizer: {type(tokenizer).__name__}")
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize the text
        # RoBERTa uses same tokenization interface as BERT
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_dataset_from_dataframe(
    df: pd.DataFrame,
    tokenizer,
    text_column: str = DataConfig.CLEANED_TEXT_COLUMN,
    label_column: str = DataConfig.LABEL_COLUMN,
    max_length: int = ModelConfig.MAX_LENGTH
) -> FakeNewsDataset:
    """
    Create a FakeNewsDataset from a pandas DataFrame
    
    Args:
        df: DataFrame containing text and labels
        tokenizer: Hugging Face tokenizer (AutoTokenizer)
        text_column: Name of the text column
        label_column: Name of the label column
        max_length: Maximum sequence length
        
    Returns:
        FakeNewsDataset instance
    """
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    logger.info(f"Creating dataset from DataFrame:")
    logger.info(f"  - Samples: {len(texts)}")
    logger.info(f"  - Text column: {text_column}")
    logger.info(f"  - Label column: {label_column}")
    
    return FakeNewsDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    batch_size: int = ModelConfig.BATCH_SIZE,
    num_workers: int = TrainingConfig.NUM_WORKERS
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: Hugging Face tokenizer (AutoTokenizer)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        
    Returns:
        Dictionary containing train, val, and test DataLoaders
    """
    logger.info("Creating DataLoaders...")
    
    # Create datasets
    train_dataset = create_dataset_from_dataframe(train_df, tokenizer)
    val_dataset = create_dataset_from_dataframe(val_df, tokenizer)
    test_dataset = create_dataset_from_dataframe(test_df, tokenizer)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=TrainingConfig.PIN_MEMORY
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TrainingConfig.PIN_MEMORY
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=TrainingConfig.PIN_MEMORY
    )
    
    logger.info(f"DataLoaders created:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num workers: {num_workers}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def get_sample_batch(data_loader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
    """
    Get a sample batch from a DataLoader for inspection
    
    Args:
        data_loader: PyTorch DataLoader
        
    Returns:
        Sample batch dictionary
    """
    for batch in data_loader:
        logger.info(f"Sample batch keys: {batch.keys()}")
        logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
        logger.info(f"  attention_mask shape: {batch['attention_mask'].shape}")
        logger.info(f"  labels shape: {batch['labels'].shape}")
        return batch


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("DATASET.PY MODULE TEST")
    print("="*80)
    
    print(f"\n✅ TRANSFORMERS_AVAILABLE: {TRANSFORMERS_AVAILABLE}")
    
    if TRANSFORMERS_AVAILABLE:
        print(f"\n🧪 Testing dataset creation with {ModelConfig.MODEL_NAME}:")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)
            print(f"   ✅ Tokenizer loaded: {ModelConfig.MODEL_NAME}")
            print(f"   ✅ Tokenizer type: {type(tokenizer).__name__}")
            
            # Create sample data
            sample_texts = [
                "This is a sample news article about politics.",
                "Breaking news: Major event happening now!",
                "Scientists discover new breakthrough in medicine."
            ]
            sample_labels = [0, 1, 0]
            
            # Create dataset
            dataset = FakeNewsDataset(
                texts=sample_texts,
                labels=sample_labels,
                tokenizer=tokenizer,
                max_length=128
            )
            
            print(f"   ✅ Dataset created with {len(dataset)} samples")
            
            # Test get item
            sample_item = dataset[0]
            print(f"\n   Sample item inspection:")
            print(f"   - Keys: {sample_item.keys()}")
            print(f"   - input_ids shape: {sample_item['input_ids'].shape}")
            print(f"   - attention_mask shape: {sample_item['attention_mask'].shape}")
            print(f"   - label: {sample_item['labels'].item()}")
            
            # Test DataLoader
            print(f"\n🧪 Testing DataLoader:")
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            print(f"   ✅ Batch created")
            print(f"   - Batch size: {batch['input_ids'].shape[0]}")
            print(f"   - input_ids shape: {batch['input_ids'].shape}")
            print(f"   - attention_mask shape: {batch['attention_mask'].shape}")
            print(f"   - labels shape: {batch['labels'].shape}")
            
        except Exception as e:
            print(f"   ❌ Dataset test failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("\n⚠️  Transformers not available - dataset creation requires transformers")
        print("💡 Install: pip install transformers")
    
    print("\n" + "="*80)