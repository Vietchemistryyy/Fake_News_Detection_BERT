"""
PyTorch Dataset classes for Fake News Detection
Handles tokenization and data loading for BERT models
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

from .config import DataConfig, ModelConfig, TrainingConfig
from .utils import setup_logger

logger = setup_logger('dataset')


class FakeNewsDataset(Dataset):
    """
    PyTorch Dataset for Fake News Detection
    Tokenizes text and prepares it for BERT models
    """
    
    def __init__(self, 
                 texts: List[str],
                 labels: List[int],
                 tokenizer,
                 max_length: int = ModelConfig.MAX_LENGTH):
        """
        Initialize dataset
        
        Args:
            texts: List of text strings
            labels: List of labels (0 or 1)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Dataset initialized with {len(texts)} samples")
        logger.info(f"Max length: {max_length}")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from dataset
        
        Args:
            idx: Index of item
            
        Returns:
            Dictionary with input_ids, attention_mask, and label
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df: pd.DataFrame,
                      tokenizer,
                      max_length: int = ModelConfig.MAX_LENGTH,
                      batch_size: int = ModelConfig.BATCH_SIZE,
                      shuffle: bool = True,
                      text_column: str = DataConfig.CLEANED_TEXT_COLUMN,
                      label_column: str = DataConfig.LABEL_COLUMN) -> DataLoader:
    """
    Create DataLoader from DataFrame
    
    Args:
        df: DataFrame with text and labels
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle data
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        PyTorch DataLoader
    """
    dataset = FakeNewsDataset(
        texts=df[text_column].tolist(),
        labels=df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=TrainingConfig.PIN_MEMORY
    )


def load_data_loaders(tokenizer_name: str = ModelConfig.MODEL_NAME,
                     train_path: str = None,
                     val_path: str = None,
                     test_path: str = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load train, validation, and test data loaders
    
    Args:
        tokenizer_name: Name of tokenizer to use
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("="*80)
    logger.info("LOADING DATA LOADERS")
    logger.info("="*80)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) #old
    # new tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name, 
    revision='main', 
    use_fast=True,
    trust_remote_code=False # Thêm cờ an toàn này nếu cần
)
    
    # Set default paths
    if train_path is None:
        train_path = DataConfig.TRAIN_PATH
    if val_path is None:
        val_path = DataConfig.VAL_PATH
    if test_path is None:
        test_path = DataConfig.TEST_PATH
    
    # Load DataFrames
    logger.info(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    
    logger.info(f"Loading validation data from: {val_path}")
    val_df = pd.read_csv(val_path)
    
    logger.info(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    # Create DataLoaders
    logger.info("\nCreating data loaders...")
    
    train_loader = create_data_loader(
        train_df,
        tokenizer,
        batch_size=ModelConfig.BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = create_data_loader(
        val_df,
        tokenizer,
        batch_size=ModelConfig.BATCH_SIZE,
        shuffle=False
    )
    
    test_loader = create_data_loader(
        test_df,
        tokenizer,
        batch_size=ModelConfig.BATCH_SIZE,
        shuffle=False
    )
    
    logger.info("\n✅ Data loaders created successfully!")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches:   {len(val_loader)}")
    logger.info(f"   Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Test dataset creation
    logger.info("Testing FakeNewsDataset...")
    
    # Sample data
    texts = [
        "This is a real news article about politics.",
        "This is fake news spreading misinformation.",
        "Breaking news: Important event happened today."
    ]
    labels = [0, 1, 0]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)
    
    # Create dataset
    dataset = FakeNewsDataset(texts, labels, tokenizer, max_length=64)
    
    # Test __getitem__
    sample = dataset[0]
    logger.info(f"\nSample item:")
    logger.info(f"   Input IDs shape: {sample['input_ids'].shape}")
    logger.info(f"   Attention mask shape: {sample['attention_mask'].shape}")
    logger.info(f"   Label: {sample['label']}")
    
    # Decode back to text
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    logger.info(f"   Decoded text: {decoded}")
    
    logger.info("\n✅ Dataset test passed!")
    
    # Test data loader
    logger.info("\nTesting DataLoader...")
    df_test = pd.DataFrame({
        DataConfig.CLEANED_TEXT_COLUMN: texts,
        DataConfig.LABEL_COLUMN: labels
    })
    
    loader = create_data_loader(df_test, tokenizer, batch_size=2, shuffle=False)
    
    batch = next(iter(loader))
    logger.info(f"\nBatch:")
    logger.info(f"   Input IDs shape: {batch['input_ids'].shape}")
    logger.info(f"   Attention mask shape: {batch['attention_mask'].shape}")
    logger.info(f"   Labels shape: {batch['label'].shape}")
    
    logger.info("\n✅ DataLoader test passed!")