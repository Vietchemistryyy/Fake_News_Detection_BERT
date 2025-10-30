"""
Text preprocessing pipeline for Fake News Detection
Handles text cleaning, normalization, and data splitting
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# from config import DataConfig, PROCESSED_DATA_DIR, RAW_DATA_DIR
# from utils import setup_logger, save_csv, Timer, set_seed

from src.config import DataConfig, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils import setup_logger, save_csv, Timer, set_seed

# Setup logger
logger = setup_logger('preprocessing')

class TextPreprocessor:
    """
    Text preprocessing class for cleaning and normalizing text data
    """
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_mentions: bool = True,
                 remove_hashtags: bool = True,
                 lowercase: bool = True,
                 remove_extra_spaces: bool = True):
        """
        Initialize preprocessor with configuration
        
        Args:
            remove_urls: Remove HTTP/HTTPS URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            lowercase: Convert text to lowercase
            remove_extra_spaces: Remove extra whitespace
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.lowercase = lowercase
        self.remove_extra_spaces = remove_extra_spaces
        
        logger.info("TextPreprocessor initialized")
        logger.info(f"  - Remove URLs: {self.remove_urls}")
        logger.info(f"  - Remove Mentions: {self.remove_mentions}")
        logger.info(f"  - Remove Hashtags: {self.remove_hashtags}")
        logger.info(f"  - Lowercase: {self.lowercase}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            text = re.sub(r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', '', text)
        
        # Remove mentions (@username)
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (#topic)
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        
        # Remove extra whitespace
        if self.remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def clean_dataframe(self, df: pd.DataFrame, 
                       text_column: str = DataConfig.TEXT_COLUMN) -> pd.DataFrame:
        """
        Clean text column in entire DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of text column to clean
            
        Returns:
            DataFrame with cleaned text in new column
        """
        logger.info(f"Cleaning {len(df)} records...")
        
        df_clean = df.copy()
        
        # Apply cleaning
        df_clean[DataConfig.CLEANED_TEXT_COLUMN] = df_clean[text_column].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        initial_len = len(df_clean)
        df_clean = df_clean[df_clean[DataConfig.CLEANED_TEXT_COLUMN].str.len() > 0]
        removed = initial_len - len(df_clean)
        
        logger.info(f"Cleaning complete!")
        logger.info(f"  - Removed {removed} empty records")
        logger.info(f"  - Final dataset: {len(df_clean)} records")
        
        return df_clean


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """
    Load raw dataset from CSV
    
    Args:
        filepath: Path to CSV file (default: from config)
        
    Returns:
        DataFrame with raw data
    """
    if filepath is None:
        filepath = DataConfig.RAW_DATA_PATH
    
    logger.info(f"Loading data from: {filepath}")
    
    df = pd.read_csv(filepath)
    
    logger.info(f"✅ Loaded {len(df)} records")
    logger.info(f"   Columns: {df.columns.tolist()}")
    logger.info(f"   Shape: {df.shape}")
    
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check data quality and return statistics
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    logger.info("Checking data quality...")
    
    quality = {}
    
    # Missing values
    missing = df.isnull().sum()
    quality['missing_values'] = missing.to_dict()
    quality['total_missing'] = int(missing.sum())
    
    # Empty strings
    empty_content = df[df[DataConfig.TEXT_COLUMN].astype(str).str.strip() == ''].shape[0]
    quality['empty_content'] = empty_content
    
    # Duplicates
    duplicates = df.duplicated().sum()
    quality['duplicates'] = int(duplicates)
    
    # Label distribution
    label_counts = df[DataConfig.LABEL_COLUMN].value_counts()
    quality['label_distribution'] = label_counts.to_dict()
    
    # Invalid labels (not 0 or 1)
    invalid_labels = df[~df[DataConfig.LABEL_COLUMN].isin([0, 1])].shape[0]
    quality['invalid_labels'] = invalid_labels
    
    logger.info("Data quality check complete:")
    logger.info(f"  - Missing values: {quality['total_missing']}")
    logger.info(f"  - Empty content: {quality['empty_content']}")
    logger.info(f"  - Duplicates: {quality['duplicates']}")
    logger.info(f"  - Invalid labels: {quality['invalid_labels']}")
    
    return quality


def split_data(df: pd.DataFrame, 
               train_ratio: float = DataConfig.TRAIN_RATIO,
               val_ratio: float = DataConfig.VAL_RATIO,
               test_ratio: float = DataConfig.TEST_RATIO,
               random_state: int = DataConfig.RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets (stratified)
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training (default: 0.70)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for test (default: 0.15)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Splitting data...")
    logger.info(f"  - Train: {train_ratio*100:.1f}%")
    logger.info(f"  - Validation: {val_ratio*100:.1f}%")
    logger.info(f"  - Test: {test_ratio*100:.1f}%")
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    # First split: train + temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=df[DataConfig.LABEL_COLUMN]
    )
    
    # Second split: val + test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state,
        stratify=temp_df[DataConfig.LABEL_COLUMN]
    )
    
    logger.info("Split complete:")
    logger.info(f"  - Training:   {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  - Validation: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  - Test:       {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify stratification
    train_fake_pct = (train_df[DataConfig.LABEL_COLUMN] == 1).sum() / len(train_df) * 100
    val_fake_pct = (val_df[DataConfig.LABEL_COLUMN] == 1).sum() / len(val_df) * 100
    test_fake_pct = (test_df[DataConfig.LABEL_COLUMN] == 1).sum() / len(test_df) * 100
    
    logger.info("Stratification verification:")
    logger.info(f"  - Train fake news: {train_fake_pct:.2f}%")
    logger.info(f"  - Val fake news:   {val_fake_pct:.2f}%")
    logger.info(f"  - Test fake news:  {test_fake_pct:.2f}%")
    
    return train_df, val_df, test_df


def calculate_class_weights(labels: np.ndarray) -> dict:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        labels: Array of labels
        
    Returns:
        Dictionary mapping class to weight
    """
    logger.info("Calculating class weights...")
    
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    weights_dict = {i: float(weight) for i, weight in enumerate(class_weights)}
    
    logger.info("Class weights computed:")
    logger.info(f"  - Class 0 (Real): {weights_dict[0]:.4f}")
    logger.info(f"  - Class 1 (Fake): {weights_dict[1]:.4f}")
    
    return weights_dict


def save_splits(train_df: pd.DataFrame, 
                val_df: pd.DataFrame, 
                test_df: pd.DataFrame,
                output_dir: str = None):
    """
    Save train/val/test splits to CSV files
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output directory (default: from config)
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    
    logger.info(f"Saving splits to: {output_dir}")
    
    # Create directory if not exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save files
    train_path = output_dir / DataConfig.TRAIN_FILE
    val_path = output_dir / DataConfig.VAL_FILE
    test_path = output_dir / DataConfig.TEST_FILE
    
    save_csv(train_df, train_path)
    save_csv(val_df, val_path)
    save_csv(test_df, test_path)
    
    logger.info("✅ All splits saved successfully!")


def preprocess_pipeline(input_file: str = None, 
                       save_output: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline
    
    Args:
        input_file: Path to input CSV (default: from config)
        save_output: Whether to save processed splits
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("="*80)
    logger.info("PREPROCESSING PIPELINE STARTED")
    logger.info("="*80)
    
    # Set random seed for reproducibility
    set_seed(DataConfig.RANDOM_SEED)
    
    # 1. Load data
    with Timer("Data loading"):
        df = load_raw_data(input_file)
    
    # 2. Check data quality
    with Timer("Quality check"):
        quality = check_data_quality(df)
    
    # 3. Handle missing/invalid data
    logger.info("\nHandling missing and invalid data...")
    initial_len = len(df)
    
    # Drop missing values
    if quality['total_missing'] > 0:
        df = df.dropna()
        logger.info(f"  - Dropped {initial_len - len(df)} rows with missing values")
    
    # Remove empty content
    df = df[df[DataConfig.TEXT_COLUMN].astype(str).str.strip() != '']
    logger.info(f"  - Removed {initial_len - len(df)} rows with empty content")
    
    # Remove invalid labels
    df = df[df[DataConfig.LABEL_COLUMN].isin([0, 1])]
    logger.info(f"  - Final dataset: {len(df)} rows")
    
    # 4. Clean text
    with Timer("Text cleaning"):
        preprocessor = TextPreprocessor(
            remove_urls=DataConfig.REMOVE_URLS,
            remove_mentions=DataConfig.REMOVE_MENTIONS,
            remove_hashtags=DataConfig.REMOVE_HASHTAGS,
            lowercase=DataConfig.LOWERCASE
        )
        df_clean = preprocessor.clean_dataframe(df)
    
    # 5. Split data
    with Timer("Data splitting"):
        train_df, val_df, test_df = split_data(df_clean)
    
    # 6. Calculate class weights
    class_weights = calculate_class_weights(train_df[DataConfig.LABEL_COLUMN].values)
    
    # 7. Save splits
    if save_output:
        with Timer("Saving splits"):
            save_splits(train_df, val_df, test_df)
    
    logger.info("="*80)
    logger.info("PREPROCESSING PIPELINE COMPLETED ✅")
    logger.info("="*80)
    
    return train_df, val_df, test_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run preprocessing pipeline
    train_df, val_df, test_df = preprocess_pipeline()
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nProcessed data saved to: {PROCESSED_DATA_DIR}")
    print(f"  - Training:   {len(train_df):,} samples")
    print(f"  - Validation: {len(val_df):,} samples")
    print(f"  - Test:       {len(test_df):,} samples")
    print("\n Next step: Run notebooks/01_eda.ipynb for exploratory analysis")
    print("="*80)