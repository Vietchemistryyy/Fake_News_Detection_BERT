"""
Model definitions for Fake News Detection
Includes both baseline models and BERT-based models
"""

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Try to import transformers, handle import error gracefully
try:
    from transformers import (
        AutoTokenizer, 
        AutoModel, 
        AutoConfig,
        DistilBertForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Transformers not available: {e}")
    print("Only baseline model will be available.")
    TRANSFORMERS_AVAILABLE = False

try:
    from .config import ModelConfig, BaselineConfig
except ImportError:
    # Fallback for when running the file directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import ModelConfig, BaselineConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Baseline model using TF-IDF + Logistic Regression
    """
    
    def __init__(self):
        self.pipeline = None
        self.is_trained = False
        
    def create_pipeline(self) -> Pipeline:
        """
        Create the TF-IDF + Logistic Regression pipeline
        
        Returns:
            Scikit-learn Pipeline
        """
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=BaselineConfig.MAX_FEATURES,
                min_df=BaselineConfig.MIN_DF,
                max_df=BaselineConfig.MAX_DF,
                ngram_range=BaselineConfig.NGRAM_RANGE,
                lowercase=True,
                stop_words='english'
            )),
            ('classifier', LogisticRegression(
                solver=BaselineConfig.SOLVER,
                max_iter=BaselineConfig.MAX_ITER,
                C=BaselineConfig.C,
                random_state=42
            ))
        ])
        
        logger.info("Baseline pipeline created with TF-IDF + Logistic Regression")
        return pipeline
    
    def train(self, X_train: list, y_train: list, X_val: list = None, y_val: list = None):
        """
        Train the baseline model
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data (optional)
            y_val: Validation labels (optional)
        """
        logger.info("Training baseline model...")
        
        # Create pipeline if not exists
        if self.pipeline is None:
            self.pipeline = self.create_pipeline()
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        logger.info("✅ Baseline model training completed")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_score = self.pipeline.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")
    
    def predict(self, X: list) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Text data to predict
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: list) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Text data to predict
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict_proba(X)
    
    def save(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Baseline model saved to: {filepath}")
    
    def load(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.pipeline = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Baseline model loaded from: {filepath}")


class BertClassifier(nn.Module):
    """
    BERT-based classifier for fake news detection
    """
    
    def __init__(
        self,
        model_name: str = ModelConfig.MODEL_NAME,
        num_labels: int = ModelConfig.NUM_LABELS,
        dropout_rate: float = 0.1
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Please install transformers to use BERT models.")
        
        super(BertClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Add dropout and classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        logger.info(f"BERT classifier initialized with {model_name}")
        logger.info(f"Hidden size: {self.config.hidden_size}, Num labels: {num_labels}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for classification
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class BertForSequenceClassification:
    """
    Wrapper class for Hugging Face BERT model
    """
    
    def __init__(self, model_name: str = ModelConfig.MODEL_NAME):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available. Please install transformers to use BERT models.")
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """
        Load the pre-trained model and tokenizer
        """
        logger.info(f"Loading {self.model_name} model and tokenizer...")
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=ModelConfig.NUM_LABELS
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        logger.info(" Model and tokenizer loaded successfully")
    
    def get_model(self):
        """
        Get the loaded model
        
        Returns:
            Loaded BERT model
        """
        if self.model is None:
            self.load_model()
        return self.model
    
    def get_tokenizer(self):
        """
        Get the loaded tokenizer
        
        Returns:
            Loaded tokenizer
        """
        if self.tokenizer is None:
            self.load_model()
        return self.tokenizer


def create_baseline_model() -> BaselineModel:
    """
    Create a baseline model instance
    
    Returns:
        BaselineModel instance
    """
    return BaselineModel()


def create_bert_model(model_name: str = ModelConfig.MODEL_NAME) -> BertForSequenceClassification:
    """
    Create a BERT model instance
    
    Args:
        model_name: Name of the BERT model
        
    Returns:
        BertForSequenceClassification instance
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not available. Please install transformers to use BERT models.")
    
    return BertForSequenceClassification(model_name)


def save_bert_model(model: nn.Module, tokenizer, save_path: str):
    """
    Save BERT model and tokenizer
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        save_path: Directory to save the model
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not available.")
    
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_path, "model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    
    logger.info(f"BERT model and tokenizer saved to: {save_path}")


def load_bert_model(model_name: str, model_path: str) -> Tuple[nn.Module, Any]:
    """
    Load BERT model and tokenizer
    
    Args:
        model_name: Name of the BERT model
        model_path: Path to the saved model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not available.")
    
    import os
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=ModelConfig.NUM_LABELS
    )
    
    # Load state dict
    model_state_path = os.path.join(model_path, "model.pt")
    model.load_state_dict(torch.load(model_state_path, map_location='cpu'))
    
    logger.info(f"BERT model and tokenizer loaded from: {model_path}")
    
    return model, tokenizer
