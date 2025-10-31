"""
Model definitions for Fake News Detection
Includes both baseline models and BERT-based models (RoBERTa, DistilBERT, etc.)
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
        AutoModelForSequenceClassification  # ✅ CHANGED: Generic Auto class
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Transformers not available: {e}")
    print("💡 Only baseline model will be available.")
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
        
        logger.info("✅ Baseline pipeline created (TF-IDF + Logistic Regression)")
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
        logger.info("🚀 Training baseline model...")
        
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
            logger.info(f"📊 Validation accuracy: {val_score:.4f}")
    
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
        logger.info(f"✅ Baseline model saved to: {filepath}")
    
    def load(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.pipeline = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"✅ Baseline model loaded from: {filepath}")


class BertClassifier(nn.Module):
    """
    Generic BERT-based classifier for fake news detection
    Compatible with: RoBERTa, DistilBERT, BERT, etc.
    """
    
    def __init__(
        self,
        model_name: str = ModelConfig.MODEL_NAME,
        num_labels: int = ModelConfig.NUM_LABELS,
        dropout_rate: float = 0.1
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library not available. "
                "Please install: pip install transformers"
            )
        
        super(BertClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained model using Auto class (works for any BERT variant)
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Add dropout and classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        logger.info(f"✅ Classifier initialized with {model_name}")
        logger.info(f"   Hidden size: {self.config.hidden_size}")
        logger.info(f"   Num labels: {num_labels}")
    
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
        
        # Use pooled output (CLS token for BERT/RoBERTa, or pooler_output)
        # Handle different model architectures
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # For models without pooler_output (like some RoBERTa variants)
            pooled_output = outputs.last_hidden_state[:, 0, :]  # Take CLS token
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class BertForSequenceClassification:
    """
    Wrapper class for Hugging Face transformer models
    Compatible with: RoBERTa, DistilBERT, BERT, ALBERT, etc.
    """
    
    def __init__(self, model_name: str = ModelConfig.MODEL_NAME):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library not available. "
                "Please install: pip install transformers"
            )
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """
        Load the pre-trained model and tokenizer using Auto classes
        Works for any transformer model (RoBERTa, BERT, DistilBERT, etc.)
        """
        logger.info(f"📥 Loading {self.model_name}...")
        
        try:
            # ✅ FIXED: Use AutoModelForSequenceClassification (works for all models)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=ModelConfig.NUM_LABELS
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"✅ {self.model_name} loaded successfully")
            logger.info(f"   Model type: {type(self.model).__name__}")
            logger.info(f"   Tokenizer type: {type(self.tokenizer).__name__}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def get_model(self):
        """
        Get the loaded model
        
        Returns:
            Loaded transformer model
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
    Create a transformer model instance
    Compatible with RoBERTa, BERT, DistilBERT, etc.
    
    Args:
        model_name: Name of the transformer model
        
    Returns:
        BertForSequenceClassification instance
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Transformers library not available. "
            "Please install: pip install transformers"
        )
    
    return BertForSequenceClassification(model_name)


def save_bert_model(model: nn.Module, tokenizer, save_path: str):
    """
    Save transformer model and tokenizer
    
    Args:
        model: Trained transformer model
        tokenizer: Tokenizer
        save_path: Directory to save the model
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not available.")
    
    import os
    from pathlib import Path
    
    # Create directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    try:
        # Save model state dict
        model_path = os.path.join(save_path, "model.pt")
        torch.save(model.state_dict(), model_path)
        
        # Save tokenizer
        tokenizer.save_pretrained(save_path)
        
        logger.info(f"✅ Model and tokenizer saved to: {save_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        raise


def load_bert_model(model_name: str, model_path: str) -> Tuple[nn.Module, Any]:
    """
    Load transformer model and tokenizer
    
    Args:
        model_name: Name of the transformer model
        model_path: Path to the saved model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library not available.")
    
    import os
    from pathlib import Path
    
    logger.info(f"📥 Loading model from: {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # ✅ FIXED: Use AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=ModelConfig.NUM_LABELS
        )
        
        # Load state dict if exists
        model_state_path = os.path.join(model_path, "model.pt")
        if os.path.exists(model_state_path):
            model.load_state_dict(torch.load(model_state_path, map_location='cpu'))
            logger.info(f"✅ Loaded model weights from: {model_state_path}")
        
        logger.info(f"✅ Model and tokenizer loaded successfully")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type: str = "roberta", **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ("baseline", "roberta", "bert", "distilbert")
        **kwargs: Additional arguments for model creation
        
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == "baseline":
        return create_baseline_model()
    
    elif model_type in ["roberta", "bert", "distilbert", "albert"]:
        # Map model type to model name
        model_names = {
            "roberta": "roberta-base",
            "bert": "bert-base-uncased",
            "distilbert": "distilbert-base-uncased",
            "albert": "albert-base-v2"
        }
        
        model_name = kwargs.get('model_name', model_names.get(model_type))
        return create_bert_model(model_name)
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: baseline, roberta, bert, distilbert, albert"
        )


# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MODEL.PY MODULE")
    print("="*80)
    
    print(f"\n✅ TRANSFORMERS_AVAILABLE: {TRANSFORMERS_AVAILABLE}")
    
    if TRANSFORMERS_AVAILABLE:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        # Test model creation
        print(f"\n🧪 Testing model creation:")
        print(f"   Default model: {ModelConfig.MODEL_NAME}")
        
        try:
            model_wrapper = create_bert_model(ModelConfig.MODEL_NAME)
            print(f"   ✅ Model wrapper created successfully")
            
            # Load model
            model_wrapper.load_model()
            print(f"   ✅ Model loaded successfully")
            print(f"   Model type: {type(model_wrapper.model).__name__}")
            print(f"   Tokenizer type: {type(model_wrapper.tokenizer).__name__}")
            
        except Exception as e:
            print(f"   ❌ Model creation failed: {e}")
    
    else:
        print("\n⚠️  Only baseline model available")
        print("💡 Install transformers: pip install transformers")
    
    print("\n" + "="*80)