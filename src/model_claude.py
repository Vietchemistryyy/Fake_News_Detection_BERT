"""
Model architecture for Fake News Detection
BERT-based classifier with custom head
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict

from .config import ModelConfig
from .utils import setup_logger

logger = setup_logger('model')


class BERTFakeNewsClassifier(nn.Module):
    """
    BERT-based classifier for fake news detection
    Uses pretrained BERT with a classification head
    """
    
    def __init__(self, 
                 model_name: str = ModelConfig.MODEL_NAME,
                 num_labels: int = ModelConfig.NUM_LABELS,
                 dropout: float = 0.3):
        """
        Initialize classifier
        
        Args:
            model_name: Name of pretrained BERT model
            num_labels: Number of output classes (2 for binary)
            dropout: Dropout rate for classification head
        """
        super(BERTFakeNewsClassifier, self).__init__()
        
        logger.info(f"Initializing model: {model_name}")
        
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from BERT config
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        logger.info(f"Model initialized successfully!")
        logger.info(f"   Hidden size: {self.hidden_size}")
        logger.info(f"   Num labels: {num_labels}")
        logger.info(f"   Dropout: {dropout}")
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            
        Returns:
            Logits (batch_size, num_labels)
        """
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Một số model (như DistilBERT, RoBERTa) không có pooler_output
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Lấy embedding của token [CLS] làm pooled output
            pooled_output = outputs.last_hidden_state[:, 0]

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification
        logits = self.classifier(pooled_output)

        return logits
        # # Pass through BERT
        # outputs = self.bert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask
        # )
        
        # # Get [CLS] token representation
        # pooled_output = outputs.pooler_output
        
        # # Apply dropout
        # pooled_output = self.dropout(pooled_output)
        
        # # Classification
        # logits = self.classifier(pooled_output)
        
        # return logits
    
    def freeze_bert_encoder(self):
        """Freeze BERT encoder layers (only train classification head)"""
        for param in self.bert.parameters():
            param.requires_grad = False
        logger.info("🔒 BERT encoder frozen")
    
    def unfreeze_bert_encoder(self):
        """Unfreeze BERT encoder layers"""
        for param in self.bert.parameters():
            param.requires_grad = True
        logger.info("🔓 BERT encoder unfrozen")
    
    def unfreeze_last_n_layers(self, n: int = 2):
        """
        Unfreeze last n layers of BERT encoder
        Useful for gradual unfreezing
        
        Args:
            n: Number of last layers to unfreeze
        """
        # Freeze all first
        self.freeze_bert_encoder()
        
        # Unfreeze last n layers
        for layer in self.bert.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.info(f"🔓 Unfroze last {n} layers of BERT encoder")
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def load_model(model_name: str = ModelConfig.MODEL_NAME,
              num_labels: int = ModelConfig.NUM_LABELS,
              device: str = None) -> BERTFakeNewsClassifier:
    """
    Load model and move to device
    
    Args:
        model_name: Name of pretrained model
        num_labels: Number of output classes
        device: Device to move model to (cuda/cpu)
        
    Returns:
        Initialized model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading model on device: {device}")
    
    model = BERTFakeNewsClassifier(
        model_name=model_name,
        num_labels=num_labels
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = model.get_total_params()
    trainable_params = model.get_trainable_params()
    
    logger.info(f"\nModel Statistics:")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    logger.info(f"   Trainable %: {trainable_params/total_params*100:.2f}%")
    
    return model


def save_model(model: BERTFakeNewsClassifier,
              save_path: str,
              optimizer=None,
              epoch: int = None,
              metrics: Dict = None):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
        optimizer: Optimizer state (optional)
        epoch: Current epoch (optional)
        metrics: Training metrics (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_name': ModelConfig.MODEL_NAME,
            'num_labels': ModelConfig.NUM_LABELS,
        }
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path)
    logger.info(f"💾 Model saved to: {save_path}")


def load_checkpoint(checkpoint_path: str,
                   device: str = None) -> BERTFakeNewsClassifier:
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = BERTFakeNewsClassifier(
        model_name=checkpoint['model_config']['model_name'],
        num_labels=checkpoint['model_config']['num_labels']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info("✅ Model loaded successfully!")
    
    if 'epoch' in checkpoint:
        logger.info(f"   Epoch: {checkpoint['epoch']}")
    
    if 'metrics' in checkpoint:
        logger.info(f"   Metrics: {checkpoint['metrics']}")
    
    return model


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logger.info("Testing BERTFakeNewsClassifier...")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device=device)
    
    # Create dummy input
    batch_size = 4
    seq_length = 128
    
    dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_length)).to(device)
    dummy_attention_mask = torch.ones((batch_size, seq_length)).to(device)
    
    # Forward pass
    logger.info("\nTesting forward pass...")
    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)
    
    logger.info(f"✅ Forward pass successful!")
    logger.info(f"   Input shape: {dummy_input_ids.shape}")
    logger.info(f"   Output shape: {logits.shape}")
    logger.info(f"   Output logits: {logits[0]}")
    
    # Test predictions
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    
    logger.info(f"\n   Probabilities: {probs[0]}")
    logger.info(f"   Prediction: {predictions[0].item()}")
    
    # Test freezing
    logger.info("\nTesting layer freezing...")
    model.freeze_bert_encoder()
    frozen_params = model.get_trainable_params()
    
    model.unfreeze_bert_encoder()
    unfrozen_params = model.get_trainable_params()
    
    logger.info(f"   Frozen trainable params: {frozen_params:,}")
    logger.info(f"   Unfrozen trainable params: {unfrozen_params:,}")
    
    logger.info("\n✅ All model tests passed!")