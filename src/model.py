"""
Model definitions and loading utilities
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    AutoConfig
)
from typing import Optional


def load_model_and_tokenizer(
    model_name: str = "roberta-base",
    num_labels: int = 2,
    dropout_rate: float = 0.1
):
    """Load pretrained model and tokenizer"""
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load config with custom dropout
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout_rate,
        attention_probs_dropout_prob=dropout_rate
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    print(f"✓ Model loaded: {model_name}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, tokenizer


def save_model(model, tokenizer, save_dir: str):
    """Save model and tokenizer"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print(f"✓ Model saved to: {save_dir}")


def load_trained_model(model_dir: str):
    """Load a trained model"""
    print(f"Loading trained model from: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    print(f"✓ Model loaded successfully")
    
    return model, tokenizer


def freeze_layers(model, num_layers_to_freeze: int = 0):
    """Freeze bottom N layers of the model"""
    if num_layers_to_freeze == 0:
        return model
    
    # For RoBERTa/BERT models
    if hasattr(model, 'roberta'):
        encoder = model.roberta.encoder
    elif hasattr(model, 'bert'):
        encoder = model.bert.encoder
    else:
        print("⚠ Model type not recognized for layer freezing")
        return model
    
    # Freeze embeddings
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False
    
    # Freeze specified layers
    for layer in encoder.layer[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Froze {num_layers_to_freeze} layers")
    print(f"  - Trainable parameters: {trainable:,}")
    
    return model


def get_model_summary(model):
    """Print model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable:        {total_params - trainable_params:,}")
    print("="*70 + "\n")
