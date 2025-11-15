import os
import logging
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load and manage BERT model for inference."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.labels = {0: "real", 1: "fake"}
        self.label_names = ["real", "fake"]
    
    def load_model(self) -> bool:
        """Load BERT model from local path or HuggingFace."""
        try:
            # Try loading from local path first
            if os.path.exists(config.MODEL_PATH):
                logger.info(f"Loading model from local path: {config.MODEL_PATH}")
                self.model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_PATH)
                self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
            else:
                logger.info(f"Loading model from HuggingFace: {config.MODEL_NAME}")
                self.model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME)
                self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(
        self,
        text: str,
        return_probabilities: bool = True,
        mc_dropout: bool = False,
        mc_iterations: int = 5
    ) -> dict:
        """Perform inference on text."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=config.MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if mc_dropout and config.MC_DROPOUT_ENABLED:
                    # MC Dropout inference
                    logits_list = []
                    
                    # Enable dropout during inference
                    self.model.train()
                    
                    for _ in range(mc_iterations):
                        outputs = self.model(**inputs)
                        logits_list.append(outputs.logits)
                    
                    # Disable dropout
                    self.model.eval()
                    
                    # Average logits
                    logits = torch.stack(logits_list).mean(dim=0)
                else:
                    # Standard inference
                    outputs = self.model(**inputs)
                    logits = outputs.logits
            
            # Apply temperature scaling
            logits = logits / config.TEMPERATURE
            
            # Get predictions
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
            result = {
                "label": self.labels[predicted_class],
                "class_id": predicted_class,
                "confidence": float(confidence),
                "probabilities": {
                    "real": float(probabilities[0].item()),
                    "fake": float(probabilities[1].item())
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def batch_predict(self, texts: list) -> list:
        """Perform batch inference."""
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting text: {str(e)}")
                results.append({
                    "label": "unknown",
                    "confidence": 0.0,
                    "error": str(e)
                })
        return results
    
    @staticmethod
    def _get_device() -> torch.device:
        """Get appropriate device (GPU if available, else CPU)."""
        if torch.cuda.is_available():
            logger.info("Using GPU for inference")
            return torch.device("cuda")
        else:
            logger.info("Using CPU for inference")
            return torch.device("cpu")
