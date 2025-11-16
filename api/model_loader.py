import os
import logging
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load and manage BERT models for inference (multi-language support)."""
    
    def __init__(self):
        # English model (RoBERTa)
        self.model_en = None
        self.tokenizer_en = None
        
        # Vietnamese model (PhoBERT)
        self.model_vi = None
        self.tokenizer_vi = None
        
        self.device = self._get_device()
        self.labels = {0: "real", 1: "fake"}
        self.label_names = ["real", "fake"]
        
        # Track which models are loaded
        self.models_loaded = {
            "en": False,
            "vi": False
        }
    
    def load_model_en(self) -> bool:
        """Load English model (RoBERTa)."""
        try:
            # Try loading from local path first
            if os.path.exists(config.MODEL_PATH):
                logger.info(f"Loading English model from local path: {config.MODEL_PATH}")
                
                # Check if required files exist
                required_files = ["config.json", "pytorch_model.bin"]
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(config.MODEL_PATH, f))]
                
                if missing_files:
                    logger.error(f"Missing required files: {missing_files}")
                    raise FileNotFoundError(f"Missing model files: {missing_files}")
                
                # Load model and tokenizer
                self.model_en = AutoModelForSequenceClassification.from_pretrained(
                    config.MODEL_PATH,
                    local_files_only=True,
                    num_labels=2
                )
                self.tokenizer_en = AutoTokenizer.from_pretrained(
                    config.MODEL_PATH,
                    local_files_only=True
                )
                logger.info("✓ English model and tokenizer loaded from local path")
            else:
                logger.warning(f"Local model path not found: {config.MODEL_PATH}")
                logger.info(f"Loading English model from HuggingFace: {config.MODEL_NAME}")
                self.model_en = AutoModelForSequenceClassification.from_pretrained(
                    config.MODEL_NAME,
                    num_labels=2
                )
                self.tokenizer_en = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            
            # Move to device
            self.model_en = self.model_en.to(self.device)
            self.model_en.eval()
            
            self.models_loaded["en"] = True
            logger.info(f"✓ English model loaded successfully on device: {self.device}")
            logger.info(f"✓ Model type: {type(self.model_en).__name__}")
            logger.info(f"✓ Number of labels: {self.model_en.config.num_labels}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading English model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_model_vi(self) -> bool:
        """Load Vietnamese model (PhoBERT)."""
        try:
            logger.info("Loading Vietnamese model (PhoBERT)...")
            
            # Check if local path exists
            if config.PHOBERT_MODEL_PATH and os.path.exists(config.PHOBERT_MODEL_PATH):
                logger.info(f"Loading PhoBERT from local path: {config.PHOBERT_MODEL_PATH}")
                self.model_vi = AutoModelForSequenceClassification.from_pretrained(
                    config.PHOBERT_MODEL_PATH,
                    local_files_only=True,
                    num_labels=2
                )
                self.tokenizer_vi = AutoTokenizer.from_pretrained(
                    config.PHOBERT_MODEL_PATH,
                    local_files_only=True
                )
            else:
                logger.info(f"Loading PhoBERT from HuggingFace: {config.PHOBERT_MODEL_NAME}")
                self.model_vi = AutoModelForSequenceClassification.from_pretrained(
                    config.PHOBERT_MODEL_NAME,
                    num_labels=2
                )
                self.tokenizer_vi = AutoTokenizer.from_pretrained(
                    config.PHOBERT_MODEL_NAME
                )
            
            # Move to device
            self.model_vi = self.model_vi.to(self.device)
            self.model_vi.eval()
            
            self.models_loaded["vi"] = True
            logger.info(f"✓ Vietnamese model loaded successfully on device: {self.device}")
            logger.info(f"✓ Model type: {type(self.model_vi).__name__}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading Vietnamese model: {str(e)}")
            logger.error("PhoBERT will not be available. Install: pip install transformers")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_model(self, language: str = "en") -> bool:
        """Load BERT model from local path or HuggingFace."""
        try:
            # Try loading from local path first
            if os.path.exists(config.MODEL_PATH):
                logger.info(f"Loading model from local path: {config.MODEL_PATH}")
                
                # Check if required files exist
                required_files = ["config.json", "pytorch_model.bin"]
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(config.MODEL_PATH, f))]
                
                if missing_files:
                    logger.error(f"Missing required files: {missing_files}")
                    raise FileNotFoundError(f"Missing model files: {missing_files}")
                
                # Load model and tokenizer
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    config.MODEL_PATH,
                    local_files_only=True,
                    num_labels=2
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config.MODEL_PATH,
                    local_files_only=True
                )
                logger.info("✓ Model and tokenizer loaded from local path")
            else:
                logger.warning(f"Local model path not found: {config.MODEL_PATH}")
                logger.info(f"Loading model from HuggingFace: {config.MODEL_NAME}")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    config.MODEL_NAME,
                    num_labels=2
                )
                self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✓ Model loaded successfully on device: {self.device}")
            logger.info(f"✓ Model type: {type(self.model).__name__}")
            logger.info(f"✓ Number of labels: {self.model.config.num_labels}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def predict(
        self,
        text: str,
        language: str = "en",
        return_probabilities: bool = True,
        mc_dropout: bool = False,
        mc_iterations: int = 5
    ) -> dict:
        """Perform inference on text with language support."""
        # Select model and tokenizer based on language
        if language == "vi":
            if not self.models_loaded["vi"]:
                raise RuntimeError("Vietnamese model not loaded")
            model = self.model_vi
            tokenizer = self.tokenizer_vi
        else:
            if not self.models_loaded["en"]:
                raise RuntimeError("English model not loaded")
            model = self.model_en
            tokenizer = self.tokenizer_en
        
        try:
            # Tokenize input
            inputs = tokenizer(
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
                    model.train()
                    
                    for _ in range(mc_iterations):
                        outputs = model(**inputs)
                        logits_list.append(outputs.logits)
                    
                    # Disable dropout
                    model.eval()
                    
                    # Average logits
                    logits = torch.stack(logits_list).mean(dim=0)
                else:
                    # Standard inference
                    outputs = model(**inputs)
                    logits = outputs.logits
            
            # Apply temperature scaling (higher temp = less confident)
            logits = logits / config.TEMPERATURE
            
            # Apply bias correction if model tends to predict one class too often
            # This helps with imbalanced training data
            if config.BIAS_CORRECTION_ENABLED:
                # Shift logits slightly toward "real" to counteract fake bias
                logits[0][0] += config.BIAS_CORRECTION_SHIFT  # boost "real" class
            
            # Get predictions
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
            # Additional calibration: cap extreme confidences
            # This helps with overfit models that produce extreme logits
            if confidence > 0.95:
                # Reduce extreme confidence
                adjustment_factor = 0.85 + (confidence - 0.95) * 0.10 / 0.05
                confidence = min(confidence * adjustment_factor, 0.95)
                
                # Recalculate probabilities with adjusted confidence
                other_prob = 1.0 - confidence
                if predicted_class == 0:
                    probabilities = torch.tensor([confidence, other_prob])
                else:
                    probabilities = torch.tensor([other_prob, confidence])
            
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
    
    def batch_predict(self, texts: list, language: str = "en") -> list:
        """Perform batch inference."""
        results = []
        for text in texts:
            try:
                result = self.predict(text, language=language)
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
