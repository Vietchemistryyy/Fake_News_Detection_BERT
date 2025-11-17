import os
from typing import Optional

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "roberta-base")
# Get absolute path to model directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
_default_model_path = os.path.join(_current_dir, "..", "models", "BERT")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.normpath(_default_model_path))
MAX_LENGTH = 256
BATCH_SIZE = 16

# API configuration
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

# AI Verification APIs configuration
# OpenAI (paid, requires credit card)
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "10"))
ENABLE_OPENAI = os.getenv("ENABLE_OPENAI", "false").lower() == "true"

# Gemini (FREE, recommended)
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "false").lower() == "true"

# Groq (FREE, fast)
GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
ENABLE_GROQ = os.getenv("ENABLE_GROQ", "false").lower() == "true"

# Verification provider priority (first available will be used)
VERIFICATION_PROVIDER = os.getenv("VERIFICATION_PROVIDER", "auto")  # auto, openai, gemini, groq

# Model inference settings
# Higher temperature = less confident predictions (better for overfit models)
TEMPERATURE = float(os.getenv("TEMPERATURE", "2.5"))
MC_DROPOUT_ENABLED = os.getenv("MC_DROPOUT_ENABLED", "true").lower() == "true"
MC_DROPOUT_ITERATIONS = int(os.getenv("MC_DROPOUT_ITERATIONS", "10"))

# Bias correction (if model is biased toward one class)
BIAS_CORRECTION_ENABLED = os.getenv("BIAS_CORRECTION_ENABLED", "true").lower() == "true"
BIAS_CORRECTION_SHIFT = float(os.getenv("BIAS_CORRECTION_SHIFT", "0.5"))  # Positive = favor "real"

# Text preprocessing
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 5000

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "fake_news_detection")

# Authentication configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production-use-openssl-rand-hex-32")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))  # 7 days

# Multi-language support
SUPPORTED_LANGUAGES = ["en", "vi"]
DEFAULT_LANGUAGE = "en"

# PhoBERT configuration (Vietnamese)
PHOBERT_MODEL_NAME = os.getenv("PHOBERT_MODEL_NAME", "vinai/phobert-base")
# Get absolute path to PhoBERT model directory
_default_phobert_path = os.path.join(_current_dir, "..", "models", "PhoBERT")
PHOBERT_MODEL_PATH = os.getenv("PHOBERT_MODEL_PATH", os.path.normpath(_default_phobert_path))
