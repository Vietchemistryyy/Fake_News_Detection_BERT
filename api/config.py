import os
from typing import Optional

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "roberta-base")
MODEL_PATH = os.getenv("MODEL_PATH", "../models/BERT")
MAX_LENGTH = 256
BATCH_SIZE = 16

# API configuration
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

# OpenAI configuration
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "10"))
ENABLE_OPENAI = os.getenv("ENABLE_OPENAI", "false").lower() == "true"

# Model inference settings
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MC_DROPOUT_ENABLED = os.getenv("MC_DROPOUT_ENABLED", "true").lower() == "true"
MC_DROPOUT_ITERATIONS = int(os.getenv("MC_DROPOUT_ITERATIONS", "5"))

# Text preprocessing
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 5000
