"""Pydantic models for API"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ==================== Auth Models ====================

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]

class UserResponse(BaseModel):
    username: str
    email: str
    created_at: datetime
    last_login: Optional[datetime] = None

# ==================== Prediction Models ====================

class PredictRequest(BaseModel):
    text: str
    language: str = "en"  # "en" or "vi"
    verify_with_ai: bool = False
    mc_dropout: bool = False

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    language: str
    groq_result: Optional[dict] = None
    combined_result: Optional[dict] = None

# ==================== Query History Models ====================

class QueryHistoryResponse(BaseModel):
    queries: List[Dict[str, Any]]
    total: int
    page: int
    limit: int

class QueryStatsResponse(BaseModel):
    total_queries: int
    by_language: Dict[str, int]
    by_prediction: Dict[str, int]

# ==================== Health Check ====================

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    groq_available: bool
    database_connected: bool
    message: str
