import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import config
from model_loader import ModelLoader
from openai_verifier import OpenAIVerifier
from utils import clean_text, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global model and verifier instances
model_loader: Optional[ModelLoader] = None
openai_verifier: Optional[OpenAIVerifier] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model loading on startup."""
    global model_loader, openai_verifier
    
    logger.info("Loading BERT model...")
    model_loader = ModelLoader()
    if not model_loader.load_model():
        logger.error("Failed to load BERT model")
        raise RuntimeError("Failed to load BERT model")
    
    logger.info("Initializing OpenAI verifier...")
    openai_verifier = OpenAIVerifier()
    if openai_verifier.enabled:
        logger.info("OpenAI verifier enabled")
    else:
        logger.warning("OpenAI verifier disabled or API key not set")
    
    logger.info("✓ API startup complete")
    
    yield
    
    logger.info("Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="BERT + OpenAI powered fake news detection system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic Models ====================

class PredictRequest(BaseModel):
    text: str
    verify_with_openai: bool = False
    mc_dropout: bool = False

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    openai_result: Optional[dict] = None
    combined_result: Optional[dict] = None

class BatchPredictRequest(BaseModel):
    texts: List[str]
    verify_with_openai: bool = False

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    openai_available: bool
    message: str

# ==================== Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if model_loader else "error",
        model_loaded=model_loader is not None,
        openai_available=openai_verifier.enabled if openai_verifier else False,
        message="API is running"
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict if news is real or fake."""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate input
    text = clean_text(request.text)
    if len(text) < config.MIN_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too short (minimum {config.MIN_TEXT_LENGTH} characters)"
        )
    if len(text) > config.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long (maximum {config.MAX_TEXT_LENGTH} characters)"
        )
    
    try:
        # BERT prediction
        bert_result = model_loader.predict(
            text,
            mc_dropout=request.mc_dropout
        )
        
        openai_result = None
        combined_result = None
        
        # OpenAI verification if requested
        if request.verify_with_openai and openai_verifier:
            logger.info("Performing OpenAI verification...")
            openai_result = openai_verifier.verify_news(text)
            
            # Combine results
            if openai_result.get("is_available"):
                combined_result = OpenAIVerifier.combine_verdicts(bert_result, openai_result)
        
        return PredictResponse(
            label=bert_result["label"],
            confidence=bert_result["confidence"],
            probabilities=bert_result["probabilities"],
            openai_result=openai_result,
            combined_result=combined_result
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(request: BatchPredictRequest):
    """Batch prediction endpoint."""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    try:
        results = []
        
        for text in request.texts:
            text = clean_text(text)
            
            if len(text) < config.MIN_TEXT_LENGTH or len(text) > config.MAX_TEXT_LENGTH:
                results.append({
                    "label": "unknown",
                    "confidence": 0.0,
                    "error": "Text length invalid"
                })
                continue
            
            try:
                bert_result = model_loader.predict(text)
                
                openai_result = None
                combined_result = None
                
                if request.verify_with_openai and openai_verifier:
                    openai_result = openai_verifier.verify_news(text)
                    if openai_result.get("is_available"):
                        combined_result = OpenAIVerifier.combine_verdicts(bert_result, openai_result)
                
                results.append({
                    "label": bert_result["label"],
                    "confidence": bert_result["confidence"],
                    "probabilities": bert_result["probabilities"],
                    "openai_result": openai_result,
                    "combined_result": combined_result
                })
                
            except Exception as e:
                results.append({
                    "label": "unknown",
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def get_model_info():
    """Get model information."""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": config.MODEL_NAME,
        "model_path": config.MODEL_PATH,
        "device": str(model_loader.device),
        "labels": model_loader.labels,
        "max_length": config.MAX_LENGTH,
        "temperature": config.TEMPERATURE,
        "mc_dropout_enabled": config.MC_DROPOUT_ENABLED,
        "mc_dropout_iterations": config.MC_DROPOUT_ITERATIONS,
        "openai_available": openai_verifier.enabled if openai_verifier else False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
