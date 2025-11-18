import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import config
from model_loader import ModelLoader
from utils import clean_text, setup_logging
from database import db
import auth
from middleware import get_current_user, get_current_user_optional, get_current_admin, security
from models_pydantic import (
    UserRegister, UserLogin, Token,
    PredictRequest, PredictResponse,
    QueryHistoryResponse, QueryStatsResponse,
    HealthResponse
)

# Import AI verifiers (FREE only)
try:
    from gemini_verifier import GeminiVerifier
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiVerifier = None

try:
    from groq_verifier import GroqVerifier
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    GroqVerifier = None

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global model and verifier instances
model_loader: Optional[ModelLoader] = None
gemini_verifier: Optional[GeminiVerifier] = None
groq_verifier: Optional[GroqVerifier] = None
active_verifier = None  # Will be set to the first available verifier

def combine_all_verdicts(bert_result: dict, ai_results: list) -> dict:
    """Combine BERT and multiple AI verdicts using majority voting."""
    # Collect all verdicts
    verdicts = [bert_result["label"]]
    for name, result in ai_results:
        verdicts.append(result.get("verdict", "unknown"))
    
    # Count votes
    fake_votes = verdicts.count("fake")
    real_votes = verdicts.count("real")
    
    # Majority voting
    final_verdict = "fake" if fake_votes > real_votes else "real"
    
    # Calculate confidence (average of matching verdicts)
    matching_confidences = [bert_result["confidence"]]
    for name, result in ai_results:
        if result.get("verdict") == final_verdict:
            matching_confidences.append(result.get("confidence", 0.5))
    
    avg_confidence = sum(matching_confidences) / len(matching_confidences)
    
    # Build detailed breakdown
    breakdown = {
        "bert": {"verdict": bert_result["label"], "confidence": bert_result["confidence"]}
    }
    for name, result in ai_results:
        breakdown[name.lower()] = {
            "verdict": result.get("verdict", "unknown"),
            "confidence": result.get("confidence", 0.0)
        }
    
    return {
        "verdict": final_verdict,
        "confidence": avg_confidence,
        "total_votes": len(verdicts),
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "breakdown": breakdown,
        "agreement": "unanimous" if (fake_votes == len(verdicts) or real_votes == len(verdicts)) else "majority"
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model loading on startup."""
    global model_loader, gemini_verifier, groq_verifier, active_verifier
    
    # Connect to MongoDB
    logger.info("Connecting to MongoDB...")
    db.connect()
    
    # Create default admin user
    if db.connected:
        admin_password_hash = auth.hash_password("123456")
        db.create_admin_user("admin", "admin@fakenews.com", admin_password_hash)
        logger.info("✓ Admin user initialized")
    
    # Load models
    logger.info("Loading models...")
    model_loader = ModelLoader()
    
    # Load English model (required)
    if not model_loader.load_model_en():
        logger.error("Failed to load English model")
        raise RuntimeError("Failed to load English model")
    
    # Load Vietnamese model (optional)
    try:
        model_loader.load_model_vi()
    except Exception as e:
        logger.warning(f"Vietnamese model not loaded: {e}")
        logger.warning("Vietnamese predictions will not be available")
    
    # Initialize AI verifiers (FREE only)
    logger.info("Initializing AI verifiers...")
    
    # Gemini (recommended, free)
    if GEMINI_AVAILABLE and GeminiVerifier:
        gemini_verifier = GeminiVerifier()
        if gemini_verifier.enabled:
            logger.info("✓ Gemini verifier enabled (FREE)")
            if not active_verifier:
                active_verifier = gemini_verifier
    
    # Groq (fast, free)
    if GROQ_AVAILABLE and GroqVerifier:
        groq_verifier = GroqVerifier()
        if groq_verifier.enabled:
            logger.info("✓ Groq verifier enabled (FREE)")
            if not active_verifier:
                active_verifier = groq_verifier
    
    if not active_verifier:
        logger.warning("⚠ No AI verifier enabled - only BERT will be used")
    else:
        logger.info(f"✓ Active verifier: {active_verifier.__class__.__name__}")
    
    logger.info("✓ API startup complete")
    
    yield
    
    # Cleanup
    db.disconnect()
    logger.info("Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="BERT + AI (Gemini & Groq) powered fake news detection system - 100% FREE",
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

# ==================== Pydantic Models (Legacy - keeping for compatibility) ====================

class BatchPredictRequest(BaseModel):
    texts: List[str]
    language: str = "en"
    verify_with_openai: bool = False

# ==================== Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    ai_available = False
    if active_verifier:
        ai_available = active_verifier.enabled
    
    return HealthResponse(
        status="ok" if model_loader else "error",
        models_loaded=model_loader.models_loaded if model_loader else {"en": False, "vi": False},
        ai_verification_available=ai_available,
        database_connected=db.connected,
        message="API is running"
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    current_user: dict = Depends(get_current_user_optional)
):
    """Predict if news is real or fake (with language support)."""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate language
    if request.language not in config.SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {request.language}. Supported: {config.SUPPORTED_LANGUAGES}"
        )
    
    # Check if model for language is loaded
    if not model_loader.models_loaded.get(request.language):
        raise HTTPException(
            status_code=503,
            detail=f"Model for language '{request.language}' not loaded"
        )
    
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
        # BERT prediction with language support
        bert_result = model_loader.predict(
            text,
            language=request.language,
            mc_dropout=request.mc_dropout
        )
        
        ai_result = None
        gemini_result = None
        groq_result = None
        combined_result = None
        
        # AI verification if requested - Run ALL enabled verifiers (FREE only)
        if request.verify_with_ai:
            ai_results = []
            
            # Gemini
            if gemini_verifier and gemini_verifier.enabled:
                logger.info("Performing AI verification with Gemini...")
                gemini_result = gemini_verifier.verify_news(text)
                if gemini_result.get("is_available"):
                    ai_results.append(("Gemini", gemini_result))
            
            # Groq
            if groq_verifier and groq_verifier.enabled:
                logger.info("Performing AI verification with Groq...")
                groq_result = groq_verifier.verify_news(text)
                if groq_result.get("is_available"):
                    ai_results.append(("Groq", groq_result))
            
            # Combine results with majority voting
            if ai_results:
                combined_result = combine_all_verdicts(bert_result, ai_results)
                # Keep backward compatibility
                ai_result = ai_results[0][1] if ai_results else None
        
        # Save to database if user is logged in
        if current_user:
            logger.info(f"User authenticated: {current_user.get('username')}")
            if db.connected:
                try:
                    query_id = db.save_query(
                        user_id=current_user["_id"],
                        text=text,
                        language=request.language,
                        prediction={
                            "label": bert_result["label"],
                            "confidence": bert_result["confidence"],
                            "probabilities": bert_result["probabilities"],
                            "openai_result": ai_result,
                            "combined_result": combined_result,
                            "mc_dropout": request.mc_dropout
                        }
                    )
                    if query_id:
                        logger.info(f"✓ Query saved for user: {current_user['username']} (ID: {query_id})")
                    else:
                        logger.error(f"✗ Failed to save query - save_query returned None")
                except Exception as e:
                    logger.error(f"✗ Failed to save query: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.warning(f"⚠ Database not connected - query not saved")
        else:
            logger.info("ℹ No user authenticated - query not saved")
        
        return PredictResponse(
            label=bert_result["label"],
            confidence=bert_result["confidence"],
            probabilities=bert_result["probabilities"],
            language=request.language,
            gemini_result=gemini_result,
            groq_result=groq_result,
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
        "models_loaded": model_loader.models_loaded,
        "supported_languages": config.SUPPORTED_LANGUAGES,
        "device": str(model_loader.device),
        "labels": model_loader.labels,
        "max_length": config.MAX_LENGTH,
        "temperature": config.TEMPERATURE,
        "mc_dropout_enabled": config.MC_DROPOUT_ENABLED,
        "mc_dropout_iterations": config.MC_DROPOUT_ITERATIONS,
        "ai_verification_available": active_verifier is not None,
        "database_connected": db.connected
    }

# ==================== Authentication Endpoints ====================

@app.post("/auth/register", response_model=Token)
async def register(user_data: UserRegister):
    """Register new user"""
    if not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Validate
    valid, msg = auth.validate_username(user_data.username)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)
    
    valid, msg = auth.validate_email(user_data.email)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)
    
    valid, msg = auth.validate_password(user_data.password)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)
    
    # Check if exists
    if db.get_user_by_username(user_data.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    if db.get_user_by_email(user_data.email):
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Create user
    password_hash = auth.hash_password(user_data.password)
    user_id = db.create_user(user_data.username, user_data.email, password_hash)
    
    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to create user")
    
    # Generate token
    access_token = auth.create_access_token(data={"sub": user_data.username})
    
    user = db.get_user_by_username(user_data.username)
    
    return Token(
        access_token=access_token,
        user={
            "username": user["username"],
            "email": user["email"],
            "created_at": user["created_at"].isoformat() if user.get("created_at") else None
        }
    )

@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login user"""
    if not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    user = db.get_user_by_username(credentials.username)
    
    if not user or not auth.verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Update last login
    db.update_last_login(credentials.username)
    
    # Generate token
    access_token = auth.create_access_token(data={"sub": credentials.username})
    
    return Token(
        access_token=access_token,
        user={
            "username": user["username"],
            "email": user["email"],
            "role": user.get("role", "user"),
            "created_at": user["created_at"].isoformat() if user.get("created_at") else None
        }
    )

@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user"""
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"].isoformat() if current_user.get("created_at") else None,
        "last_login": current_user["last_login"].isoformat() if current_user.get("last_login") else None
    }

# ==================== Query History Endpoints ====================

@app.get("/history")
async def get_history(
    limit: int = 50,
    skip: int = 0,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get user's query history"""
    if not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Get current user
    current_user = await get_current_user(credentials)
    
    queries = db.get_user_queries(current_user["_id"], limit, skip)
    total = db.queries.count_documents({"user_id": current_user["_id"]})
    
    return {
        "queries": queries,
        "total": total,
        "page": skip // limit + 1 if limit > 0 else 1,
        "limit": limit
    }

@app.get("/history/stats")
async def get_stats(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get user's query statistics"""
    if not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Get current user
    current_user = await get_current_user(credentials)
    
    stats = db.get_query_stats(current_user["_id"])
    return stats

# ==================== Admin Endpoints ====================

@app.get("/admin/users")
async def get_all_users(
    skip: int = 0,
    limit: int = 50,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get all users (admin only)"""
    if not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Verify admin
    admin_user = await get_current_admin(credentials)
    
    users = db.get_all_users(skip, limit)
    total = db.users.count_documents({})
    
    return {
        "users": users,
        "total": total,
        "page": skip // limit + 1 if limit > 0 else 1,
        "limit": limit
    }

@app.get("/admin/stats")
async def get_system_stats(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get system-wide statistics (admin only)"""
    if not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Verify admin
    admin_user = await get_current_admin(credentials)
    
    stats = db.get_system_stats()
    return stats

@app.delete("/admin/users/{user_id}")
async def delete_user(
    user_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Delete a user (admin only)"""
    if not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Verify admin
    admin_user = await get_current_admin(credentials)
    
    # Prevent admin from deleting themselves
    if admin_user["_id"] == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    # Delete user
    from bson import ObjectId
    from bson.errors import InvalidId
    
    try:
        # Try to convert to ObjectId
        object_id = ObjectId(user_id)
        logger.info(f"Attempting to delete user with ObjectId: {object_id}")
        
        result = db.users.delete_one({"_id": object_id})
        
        if result.deleted_count == 0:
            logger.warning(f"User not found with ObjectId: {object_id}")
            raise HTTPException(status_code=404, detail="User not found")
        
        # Also delete user's queries
        db.queries.delete_many({"user_id": user_id})
        
        logger.info(f"User deleted successfully: {user_id}")
        return {"message": "User deleted successfully"}
        
    except InvalidId:
        logger.error(f"Invalid ObjectId format: {user_id}")
        raise HTTPException(status_code=400, detail="Invalid user ID format")

@app.put("/admin/users/{user_id}")
async def update_user(
    user_id: str,
    email: Optional[str] = None,
    role: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Update user (admin only)"""
    if not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Verify admin
    admin_user = await get_current_admin(credentials)
    
    # Build update dict
    update_data = {}
    if email:
        update_data["email"] = email
    if role and role in ["user", "admin"]:
        # Prevent admin from demoting themselves
        if admin_user["_id"] == user_id and role != "admin":
            raise HTTPException(status_code=400, detail="Cannot change your own role")
        update_data["role"] = role
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    
    # Update user
    from bson import ObjectId
    from bson.errors import InvalidId
    
    try:
        object_id = ObjectId(user_id)
        logger.info(f"Updating user {object_id} with data: {update_data}")
        
        result = db.users.update_one(
            {"_id": object_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            logger.warning(f"User not found with ObjectId: {object_id}")
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"User updated successfully: {user_id}")
        return {"message": "User updated successfully"}
        
    except InvalidId:
        logger.error(f"Invalid ObjectId format: {user_id}")
        raise HTTPException(status_code=400, detail="Invalid user ID format")

@app.get("/admin/queries")
async def get_all_queries(
    skip: int = 0,
    limit: int = 50,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get all queries (admin only)"""
    if not db.connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Verify admin
    admin_user = await get_current_admin(credentials)
    
    queries = list(
        db.queries.find({})
        .sort("timestamp", -1)
        .skip(skip)
        .limit(limit)
    )
    
    for query in queries:
        query["_id"] = str(query["_id"])
        query["timestamp"] = query["timestamp"].isoformat()
    
    total = db.queries.count_documents({})
    
    return {
        "queries": queries,
        "total": total,
        "page": skip // limit + 1 if limit > 0 else 1,
        "limit": limit
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )
