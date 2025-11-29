import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean input text for model inference."""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_language(text: str) -> str:
    """
    Detect language of text (Vietnamese or English)
    
    Args:
        text: Input text
        
    Returns:
        'vi' for Vietnamese, 'en' for English
    """
    if not text:
        return 'en'
    
    # Vietnamese specific characters
    vietnamese_chars = 'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
    vietnamese_chars += vietnamese_chars.upper()
    
    # Count Vietnamese characters
    viet_count = sum(1 for c in text if c in vietnamese_chars)
    
    # If more than 5% of characters are Vietnamese, classify as Vietnamese
    if len(text) > 0 and (viet_count / len(text)) > 0.05:
        return 'vi'
    
    return 'en'


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text for OpenAI API (token limit considerations)."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def parse_json_response(text: str) -> Dict[str, Any]:
    """Extract JSON from text, handling markdown code blocks."""
    try:
        # Try direct JSON parsing
        import json
        return json.loads(text)
    except:
        pass
    
    # Try extracting from markdown code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            import json
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Try extracting any JSON-like object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            import json
            return json.loads(json_match.group(0))
        except:
            pass
    
    logger.warning(f"Could not parse JSON from response: {text}")
    return {"error": "Failed to parse response"}

def setup_logging():
    """Configure logging for the API."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('api.log')
        ]
    )
