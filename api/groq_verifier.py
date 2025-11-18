import json
import logging
from typing import Dict, Any, Optional
import config
from utils import truncate_text, parse_json_response

logger = logging.getLogger(__name__)

class GroqVerifier:
    """Groq-based fake news verifier (Llama 3.1)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = config.GROQ_MODEL
        self.enabled = bool(self.api_key) and config.ENABLE_GROQ
        
        if self.enabled:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info(f"✓ Groq verifier initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq: {e}")
                self.enabled = False
                self.client = None
        else:
            self.client = None
            logger.warning("Groq verifier disabled (no API key or ENABLE_GROQ=false)")
    
    def verify_news(self, text: str) -> Dict[str, Any]:
        """Verify news article using Groq."""
        if not self.enabled:
            return self._default_response()
        
        try:
            truncated_text = truncate_text(text, 500)
            
            system_prompt = """You are a neutral fact-checking assistant. Analyze the given news article objectively.

IMPORTANT GUIDELINES:
1. Do NOT assume an article is fake just because it lacks sources
2. Legitimate news articles often summarize without listing sources
3. Only flag as "fake" if you find clear misinformation or fabricated claims
4. If the content seems plausible but unverifiable, lean toward "real" with lower confidence

Respond ONLY with valid JSON (no markdown, no extra text):
{
    "verdict": "real" or "fake",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "concerns": ["concern1", "concern2"]
}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this news article:\n\n{truncated_text}"
                    }
                ],
                temperature=0.3,
                max_tokens=200,
            )
            
            response_text = response.choices[0].message.content.strip()
            result = parse_json_response(response_text)
            
            # Validate response
            if "verdict" not in result or "confidence" not in result:
                logger.warning(f"Invalid Groq response: {result}")
                return self._default_response()
            
            # Ensure verdict is real/fake
            if result["verdict"].lower() not in ["real", "fake"]:
                result["verdict"] = "real" if result.get("confidence", 0.5) < 0.5 else "fake"
            
            result["is_available"] = True
            
            # Adjust confidence if too certain without evidence
            if result["confidence"] > 0.85 and len(result.get("concerns", [])) < 2:
                result["confidence"] = min(0.75, result["confidence"])
            
            return result
            
        except Exception as e:
            logger.error(f"Groq verification error: {str(e)}")
            return self._error_response()
    
    def _default_response(self) -> Dict[str, Any]:
        """Return default response when verification is disabled."""
        return {
            "verdict": "unknown",
            "confidence": 0.5,
            "reasoning": "Groq verification disabled",
            "concerns": [],
            "is_available": False
        }
    
    def _error_response(self) -> Dict[str, Any]:
        """Return error response."""
        return {
            "verdict": "unknown",
            "confidence": 0.5,
            "reasoning": "Groq API error occurred",
            "concerns": ["API connection failed"],
            "is_available": False
        }
