import json
import logging
import os
from typing import Dict, Any, Optional

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from utils import truncate_text

logger = logging.getLogger(__name__)

class GroqVerifier:
    """Groq-based fake news verifier (FREE & FAST with Llama 3)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.enabled = bool(self.api_key) and GROQ_AVAILABLE and os.getenv("ENABLE_GROQ", "false").lower() == "true"
        
        if not GROQ_AVAILABLE:
            logger.warning("groq not installed. Install: pip install groq")
            self.enabled = False
        
        if self.enabled:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Groq verifier initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq: {e}")
                self.enabled = False
        else:
            self.client = None
    
    def verify_news(self, text: str) -> Dict[str, Any]:
        """Verify news article using Groq (Llama 3)."""
        if not self.enabled:
            return self._default_response("Groq verification disabled")
        
        try:
            truncated_text = truncate_text(text, 800)
            
            system_prompt = """You are a fact-checking AI. Analyze news articles objectively.

Guidelines:
- Only mark as "fake" if you find clear misinformation
- Real news can be brief without sources
- Consider writing quality and logical consistency
- If uncertain, lean toward "real" with lower confidence

Respond ONLY with valid JSON:
{
    "verdict": "real" or "fake",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "concerns": ["concern1", "concern2"]
}"""

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this news:\n\n{truncated_text}"}
                ],
                temperature=0.3,
                max_tokens=300,
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            # Parse JSON response
            result = self._parse_response(response_text)
            
            # Validate and adjust
            result = self._validate_result(result)
            result["is_available"] = True
            result["provider"] = "groq"
            
            return result
            
        except Exception as e:
            logger.error(f"Groq verification error: {str(e)}")
            return self._error_response(str(e))
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Groq response."""
        text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            
            logger.warning(f"Could not parse Groq response: {text}")
            return {"error": "Failed to parse response"}
    
    def _validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust result."""
        if "error" in result:
            return self._default_response("Parse error")
        
        if "verdict" not in result or "confidence" not in result:
            return self._default_response("Invalid response structure")
        
        verdict = result["verdict"].lower()
        if verdict not in ["real", "fake"]:
            verdict = "real" if result.get("confidence", 0.5) < 0.5 else "fake"
        result["verdict"] = verdict
        
        result["confidence"] = float(result.get("confidence", 0.5))
        
        # Cap extreme confidence
        if result["confidence"] > 0.90:
            result["confidence"] = min(0.85, result["confidence"])
        
        result["reasoning"] = result.get("reasoning", "No reasoning provided")
        result["concerns"] = result.get("concerns", [])
        
        return result
    
    def _default_response(self, reason: str = "Verification disabled") -> Dict[str, Any]:
        """Return default response."""
        return {
            "verdict": "unknown",
            "confidence": 0.5,
            "reasoning": reason,
            "concerns": [],
            "is_available": False,
            "provider": "groq"
        }
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """Return error response."""
        return {
            "verdict": "unknown",
            "confidence": 0.5,
            "reasoning": f"Groq API error: {error}",
            "concerns": ["API connection failed"],
            "is_available": False,
            "provider": "groq"
        }
