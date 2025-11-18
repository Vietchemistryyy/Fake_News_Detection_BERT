import json
import logging
from typing import Dict, Any, Optional
import config
from utils import truncate_text, parse_json_response

logger = logging.getLogger(__name__)

class GeminiVerifier:
    """Gemini-based fake news verifier."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.enabled = bool(self.api_key) and config.ENABLE_GEMINI
        self.client = None
        
        if self.enabled:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                # Use model from config (gemini-2.5-flash for students)
                model_name = config.GEMINI_MODEL
                self.client = genai.GenerativeModel(model_name)
                logger.info(f"✓ Gemini verifier initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.enabled = False
                self.client = None
        else:
            logger.warning("Gemini verifier disabled (no API key or ENABLE_GEMINI=false)")
    
    def verify_news(self, text: str) -> Dict[str, Any]:
        """Verify news article using Gemini."""
        if not self.enabled or not self.client:
            return self._default_response()
        
        try:
            truncated_text = truncate_text(text, 500)
            
            prompt = f"""You are a professional news analyst. Evaluate the credibility of this article.

Guidelines:
- Assess writing quality and logical consistency
- Check for verifiable claims
- Consider journalistic standards
- Be objective and balanced

Respond with JSON only:
{{
    "verdict": "real" or "fake",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "concerns": ["concern1", "concern2"]
}}

Text to analyze:
{truncated_text}"""

            # Generate content with Gemini (disable safety filters)
            import google.generativeai as genai
            
            safety_settings = []
            
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 200,
                },
                safety_settings=safety_settings
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning(f"Gemini response blocked. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'}")
                return self._error_response()
            
            response_text = response.text.strip()
            result = parse_json_response(response_text)
            
            # Validate response
            if "verdict" not in result or "confidence" not in result:
                logger.warning(f"Invalid Gemini response: {result}")
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
            logger.error(f"Gemini verification error: {str(e)}")
            return self._error_response()
    
    def _default_response(self) -> Dict[str, Any]:
        """Return default response when verification is disabled."""
        return {
            "verdict": "unknown",
            "confidence": 0.5,
            "reasoning": "Gemini verification disabled",
            "concerns": [],
            "is_available": False
        }
    
    def _error_response(self) -> Dict[str, Any]:
        """Return error response."""
        return {
            "verdict": "unknown",
            "confidence": 0.5,
            "reasoning": "Gemini API error occurred",
            "concerns": ["API connection failed"],
            "is_available": False
        }
