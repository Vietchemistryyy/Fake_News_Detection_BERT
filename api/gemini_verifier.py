import json
import logging
import os
from typing import Dict, Any, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from utils import truncate_text

logger = logging.getLogger(__name__)

class GeminiVerifier:
    """Google Gemini-based fake news verifier (FREE & MODERN)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.enabled = bool(self.api_key) and GEMINI_AVAILABLE and os.getenv("ENABLE_GEMINI", "false").lower() == "true"
        
        if not GEMINI_AVAILABLE:
            logger.warning("google-generativeai not installed. Install: pip install google-generativeai")
            self.enabled = False
        
        if self.enabled:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                logger.info(f"Gemini verifier initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.enabled = False
        else:
            self.model = None
    
    def verify_news(self, text: str) -> Dict[str, Any]:
        """Verify news article using Google Gemini."""
        if not self.enabled:
            return self._default_response("Gemini verification disabled")
        
        try:
            truncated_text = truncate_text(text, 1000)  # Gemini can handle more
            
            prompt = f"""You are a professional fact-checker. Analyze this news article and determine if it's real or fake.

IMPORTANT GUIDELINES:
1. Real news can be brief summaries without sources
2. Only mark as "fake" if you find clear misinformation or fabricated claims
3. Consider writing quality, logical consistency, and factual plausibility
4. If uncertain, lean toward "real" with lower confidence

Article to analyze:
{truncated_text}

Respond ONLY with valid JSON (no markdown):
{{
    "verdict": "real" or "fake",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "concerns": ["concern1", "concern2"]
}}"""

            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=300,
                )
            )
            
            response_text = response.text.strip()
            
            # Parse JSON response
            result = self._parse_response(response_text)
            
            # Validate and adjust
            result = self._validate_result(result)
            result["is_available"] = True
            result["provider"] = "gemini"
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini verification error: {str(e)}")
            return self._error_response(str(e))
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from Gemini response."""
        # Remove markdown code blocks if present
        text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            
            logger.warning(f"Could not parse Gemini response: {text}")
            return {"error": "Failed to parse response"}
    
    def _validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust result."""
        if "error" in result:
            return self._default_response("Parse error")
        
        # Ensure required fields
        if "verdict" not in result or "confidence" not in result:
            return self._default_response("Invalid response structure")
        
        # Normalize verdict
        verdict = result["verdict"].lower()
        if verdict not in ["real", "fake"]:
            verdict = "real" if result.get("confidence", 0.5) < 0.5 else "fake"
        result["verdict"] = verdict
        
        # Ensure confidence is float
        result["confidence"] = float(result.get("confidence", 0.5))
        
        # Cap extreme confidence
        if result["confidence"] > 0.90:
            result["confidence"] = min(0.85, result["confidence"])
        
        # Ensure other fields
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
            "provider": "gemini"
        }
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """Return error response."""
        return {
            "verdict": "unknown",
            "confidence": 0.5,
            "reasoning": f"Gemini API error: {error}",
            "concerns": ["API connection failed"],
            "is_available": False,
            "provider": "gemini"
        }
    
    @staticmethod
    def combine_verdicts(
        bert_result: Dict[str, Any],
        gemini_result: Dict[str, Any],
        bert_weight: float = 0.6
    ) -> Dict[str, Any]:
        """Combine BERT and Gemini verdicts."""
        
        # Extract confidence scores
        bert_fake_conf = bert_result.get("probabilities", {}).get("fake", 0.5)
        gemini_verdict = gemini_result.get("verdict", "unknown").lower()
        gemini_conf = gemini_result.get("confidence", 0.5)
        
        # Convert Gemini verdict to fake probability
        gemini_fake_conf = gemini_conf if gemini_verdict == "fake" else (1 - gemini_conf)
        
        # Weighted average
        gemini_weight = 1 - bert_weight
        combined_fake_conf = (bert_fake_conf * bert_weight) + (gemini_fake_conf * gemini_weight)
        
        # Determine final verdict with threshold
        threshold = 0.55
        final_verdict = "fake" if combined_fake_conf > threshold else "real"
        final_confidence = max(combined_fake_conf, 1 - combined_fake_conf)
        
        return {
            "verdict": final_verdict,
            "confidence": float(final_confidence),
            "bert_verdict": bert_result.get("label", "unknown"),
            "bert_confidence": float(bert_result.get("confidence", 0.5)),
            "gemini_verdict": gemini_verdict,
            "gemini_confidence": float(gemini_result.get("confidence", 0.5)),
            "reasoning": gemini_result.get("reasoning", ""),
            "concerns": gemini_result.get("concerns", []),
            "combined_fake_probability": float(combined_fake_conf),
            "threshold_used": threshold,
            "provider": "gemini"
        }
