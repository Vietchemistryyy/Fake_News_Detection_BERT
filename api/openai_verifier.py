import json
import logging
import asyncio
from typing import Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI
import config
from utils import truncate_text, parse_json_response

logger = logging.getLogger(__name__)

class OpenAIVerifier:
    """OpenAI-based fake news verifier (UNBIASED VERSION)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = config.OPENAI_MODEL
        self.enabled = bool(self.api_key) and config.ENABLE_OPENAI
        
        if self.enabled:
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None
            self.async_client = None
    
    def verify_news(self, text: str) -> Dict[str, Any]:
        """Synchronously verify news article using OpenAI (UNBIASED)."""
        if not self.enabled:
            return {
                "verdict": "unknown",
                "confidence": 0.0,
                "reasoning": "OpenAI verification disabled",
                "concerns": [],
                "is_available": False
            }
        
        try:
            truncated_text = truncate_text(text, 500)
            
            # 🔧 FIX 1: Improved system prompt (less biased)
            system_prompt = """You are a neutral fact-checking assistant. Analyze the given news article objectively.

IMPORTANT GUIDELINES:
1. Do NOT assume an article is fake just because it lacks sources
2. Legitimate news articles from Reuters, AP, BBC often summarize without listing sources
3. Only flag as "fake" if you find clear misinformation, contradictions, or fabricated claims
4. If the content seems plausible but unverifiable, lean toward "real" with lower confidence
5. Consider the writing quality and journalistic style

Respond ONLY with valid JSON (no markdown, no extra text) in this format:
{
    "verdict": "real" or "fake",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "concerns": ["concern1", "concern2"]
}

Evaluation criteria:
- Writing quality and professionalism
- Logical consistency
- Emotional manipulation indicators
- Verifiable factual claims
- Source credibility signals"""

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
                temperature=0.3,  # 🔧 FIX 2: Lower temperature for more consistent results
                max_tokens=200,
                timeout=config.OPENAI_TIMEOUT
            )
            
            response_text = response.choices[0].message.content.strip()
            result = parse_json_response(response_text)
            
            # Validate response structure
            if "verdict" not in result or "confidence" not in result:
                logger.warning(f"Invalid OpenAI response structure: {result}")
                return self._default_response()
            
            # Ensure verdict is real/fake
            if result["verdict"].lower() not in ["real", "fake"]:
                result["verdict"] = "real" if result.get("confidence", 0.5) < 0.5 else "fake"
            
            result["is_available"] = True
            
            # 🔧 FIX 3: Adjust confidence if too certain without evidence
            if result["confidence"] > 0.85 and len(result.get("concerns", [])) < 2:
                result["confidence"] = min(0.75, result["confidence"])
                logger.info(f"Adjusted confidence to {result['confidence']} (lack of strong evidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI verification error: {str(e)}")
            return self._error_response()
    
    async def verify_news_async(self, text: str) -> Dict[str, Any]:
        """Asynchronously verify news article using OpenAI."""
        if not self.enabled:
            return {
                "verdict": "unknown",
                "confidence": 0.0,
                "reasoning": "OpenAI verification disabled",
                "concerns": [],
                "is_available": False
            }
        
        try:
            truncated_text = truncate_text(text, 500)
            
            system_prompt = """You are a neutral fact-checking assistant. Analyze the given news article objectively.

IMPORTANT GUIDELINES:
1. Do NOT assume an article is fake just because it lacks sources
2. Legitimate news articles from Reuters, AP, BBC often summarize without listing sources
3. Only flag as "fake" if you find clear misinformation, contradictions, or fabricated claims
4. If the content seems plausible but unverifiable, lean toward "real" with lower confidence
5. Consider the writing quality and journalistic style

Respond ONLY with valid JSON (no markdown, no extra text) in this format:
{
    "verdict": "real" or "fake",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "concerns": ["concern1", "concern2"]
}

Evaluation criteria:
- Writing quality and professionalism
- Logical consistency
- Emotional manipulation indicators
- Verifiable factual claims
- Source credibility signals"""
            
            response = await self.async_client.chat.completions.create(
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
                timeout=config.OPENAI_TIMEOUT
            )
            
            response_text = response.choices[0].message.content.strip()
            result = parse_json_response(response_text)
            
            if "verdict" not in result or "confidence" not in result:
                logger.warning(f"Invalid OpenAI response structure: {result}")
                return self._default_response()
            
            if result["verdict"].lower() not in ["real", "fake"]:
                result["verdict"] = "real" if result.get("confidence", 0.5) < 0.5 else "fake"
            
            result["is_available"] = True
            
            # Adjust confidence if too certain without evidence
            if result["confidence"] > 0.85 and len(result.get("concerns", [])) < 2:
                result["confidence"] = min(0.75, result["confidence"])
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI async verification error: {str(e)}")
            return self._error_response()
    
    def _default_response(self) -> Dict[str, Any]:
        """Return default response when verification is disabled."""
        return {
            "verdict": "unknown",
            "confidence": 0.5,
            "reasoning": "Could not verify with OpenAI",
            "concerns": [],
            "is_available": False
        }
    
    def _error_response(self) -> Dict[str, Any]:
        """Return error response."""
        return {
            "verdict": "unknown",
            "confidence": 0.5,
            "reasoning": "OpenAI API error occurred",
            "concerns": ["API connection failed"],
            "is_available": False
        }
    
    @staticmethod
    def combine_verdicts(
        bert_result: Dict[str, Any],
        openai_result: Dict[str, Any],
        bert_weight: float = 0.5  # 🔧 FIX 4: Equal weight (was 0.6)
    ) -> Dict[str, Any]:
        """Combine BERT and OpenAI verdicts into final prediction (BALANCED)."""
        
        # Extract confidence scores
        bert_fake_conf = bert_result.get("probabilities", {}).get("fake", 0.5)
        openai_verdict = openai_result.get("verdict", "unknown").lower()
        openai_conf = openai_result.get("confidence", 0.5)
        
        # Convert OpenAI verdict to fake probability
        openai_fake_conf = openai_conf if openai_verdict == "fake" else (1 - openai_conf)
        
        # 🔧 FIX 5: Weighted average with equal importance
        openai_weight = 1 - bert_weight
        combined_fake_conf = (bert_fake_conf * bert_weight) + (openai_fake_conf * openai_weight)
        
        # 🔧 FIX 6: Require higher threshold for "fake" verdict
        threshold = 0.55  # Instead of 0.5
        final_verdict = "fake" if combined_fake_conf > threshold else "real"
        final_confidence = max(combined_fake_conf, 1 - combined_fake_conf)
        
        return {
            "verdict": final_verdict,
            "confidence": float(final_confidence),
            "bert_verdict": bert_result.get("label", "unknown"),
            "bert_confidence": float(bert_result.get("confidence", 0.5)),
            "openai_verdict": openai_verdict,
            "openai_confidence": float(openai_result.get("confidence", 0.5)),
            "reasoning": openai_result.get("reasoning", ""),
            "concerns": openai_result.get("concerns", []),
            "combined_fake_probability": float(combined_fake_conf),
            "threshold_used": threshold
        }