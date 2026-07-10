from litellm import completion
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LiteLLMGateway:
    """Gateway unifiée pour accéder à plusieurs LLMs via LiteLLM"""
    
    def __init__(self, default_model: str):
        self.default_model = default_model
    
    async def chat_completion(self, messages: List[Dict[str, str]], model: str = None) -> Any:
        """Appel LLM standard (sans tools)"""
        try:
            model = model or self.default_model
            response = completion(
                model=model,
                messages=messages,
                temperature=0.7
            )
            return response
        except Exception as e:
            logger.error(f"❌ Erreur LLM: {e}")
            raise
    
    async def chat_completion_with_tools(self, messages: List[Dict[str, str]], tools: List[Dict], model: str = None) -> Any:
        """Appel LLM avec Function Calling (tools)"""
        try:
            model = model or self.default_model
            response = completion(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7
            )
            return response
        except Exception as e:
            logger.error(f"❌ Erreur LLM avec tools: {e}")
            raise
