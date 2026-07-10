import asyncio
from litellm import completion
from litellm.exceptions import RateLimitError
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

MAX_RETRIES   = 4
RETRY_DELAYS  = [5, 15, 30, 60]   # secondes entre chaque tentative


class LiteLLMGateway:
    """Gateway unifiée pour accéder à plusieurs LLMs via LiteLLM"""

    def __init__(self, default_model: str):
        self.default_model = default_model

    async def _call_with_retry(self, **kwargs) -> Any:
        """Appel LiteLLM avec retry exponentiel sur RateLimitError"""
        for attempt, delay in enumerate(RETRY_DELAYS, start=1):
            try:
                return completion(**kwargs)
            except RateLimitError as e:
                if attempt > MAX_RETRIES:
                    logger.error(f"Rate limit dépassé après {MAX_RETRIES} tentatives")
                    raise
                logger.warning(
                    f"Rate limit atteint (tentative {attempt}/{MAX_RETRIES}). "
                    f"Nouvelle tentative dans {delay}s..."
                )
                await asyncio.sleep(delay)
            except Exception:
                raise

    async def chat_completion(self, messages: List[Dict[str, str]], model: str = None) -> Any:
        """Appel LLM standard (sans tools)"""
        model = model or self.default_model
        try:
            return await self._call_with_retry(model=model, messages=messages, temperature=0.7)
        except Exception as e:
            logger.error(f"Erreur LLM: {e}")
            raise

    async def chat_completion_with_tools(self, messages: List[Dict[str, str]], tools: List[Dict], model: str = None) -> Any:
        """Appel LLM avec Function Calling (tools)"""
        model = model or self.default_model
        try:
            return await self._call_with_retry(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Erreur LLM avec tools: {e}")
            raise
