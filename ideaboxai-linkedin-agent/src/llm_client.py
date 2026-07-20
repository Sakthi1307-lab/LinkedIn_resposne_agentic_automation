import logging
import re
from typing import Tuple

import requests
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings

logger = logging.getLogger(__name__)

def _resolve_base_url(key: str) -> str:
    """
    Pick the API base URL from the key (or an explicit override).

    - sk-or-... keys are OpenRouter keys      -> OpenRouter base URL
    - any other sk-... key is a direct OpenAI key -> OpenAI base URL
    Both speak the same OpenAI-compatible API, so the client code is identical;
    only the base URL and model naming differ (see _normalize_model).
    """
    override = getattr(settings, "llm_base_url", None)
    if override:
        return override
    if key.startswith("sk-or-"):
        return "https://openrouter.ai/api/v1"
    return "https://api.openai.com/v1"


_api_key = settings.openrouter_api_key or ""
_base_url = _resolve_base_url(_api_key)
# True when talking to OpenAI directly (so we must strip provider prefixes and
# can't route to Claude models).
_is_openai_direct = _base_url.startswith("https://api.openai.com")

_client = None
if _api_key:
    # OpenRouter and OpenAI both expose an OpenAI-compatible API. The installed
    # `openai` SDK is v1.x (client-object pattern), not the v0.x module-level API.
    _client = OpenAI(api_key=_api_key, base_url=_base_url)
    logger.info(f"LLM client using base URL: {_base_url}")

# Exceptions worth retrying: rate limits, transient network issues, and
# 5xx server errors. AuthenticationError is deliberately excluded — a bad
# or expired key needs a human to fix it, and retrying just burns time
# and 2 extra API calls while hiding the real problem.
_RETRYABLE_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class LLMClient:
    """
    OpenRouter LLM client with model routing and retry logic.
    Models: Claude (high-quality), GPT-4 (balanced), GPT-3.5 (fast/cheap)
    """

    def __init__(self):
        self.model_high_stakes = settings.openrouter_model_high_stakes
        self.model_standard = settings.openrouter_model_standard
        self.model_fast = settings.openrouter_model_fast

        logger.info("LLMClient initialized:")
        logger.info(f"  High-stakes model: {self.model_high_stakes}")
        logger.info(f"  Standard model: {self.model_standard}")
        logger.info(f"  Fast model: {self.model_fast}")

    def select_model(self, persona_tier: str) -> str:
        """
        Route to the right model based on who we're replying to.

        Tier logic:
        - A_internal_leadership: High-stakes (settings.openrouter_model_high_stakes)
        - B_external_vip: High-stakes (settings.openrouter_model_high_stakes)
        - C_decision_maker: Standard (settings.openrouter_model_standard)
        - D_general / anything else: Fast/cheap (settings.openrouter_model_fast)
        """
        if persona_tier in ("A_internal_leadership", "B_external_vip"):
            return self.model_high_stakes
        elif persona_tier == "C_decision_maker":
            return self.model_standard
        else:
            return self.model_fast

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def generate_reply(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> Tuple[str, int]:
        """
        Call OpenRouter to generate a reply.

        Args:
            system_prompt: Brand voice + persona guidance
            user_message: The engagement + context
            model: Model name (e.g., settings.openrouter_model_high_stakes)
            max_tokens: Max output tokens (default 150)
            temperature: Creativity (0.7 = balanced)

        Returns:
            (generated_text, tokens_used)

        Retries up to 3 times with exponential backoff on rate limits,
        timeouts, connection errors, and 5xx server errors. Authentication
        errors are NOT retried — they propagate immediately since retrying
        a bad key can't ever succeed.
        """
        try:
            if _client is None:
                logger.warning("OPENROUTER_API_KEY is not configured; using local fallback reply")
                return self._fallback_reply(system_prompt, user_message), 0

            logger.info(f"Generating reply with model: {model}")

            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            reply = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0

            logger.info(f"Reply generated ({model}): {tokens} tokens")
            return reply, tokens

        except AuthenticationError as e:
            logger.error(f"OpenRouter authentication failed (check OPENROUTER_API_KEY): {e}")
            raise
        except RateLimitError as e:
            logger.warning(f"OpenRouter rate limit, retrying: {e}")
            raise
        except (APITimeoutError, APIConnectionError) as e:
            logger.warning(f"OpenRouter connectivity issue, retrying: {e}")
            raise
        except InternalServerError as e:
            logger.warning(f"OpenRouter server error, retrying: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating reply: {e}", exc_info=True)
            raise

    def check_connectivity(self) -> dict:
        """
        Pre-flight check: is this client actually able to reach OpenRouter,
        or silently running in local-fallback stub mode?

        _client is constructed once, at module import time, from whatever
        OPENROUTER_API_KEY was present then (see the module-level `if
        settings.openrouter_api_key:` block above). An unset, empty, or
        placeholder-rejected key means _client is None forever for this
        process — every generate_reply() call after that quietly returns
        _fallback_reply()'s static canned string instead of ever calling
        OpenRouter, with no exception and no error log. Every prompt
        variation in response_generator.py is inert against that path,
        since the text never depends on what was actually asked. This
        check makes that condition loud and explicit instead of silently
        producing plausible-looking replies that aren't real generation.

        Provider-aware: an OpenRouter key is checked against OpenRouter's
        key-info endpoint (GET /auth/key — validates the key, costs
        nothing); a direct OpenAI key has no equivalent free endpoint, so
        it's checked with GET /models instead (also free, standard way to
        validate a key without spending tokens). See _resolve_base_url /
        _is_openai_direct above for how the provider is picked.
        """
        if _client is None:
            return {
                "ok": False,
                "mode": "local_fallback_stub",
                "detail": (
                    "OPENROUTER_API_KEY is not set (or was rejected as a placeholder). "
                    "Every reply will be one of three static canned strings from "
                    "_fallback_reply(), not real generation."
                ),
            }

        provider = "OpenAI" if _is_openai_direct else "OpenRouter"
        try:
            if _is_openai_direct:
                response = requests.get(
                    f"{_base_url}/models",
                    headers={"Authorization": f"Bearer {_api_key}"},
                    timeout=10,
                )
                if response.status_code == 200:
                    return {"ok": True, "mode": "live", "detail": "OpenAI key valid."}
            else:
                response = requests.get(
                    f"{_base_url}/auth/key",
                    headers={"Authorization": f"Bearer {_api_key}"},
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json().get("data", {})
                    return {
                        "ok": True,
                        "mode": "live",
                        "detail": f"Key valid. Limit: {data.get('limit')}, usage: {data.get('usage')}.",
                    }

            if response.status_code == 401:
                return {"ok": False, "mode": "live_invalid_key", "detail": f"{provider} rejected the API key (401)."}
            return {
                "ok": False,
                "mode": "live_error",
                "detail": f"{provider} returned {response.status_code}: {response.text[:200]}",
            }
        except requests.exceptions.Timeout:
            return {"ok": False, "mode": "network_error", "detail": f"Timed out reaching {provider} (10s)."}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "mode": "network_error", "detail": f"Could not reach {provider}: {e}"}

    def _fallback_reply(self, system_prompt: str, user_message: str) -> str:
        """
        Local offline reply used when OpenRouter is unavailable.

        This keeps local personal-account testing unblocked without external
        API credentials.
        """
        match = re.search(r'Engagement from (.+?) \(', user_message)
        name = match.group(1) if match else "there"

        # Match the reply-type markers response_generator actually emits in
        # the system prompt ("REPLY TYPE: CRITICAL FEEDBACK", etc.). These are
        # uppercase with spaces, so compare against a lowercased copy — the old
        # underscore markers ("critical_feedback") never matched, collapsing
        # every offline reply to the generic acknowledgment line.
        prompt_lower = system_prompt.lower()
        if "critical feedback" in prompt_lower:
            return f"Fair point, {name}. We’re looking at this closely and want to improve the experience."
        if "substantive question" in prompt_lower:
            return f"Good question, {name}. We keep the approach focused and practical so the answer stays useful."
        return f"Appreciate the note, {name}. That’s exactly the kind of signal we want to hear."

    def cost_estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost of a request (for internal budget/spend tracking only).

        This is a rough approximation using hardcoded per-model rates that
        will drift out of date — it is NOT billing-grade accuracy and must
        never be wired into an invoice or customer-facing cost figure. For
        real billing, use OpenRouter's own usage/cost reporting.

        Approximate pricing (as of 2024, $ per 1M tokens):
        - Claude Sonnet: $3 input / $15 output
        - GPT-4 Turbo: $10 input / $30 output
        - GPT-3.5 Turbo: $0.50 input / $1.50 output
        """
        pricing = {
            "anthropic/claude-sonnet-4.6": {"input": 3, "output": 15},
            "openai/gpt-4-turbo": {"input": 10, "output": 30},
            "openai/gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }

        if model not in pricing:
            logger.warning(f"No pricing info for model {model}, cost estimate unavailable")
            return 0.0

        rates = pricing[model]
        cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
        return round(cost, 6)


# Initialize singleton
llm_client = LLMClient()
