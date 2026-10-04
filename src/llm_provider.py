"""Shared LLM client and rate limiter for generation, enrichment, and RAGAS."""

from functools import lru_cache

from config import GOOGLE_API_KEY, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


if GOOGLE_API_KEY:
    from langchain_core.rate_limiters import InMemoryRateLimiter

    # Google free tier currently permits 15 requests/minute for the selected
    # model. Keep a safety margin so phase transitions do not produce bursts.
    LLM_RATE_LIMITER = InMemoryRateLimiter(
        requests_per_second=0.2,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
else:
    LLM_RATE_LIMITER = None


@lru_cache(maxsize=1)
def get_llm_client():
    from openai import OpenAI

    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def create_chat_completion(*, messages: list[dict], max_tokens: int | None = None,
                           response_format: dict | None = None):
    """Create one rate-limited chat completion using the configured provider."""
    if LLM_RATE_LIMITER is not None:
        LLM_RATE_LIMITER.acquire()
    options = {"model": LLM_MODEL, "messages": messages}
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if response_format is not None:
        options["response_format"] = response_format
    return get_llm_client().chat.completions.create(**options)
