"""
llm_client.py
=============
Single responsibility: send a (system_prompt, user_prompt) pair to an LLM
via OpenRouter and return the answer text. Isolated so the UI layer and
the pipeline layer don't need to know anything about HTTP/auth details.
"""

from __future__ import annotations
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# A handful of commonly used OpenRouter model ids. Any valid OpenRouter
# model string can be typed in manually too — this list is just a
# convenience shortlist, not an exhaustive/validated set.
COMMON_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-70b-instruct",
    "mistralai/mistral-7b-instruct",
]


class LLMClientError(Exception):
    pass


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 800,
    timeout: int = 60,
) -> str:
    """
    Call OpenRouter's chat completions endpoint and return the assistant's
    reply text. Raises LLMClientError with a readable message on failure.
    """
    if not api_key:
        raise LLMClientError("No OpenRouter API key provided.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for attribution/rate-limit purposes.
        "HTTP-Referer": "https://localhost",
        "X-Title": "RAG Retrieval Explorer",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise LLMClientError(f"Network error calling OpenRouter: {e}") from e

    if response.status_code != 200:
        raise LLMClientError(
            f"OpenRouter returned HTTP {response.status_code}: {response.text[:500]}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMClientError(f"Unexpected OpenRouter response shape: {data}") from e
