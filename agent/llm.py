import backoff
import os
from typing import Tuple
import requests
import litellm
from dotenv import load_dotenv
import json

load_dotenv()

# OAuth mode: use Claude Code Max subscription via ccproxy instead of API key
# Set ANTHROPIC_AUTH_MODE=oauth in .env and run: ccproxy auth login claude_api
if os.getenv("ANTHROPIC_AUTH_MODE") == "oauth":
    from utils.ccproxy_manager import maybe_start_ccproxy
    maybe_start_ccproxy()

MAX_TOKENS = 16384

CLAUDE_MODEL = "anthropic/claude-sonnet-4-5-20250929"
CLAUDE_HAIKU_MODEL = "anthropic/claude-3-haiku-20240307"
CLAUDE_35NEW_MODEL = "anthropic/claude-3-5-sonnet-20241022"
OPENAI_MODEL = "openai/gpt-4o"
OPENAI_MINI_MODEL = "openai/gpt-4o-mini"
OPENAI_O3_MODEL = "openai/o3"
OPENAI_O3MINI_MODEL = "openai/o3-mini"
OPENAI_O4MINI_MODEL = "openai/o4-mini"
OPENAI_GPT52_MODEL = "openai/gpt-5.2"
OPENAI_GPT5_MODEL = "openai/gpt-5"
OPENAI_GPT5MINI_MODEL = "openai/gpt-5-mini"
GEMINI_3_MODEL = "gemini/gemini-3-pro-preview"
GEMINI_MODEL = "gemini/gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini/gemini-2.5-flash"
# Ollama local model — configurable via OLLAMA_MODEL env var.
# Default is Qwen3.5 9B q4_K_M: best quality/size tradeoff for Apple M4.
# Override OLLAMA_API_BASE (default http://localhost:11434) to point at a remote Ollama host.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama_chat/qwen3.5:9b-q4_K_M")

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------
# Set LLM_PROVIDER in .env to choose the default model when multiple providers
# are configured. If unset, falls back to oauth → Claude, otherwise OpenAI.
#
#   LLM_PROVIDER=ollama     → OLLAMA_MODEL  (local Ollama instance)
#   LLM_PROVIDER=oauth      → CLAUDE_MODEL  (Claude via ccproxy OAuth)
#   LLM_PROVIDER=anthropic  → CLAUDE_MODEL  (Claude via direct API key)
#   LLM_PROVIDER=openai     → OPENAI_MODEL
#   LLM_PROVIDER=gemini     → GEMINI_MODEL
#
_PROVIDER_MAP = {
    "ollama": lambda: OLLAMA_MODEL,
    "oauth": lambda: CLAUDE_MODEL,
    "anthropic": lambda: CLAUDE_MODEL,
    "openai": lambda: OPENAI_MODEL,
    "gemini": lambda: GEMINI_MODEL,
}
_provider = os.getenv("LLM_PROVIDER", "").lower()
if _provider in _PROVIDER_MAP:
    DEFAULT_MODEL = _PROVIDER_MAP[_provider]()
elif os.getenv("ANTHROPIC_AUTH_MODE") == "oauth":
    DEFAULT_MODEL = CLAUDE_MODEL
else:
    DEFAULT_MODEL = OPENAI_MODEL

litellm.drop_params=True

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history=None,
) -> Tuple[str, list, dict]:
    if msg_history is None:
        msg_history = []

    # Convert text to content, compatible with LITELLM API
    msg_history = [
        {**msg, "content": msg.pop("text")} if "text" in msg else msg
        for msg in msg_history
    ]

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    # Build kwargs - handle model-specific requirements
    completion_kwargs = {
        "model": model,
        "messages": new_msg_history,
    }

    # GPT-5 and GPT-5-mini only support default temperature (1), skip it
    # GPT-5.2 supports temperature
    if model in ["openai/gpt-5", "openai/gpt-5-mini"]:
        pass  # Don't set temperature
    else:
        completion_kwargs["temperature"] = temperature

    # GPT-5 models require max_completion_tokens instead of max_tokens
    if "gpt-5" in model:
        completion_kwargs["max_completion_tokens"] = max_tokens
    else:
        # Claude Haiku has a 4096 token limit
        if "claude-3-haiku" in model:
            completion_kwargs["max_tokens"] = min(max_tokens, 4096)
        else:
            completion_kwargs["max_tokens"] = max_tokens

    response = litellm.completion(**completion_kwargs)
    response_text = response['choices'][0]['message']['content']  # pyright: ignore
    new_msg_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

    # Convert content to text, compatible with MetaGen API
    new_msg_history = [
        {**msg, "text": msg.pop("content")} if "content" in msg else msg
        for msg in new_msg_history
    ]

    return response_text, new_msg_history, {}


if __name__ == "__main__":
    msg = 'Hello there!'
    models = [
        ("CLAUDE_MODEL", CLAUDE_MODEL),
        ("CLAUDE_HAIKU_MODEL", CLAUDE_HAIKU_MODEL),
        ("CLAUDE_35NEW_MODEL", CLAUDE_35NEW_MODEL),
        ("OPENAI_MODEL", OPENAI_MODEL),
        ("OPENAI_MINI_MODEL", OPENAI_MINI_MODEL),
        ("OPENAI_O3_MODEL", OPENAI_O3_MODEL),
        ("OPENAI_O3MINI_MODEL", OPENAI_O3MINI_MODEL),
        ("OPENAI_O4MINI_MODEL", OPENAI_O4MINI_MODEL),
        ("OPENAI_GPT52_MODEL", OPENAI_GPT52_MODEL),
        ("OPENAI_GPT5_MODEL", OPENAI_GPT5_MODEL),
        ("OPENAI_GPT5MINI_MODEL", OPENAI_GPT5MINI_MODEL),
        ("GEMINI_3_MODEL", GEMINI_3_MODEL),
        ("GEMINI_MODEL", GEMINI_MODEL),
        ("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL),
        ("OLLAMA_MODEL", OLLAMA_MODEL),
    ]
    for name, model in models:
        print(f"\n{'='*50}")
        print(f"Testing {name}: {model}")
        print('='*50)
        try:
            output_msg, msg_history, info = get_response_from_llm(msg, model=model)
            print(f"OK: {output_msg[:100]}...")
        except Exception as e:
            print(f"FAIL: {str(e)[:200]}")
