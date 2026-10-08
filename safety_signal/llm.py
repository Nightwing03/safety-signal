"""Builds the real model client from the sibling llm-service-template project (provider fallback, retries,
rate limiting). Kept out of the other modules so everything else is testable with a fake model."""
import os
import sys
from pathlib import Path

_ALLOWED_KEYS = ("GROQ_API_KEY", "GROQ_MODEL", "OLLAMA_MODEL", "OLLAMA_BASE_URL")


def _load_env(path: str) -> None:
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key in _ALLOWED_KEYS:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def build_llm(dotenv_path: str = ".env"):
    _load_env(dotenv_path)
    service = Path(os.environ.get("LLM_SERVICE_PATH", Path(__file__).resolve().parents[2] / "llm-service-template"))
    if str(service) not in sys.path:
        sys.path.insert(0, str(service))
    from llm_service.client import LLMClient
    from llm_service.providers.groq import GroqProvider
    from llm_service.providers.ollama import OllamaProvider

    return LLMClient(
        [
            GroqProvider(model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")),
            OllamaProvider(model=os.environ.get("OLLAMA_MODEL", "qwen2.5")),
        ],
        requests_per_minute={"groq": 25},
    )
