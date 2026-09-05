"""Project-wide configuration constants."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Local-only model, via Ollama + LiteLLM - no cloud LLM call, no API key
# (ADR-008, supersedes ADR-002's Gemini pick). "ollama_chat/", not "ollama/":
# LiteLLM's ollama_chat provider uses Ollama's OpenAI-compatible chat
# endpoint, the one that carries tool-calling correctly (verified 2026-09-05,
# references/local-model-benchmarks.md).
DEFAULT_MODEL_URI = "ollama_chat/qwen2.5-coder:14b"
