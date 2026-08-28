"""Project-wide configuration constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Vertex AI auth (no API keys - relies on Application Default Credentials).
# ADK reads GOOGLE_GENAI_USE_VERTEXAI/GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION
# from the environment itself; these mirrors are for our own modules.
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

# Verified 2026-08-26 with a direct generate_content call against
# gd-gcp-internship-ds's live Vertex AI catalogue.
GEMINI_MODEL = "gemini-3.7-flash"
