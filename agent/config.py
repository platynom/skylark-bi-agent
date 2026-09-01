"""Central configuration. All secrets come from Streamlit secrets or env vars."""
from __future__ import annotations
import os

try:
    import streamlit as st
    _SECRETS = dict(st.secrets) if hasattr(st, "secrets") else {}
except Exception:  # running outside Streamlit (tests, CLI)
    _SECRETS = {}


def get(key: str, default=None):
    """Secret lookup order: Streamlit secrets -> environment -> default."""
    if key in _SECRETS and _SECRETS[key] not in ("", None):
        return _SECRETS[key]
    val = os.environ.get(key)
    return val if val not in ("", None) else default


MONDAY_API_TOKEN = get("MONDAY_API_TOKEN")
MONDAY_API_URL = get("MONDAY_API_URL", "https://api.monday.com/v2")
MONDAY_API_VERSION = get("MONDAY_API_VERSION", "2024-10")

# Board resolution: prefer explicit IDs, fall back to name matching.
DEALS_BOARD_ID = get("DEALS_BOARD_ID")
WORK_ORDERS_BOARD_ID = get("WORK_ORDERS_BOARD_ID")
DEALS_BOARD_NAME = get("DEALS_BOARD_NAME", "Deals")
WORK_ORDERS_BOARD_NAME = get("WORK_ORDERS_BOARD_NAME", "Work Orders")

GEMINI_API_KEY = get("GEMINI_API_KEY")
GEMINI_MODEL = get("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Provider redundancy comes from Vertex -> AI Studio, not from silently changing
# models inside AI Studio. Both quota pools deliberately serve the same model.
GEMINI_FALLBACK_MODELS: list[str] = []

# Vertex AI provider. Keep these in the central secret lookup so local
# Streamlit secrets and deployment environment variables behave identically.
GCP_SERVICE_ACCOUNT_B64 = get("GCP_SERVICE_ACCOUNT_B64")
GCP_PROJECT = get("GCP_PROJECT", "project-4f0f85f8-1fbe-4abe-b7e")
GCP_PROJECT_NUMBER = get("GCP_PROJECT_NUMBER")
GCP_SERVICE_ACCOUNT_EMAIL = get("GCP_SERVICE_ACCOUNT_EMAIL")
GCP_WORKLOAD_IDENTITY_POOL_ID = get("GCP_WORKLOAD_IDENTITY_POOL_ID")
GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID = get("GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID")
VERTEX_REGION = get("VERTEX_REGION", "global")
VERTEX_MODEL = get("VERTEX_MODEL", "gemini-3.5-flash-lite")

CACHE_TTL_SECONDS = int(get("CACHE_TTL_SECONDS", 300))
MAX_SQL_RETRIES = int(get("MAX_SQL_RETRIES", 2))
MAX_ROWS_TO_LLM = int(get("MAX_ROWS_TO_LLM", 60))

# Indian fiscal year starts in April.
FISCAL_YEAR_START_MONTH = 4
