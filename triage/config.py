"""
Configuration for the Triage Head.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TriageConfig:
    """Configuration for LLM-based triage verification."""

    enabled: bool = False
    model: str = "openrouter/anthropic/claude-sonnet-4-6"
    temperature: float = 0.1
    max_tokens: int = 2048
    api_key: Optional[str] = None  # fallback: env OPENROUTER_API_KEY
