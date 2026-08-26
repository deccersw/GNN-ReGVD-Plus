"""
Token accounting against the *real* GraphCodeBERT tokenizer (decision D-002).

`HybridPredictor.predict_single` silently truncates at ``block_size - 2``
BPE tokens, so a budget measured in characters would be off by 1.5-2x on C
code and we would either waste the window or overflow it without noticing.
"""

import hashlib
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TokenCounter:
    """Counts BPE tokens, caching by content hash.

    Falls back to a whitespace estimate when transformers is unavailable, so
    the module still runs standalone -- loudly, because the numbers change.
    """

    def __init__(self, tokenizer_name: str = "microsoft/graphcodebert-base",
                 use_real: bool = True):
        self.tokenizer_name = tokenizer_name
        self.use_real = use_real
        self._tokenizer = None
        self._loaded = False
        self._cache: Dict[str, int] = {}
        self.exact = False

    # ------------------------------------------------------------------

    def _load(self):
        if self._loaded:
            return self._tokenizer
        self._loaded = True
        if not self.use_real:
            logger.info("[Budget] real tokenizer disabled, using estimate")
            return None
        try:
            from transformers import RobertaTokenizer
            self._tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_name)
            self.exact = True
            logger.info("[Budget] tokenizer loaded: %s", self.tokenizer_name)
        except Exception as exc:
            logger.warning("[Budget] tokenizer unavailable (%s); token counts "
                           "are ESTIMATES and budgets are approximate", exc)
            self._tokenizer = None
        return self._tokenizer

    def count(self, text: str, key: Optional[str] = None) -> int:
        if not text:
            return 0
        cache_key = key or hashlib.sha1(
            text.encode("utf-8", "replace")).hexdigest()[:16]
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit

        tok = self._load()
        if tok is not None:
            # mirror exactly what predict_single does before tokenising
            normalised = " ".join(text.split())
            value = len(tok.tokenize(normalised))
        else:
            value = estimate_tokens(text)

        self._cache[cache_key] = value
        return value


def estimate_tokens(text: str) -> int:
    """Rough BPE estimate: whitespace tokens inflated for punctuation splits."""
    words = text.split()
    if not words:
        return 0
    punct = sum(1 for ch in text if ch in "(){}[];,*&->=+/%<>!|^~?:.")
    return int(len(words) * 1.35 + punct * 0.45)


class Budget:
    """Mutable remaining-token counter for one root expansion."""

    def __init__(self, limit: int):
        self.limit = limit
        self.spent = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    def can_afford(self, cost: int) -> bool:
        return self.spent + cost <= self.limit

    def spend(self, cost: int) -> None:
        self.spent += cost

    def __repr__(self) -> str:                            # pragma: no cover
        return f"Budget(spent={self.spent}/{self.limit})"
