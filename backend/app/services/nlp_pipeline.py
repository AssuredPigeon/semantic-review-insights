# app/services/nlp_pipeline.py
#
# This module isolates ALL ML concerns from the route layer.
# The route just calls functions; it never knows about tensors or models.
#
# Design decisions:
#   - Lazy singleton: model loads once on first call, not at import/startup.
#     Avoids slowing down the entire app boot for test runs or health checks.
#   - all-MiniLM-L6-v2: 384-dim model, fast inference (~5ms/sentence on CPU),
#     multilingual-friendly, and widely used for semantic similarity tasks.
#     A larger model (e.g. all-mpnet-base-v2) would give better quality but
#     is ~5x slower — not worth it for review-length text.
#   - Validation before vectorization: rejecting bad input early keeps the
#     vector space clean. Garbage embeddings are worse than no embeddings
#     because they corrupt nearest-neighbor results silently.

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_TEXT_LENGTH = 10       # characters — filters out "ok", "bad", etc.
MAX_TEXT_LENGTH = 5000     # characters — prevents OOM on huge inputs
SUPPORTED_LANGUAGES = {"english", "spanish", "french", "german", "portuguese"}


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_model():  # type: ignore[return]
    """
    Load the sentence-transformer model exactly once per process.
    lru_cache(maxsize=1) acts as a singleton: the first call downloads/loads
    the model, subsequent calls return the cached instance immediately.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading sentence-transformer model: %s", MODEL_NAME)
        model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded successfully.")
        return model
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from e


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when a review fails ingestion validation."""


def validate_review_text(text: str, language: str | None = "english") -> None:
    """
    Validate raw review text before vectorization.

    Raises ValidationError with a human-readable message on failure.
    We raise instead of returning bool so the caller gets a reason,
    not just True/False.
    """
    stripped = text.strip()

    if len(stripped) < MIN_TEXT_LENGTH:
        raise ValidationError(
            f"Review too short: {len(stripped)} chars (minimum {MIN_TEXT_LENGTH})."
        )

    if len(stripped) > MAX_TEXT_LENGTH:
        raise ValidationError(
            f"Review too long: {len(stripped)} chars (maximum {MAX_TEXT_LENGTH})."
        )

    # Spam heuristic: if >60% of the text is a single repeated character, reject it.
    most_common_char_ratio = max(stripped.count(c) for c in set(stripped)) / len(stripped)
    if most_common_char_ratio > 0.6:
        raise ValidationError("Review appears to be spam (repetitive characters).")

    if language and language.lower() not in SUPPORTED_LANGUAGES:
        raise ValidationError(
            f"Unsupported language '{language}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
        )


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------

def vectorize(text: str) -> list[float]:
    """
    Convert review text to a 384-dimensional float vector.

    Returns a plain Python list[float] — not a numpy array — so it's
    directly JSON-serializable and compatible with pgvector's column type.

    normalize_embeddings=True ensures all vectors lie on the unit sphere,
    which makes cosine similarity equivalent to dot product — slightly faster
    at query time and numerically stable.
    """
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


# ---------------------------------------------------------------------------
# High-level pipeline entry point
# ---------------------------------------------------------------------------

def process_review(text: str, language: str | None = "english") -> list[float]:
    """
    Full ingestion pipeline for a single review:
      1. Validate the raw text.
      2. Vectorize it.
      3. Return the embedding.

    Raises ValidationError if the text fails validation.
    Raises RuntimeError if the model is not available.
    """
    validate_review_text(text, language)
    return vectorize(text)


# ---------------------------------------------------------------------------
# Semantic search helper
# ---------------------------------------------------------------------------

def build_query_vector(query: str) -> list[float]:
    """
    Vectorize a search query using the same model and normalization
    as review ingestion. This is critical: if you vectorize reviews with
    model A and queries with model B, similarity scores are meaningless.
    """
    return vectorize(query)
