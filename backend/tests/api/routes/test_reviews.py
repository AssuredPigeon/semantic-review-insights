"""
Tests for the /reviews endpoints.

Test strategy:
- Unit-level: validate the nlp_pipeline service in isolation (no DB, no HTTP).
- Integration: test the full HTTP stack using FastAPI's TestClient.
- The sentence-transformers model is mocked in integration tests to avoid
  downloading a 90MB model in CI and to make tests deterministic.
"""
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Review
from app.services.nlp_pipeline import ValidationError, validate_review_text


# ---------------------------------------------------------------------------
# Unit tests: nlp_pipeline validation (no DB, no HTTP, no model)
# ---------------------------------------------------------------------------

class TestValidateReviewText:
    def test_valid_text_passes(self) -> None:
        # Should not raise
        validate_review_text("Great gameplay and impressive graphics!", "english")

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValidationError, match="too short"):
            validate_review_text("ok", "english")

    def test_too_long_raises(self) -> None:
        with pytest.raises(ValidationError, match="too long"):
            validate_review_text("x" * 6000, "english")

    def test_spam_text_raises(self) -> None:
        # >60% of the text is the same character
        with pytest.raises(ValidationError, match="spam"):
            validate_review_text("aaaaaaaaaaaaaaaa", "english")

    def test_unsupported_language_raises(self) -> None:
        with pytest.raises(ValidationError, match="Unsupported language"):
            validate_review_text("Un buen juego de verdad.", "klingon")

    def test_supported_languages_pass(self) -> None:
        for lang in ["english", "spanish", "french", "german", "portuguese"]:
            # Should not raise for any supported language
            validate_review_text("This is a valid review text.", lang)


# ---------------------------------------------------------------------------
# Integration tests: HTTP endpoints (model is mocked)
# ---------------------------------------------------------------------------

# A fake 384-dim embedding — all zeros except the first value.
# This is a valid unit vector for test purposes.
FAKE_EMBEDDING = [0.0] * 384
FAKE_EMBEDDING[0] = 1.0


@pytest.fixture(autouse=False)
def mock_nlp(monkeypatch: pytest.MonkeyPatch):
    """
    Patch the process_review and build_query_vector functions so tests
    don't require the sentence-transformers model to be downloaded.
    Both functions return a deterministic fake 384-dim vector.
    """
    with (
        patch("app.api.routes.reviews.process_review", return_value=FAKE_EMBEDDING),
        patch("app.api.routes.reviews.build_query_vector", return_value=FAKE_EMBEDDING),
    ):
        yield


def test_create_review(
    client: TestClient, superuser_token_headers: dict[str, str], mock_nlp: None
) -> None:
    """
    POST /reviews/ should create a record and return the review without the embedding.
    The NLP pipeline is mocked — we're testing the HTTP layer, not the model.
    """
    data = {
        "game_id": "game-123",
        "review_text": "Great gameplay and impressive graphics!",
        "author": "GamerOne",
        "voted_up": True,
        "playtime_hours": 42.5,
        "language": "english",
    }
    response = client.post(
        f"{settings.API_V1_STR}/reviews/",
        headers=superuser_token_headers,
        json=data,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["game_id"] == data["game_id"]
    assert content["review_text"] == data["review_text"]
    assert content["author"] == data["author"]
    assert content["voted_up"] is True
    assert content["playtime_hours"] == 42.5
    assert "id" in content
    # Embedding must never appear in the response
    assert "embedding" not in content


def test_create_review_invalid_text_returns_422(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    """
    POST /reviews/ with text that fails validation should return 422
    without creating any DB record. No mock needed — we want real validation.
    """
    data = {
        "game_id": "game-spam",
        "review_text": "xx",  # too short
        "language": "english",
    }
    response = client.post(
        f"{settings.API_V1_STR}/reviews/",
        headers=superuser_token_headers,
        json=data,
    )
    assert response.status_code == 422


def test_read_reviews(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """
    GET /reviews/?game_id=... should return reviews filtered by game.
    """
    review = Review(
        game_id="game-456",
        review_text="Fun mechanics but buggy.",
        author="PlayerTwo",
        voted_up=False,
    )
    db.add(review)
    db.commit()

    response = client.get(
        f"{settings.API_V1_STR}/reviews/?game_id=game-456",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["count"] >= 1
    assert any(r["game_id"] == "game-456" for r in content["data"])


def test_read_review_by_id(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    """GET /reviews/{id} should return the correct review."""
    review = Review(
        game_id="game-789",
        review_text="Masterpiece storyline.",
        author="ReviewerX",
        voted_up=True,
    )
    db.add(review)
    db.commit()
    db.refresh(review)

    response = client.get(
        f"{settings.API_V1_STR}/reviews/{review.id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["id"] == str(review.id)
    assert content["review_text"] == "Masterpiece storyline."


def test_read_review_not_found(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
    """GET /reviews/{id} with a non-existent UUID should return 404."""
    response = client.get(
        f"{settings.API_V1_STR}/reviews/{uuid.uuid4()}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Review not found"


def test_search_reviews(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session, mock_nlp: None
) -> None:
    """
    GET /reviews/search?q=... should return reviews ranked by semantic similarity.
    With the mock, all embeddings are identical so the ordering is arbitrary —
    what we're testing here is that the endpoint responds, parses the vector
    correctly, and returns results in the expected schema.
    """
    # Insert a review with a fake embedding so the search has something to find.
    review = Review(
        game_id="game-search",
        review_text="Amazing soundtrack and open world.",
        author="Tester",
        voted_up=True,
        embedding=FAKE_EMBEDDING,  # type: ignore[arg-type]
    )
    db.add(review)
    db.commit()

    response = client.get(
        f"{settings.API_V1_STR}/reviews/search?q=great+music",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert "data" in content
    assert "count" in content


def test_process_reviews(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session, mock_nlp: None
) -> None:
    """
    POST /reviews/process should vectorize un-processed reviews and return
    a summary message with counts.
    """
    review = Review(
        game_id="game-process",
        review_text="Amazing soundtrack and open world.",
        author="Tester",
        voted_up=True,
        embedding=None,  # not yet vectorized
    )
    db.add(review)
    db.commit()

    response = client.post(
        f"{settings.API_V1_STR}/reviews/process?game_id=game-process",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert "Successfully processed" in content["message"]
