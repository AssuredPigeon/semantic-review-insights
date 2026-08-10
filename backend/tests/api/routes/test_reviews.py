import uuid

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Review


def test_create_review(
    client: TestClient, superuser_token_headers: dict[str, str]
) -> None:
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


def test_read_reviews(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
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
    response = client.get(
        f"{settings.API_V1_STR}/reviews/{uuid.uuid4()}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 404
    content = response.json()
    assert content["detail"] == "Review not found"


def test_process_reviews(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    review = Review(
        game_id="game-process",
        review_text="Amazing soundtrack and open world.",
        author="Tester",
        voted_up=True,
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
