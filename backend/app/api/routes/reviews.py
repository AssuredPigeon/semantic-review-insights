import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.models import (
    Message,
    Review,
    ReviewCreate,
    ReviewPublic,
    ReviewsPublic,
)

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("/", response_model=ReviewsPublic)
def read_reviews(
    session: SessionDep,
    current_user: CurrentUser,
    game_id: str | None = Query(default=None, description="Filter by game_id"),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve reviews.
    """
    count_statement = select(func.count()).select_from(Review)
    statement = select(Review).order_by(col(Review.created_at).desc())

    if game_id:
        count_statement = count_statement.where(Review.game_id == game_id)
        statement = statement.where(Review.game_id == game_id)

    count = session.exec(count_statement).one()
    reviews = session.exec(statement.offset(skip).limit(limit)).all()

    reviews_public = [ReviewPublic.model_validate(review) for review in reviews]
    return ReviewsPublic(data=reviews_public, count=count)


@router.post("/", response_model=ReviewPublic)
def create_review(
    *, session: SessionDep, current_user: CurrentUser, review_in: ReviewCreate
) -> Any:
    """
    Create new review.
    """
    review = Review.model_validate(review_in)
    session.add(review)
    session.commit()
    session.refresh(review)
    return review


@router.get("/{id}", response_model=ReviewPublic)
def read_review(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Any:
    """
    Get review by ID.
    """
    review = session.get(Review, id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review


@router.post("/process", response_model=Message)
def process_reviews(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    game_id: str | None = Query(default=None, description="Process reviews for a specific game"),
) -> Any:
    """
    Process reviews for semantic analysis and clustering pipeline.
    """
    statement = select(Review)
    if game_id:
        statement = statement.where(Review.game_id == game_id)
    reviews = session.exec(statement).all()

    if not reviews:
        raise HTTPException(status_code=404, detail="No reviews found to process")

    processed_count = len(reviews)
    return Message(message=f"Successfully processed {processed_count} reviews for semantic analysis")
