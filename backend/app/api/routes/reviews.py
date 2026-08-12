import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.models import (
    Message,
    Review,
    ReviewCreate,
    ReviewPublic,
    ReviewsPublic,
)
from app.services.nlp_pipeline import (
    ValidationError,
    build_query_vector,
    process_review,
)

router = APIRouter(prefix="/reviews", tags=["reviews"])


# ---------------------------------------------------------------------------
# GET /reviews/ — list with optional game_id filter
# ---------------------------------------------------------------------------

@router.get("/", response_model=ReviewsPublic)
def read_reviews(
    session: SessionDep,
    current_user: CurrentUser,
    game_id: str | None = Query(default=None, description="Filter by game_id"),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve reviews with optional filtering by game_id.
    Returns paginated results ordered by most recent first.
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


# ---------------------------------------------------------------------------
# POST /reviews/ — create and immediately vectorize
# ---------------------------------------------------------------------------

@router.post("/", response_model=ReviewPublic)
def create_review(
    *, session: SessionDep, current_user: CurrentUser, review_in: ReviewCreate
) -> Any:
    """
    Create a new review and run the NLP pipeline on it immediately.

    The embedding is stored in the DB but never returned in the response
    (excluded from ReviewPublic) — it's an internal implementation detail.
    If the review text fails validation, a 422 is returned before any DB write.
    """
    # Run validation + vectorization before touching the DB.
    # This way, if the text is rejected, we never create a partial record.
    try:
        embedding = process_review(
            text=review_in.review_text,
            language=review_in.language,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    review = Review.model_validate(review_in, update={"embedding": embedding})
    session.add(review)
    session.commit()
    session.refresh(review)
    return review


# ---------------------------------------------------------------------------
# GET /reviews/search — semantic similarity search
# NOTE: this route MUST be defined before /{id} so FastAPI doesn't treat
#       "search" as a UUID and route it to read_review() instead.
# ---------------------------------------------------------------------------

@router.get("/search", response_model=ReviewsPublic)
def search_reviews(
    session: SessionDep,
    current_user: CurrentUser,
    q: str = Query(..., description="Natural language query to search reviews by meaning"),
    limit: int = Query(default=10, ge=1, le=50),
    game_id: str | None = Query(default=None, description="Optional: restrict search to a game"),
) -> Any:
    """
    Semantic similarity search over vectorized reviews.

    Uses pgvector cosine distance (<=> operator) to rank reviews by how
    semantically close they are to the query, regardless of exact wording.

    Example: "good story" will surface reviews mentioning "excellent narrative",
    "compelling plot", "great writing" — not just reviews containing the words
    "good story".

    Why cosine distance instead of L2?
    Because our embeddings are L2-normalized (unit sphere), cosine similarity
    and dot product are equivalent. Cosine is preferred here because it's
    scale-invariant — a short review and a long review about the same topic
    should score similarly.
    """
    if not q.strip():
        raise HTTPException(status_code=422, detail="Search query cannot be empty.")

    query_vector = build_query_vector(q)
    # Format vector for pgvector: '[0.1, 0.2, ...]'
    vector_literal = "[" + ",".join(str(v) for v in query_vector) + "]"

    # Build the raw SQL query using pgvector's <=> cosine distance operator.
    # We use text() here because SQLModel/SQLAlchemy don't have a built-in
    # operator for <=> yet — pgvector's Python package adds it to the Column
    # type but not to the ORM expression layer for SQLModel.
    if game_id:
        sql = text(
            "SELECT *, (embedding <=> :vec) AS distance "
            "FROM review "
            "WHERE game_id = :game_id AND embedding IS NOT NULL "
            "ORDER BY distance ASC "
            "LIMIT :limit"
        )
        rows = session.exec(sql, params={"vec": vector_literal, "game_id": game_id, "limit": limit}).all()  # type: ignore
    else:
        sql = text(
            "SELECT *, (embedding <=> :vec) AS distance "
            "FROM review "
            "WHERE embedding IS NOT NULL "
            "ORDER BY distance ASC "
            "LIMIT :limit"
        )
        rows = session.exec(sql, params={"vec": vector_literal, "limit": limit}).all()  # type: ignore

    # Map raw SQL rows back to ReviewPublic schema.
    reviews_public = []
    for row in rows:
        reviews_public.append(
            ReviewPublic(
                id=row.id,
                game_id=row.game_id,
                review_text=row.review_text,
                author=row.author,
                voted_up=row.voted_up,
                playtime_hours=row.playtime_hours,
                language=row.language,
                created_at=row.created_at,
            )
        )

    return ReviewsPublic(data=reviews_public, count=len(reviews_public))


# ---------------------------------------------------------------------------
# GET /reviews/{id} — get single review by UUID
# ---------------------------------------------------------------------------

@router.get("/{id}", response_model=ReviewPublic)
def read_review(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Any:
    """
    Get a single review by its UUID.
    """
    review = session.get(Review, id)
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    return review


# ---------------------------------------------------------------------------
# POST /reviews/process — batch vectorize existing un-vectorized reviews
# ---------------------------------------------------------------------------

@router.post("/process", response_model=Message)
def process_reviews(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    game_id: str | None = Query(default=None, description="Process reviews for a specific game"),
) -> Any:
    """
    Batch NLP pipeline: find reviews with no embedding yet and vectorize them.

    This is useful for:
    - Reviews created before the pipeline was in place (backfill).
    - Retrying failed vectorizations.
    - Forcing re-vectorization after a model upgrade.

    Only superusers can trigger this endpoint since it can be CPU-intensive.
    """
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions.")

    statement = select(Review).where(Review.embedding.is_(None))  # type: ignore[union-attr]
    if game_id:
        statement = statement.where(Review.game_id == game_id)

    reviews = session.exec(statement).all()

    if not reviews:
        raise HTTPException(status_code=404, detail="No un-vectorized reviews found.")

    processed_count = 0
    skipped_count = 0
    for review in reviews:
        try:
            embedding = process_review(
                text=review.review_text,
                language=review.language,
            )
            review.embedding = embedding  # type: ignore[assignment]
            session.add(review)
            processed_count += 1
        except (ValidationError, RuntimeError):
            # Don't fail the whole batch if one review is invalid.
            skipped_count += 1

    session.commit()

    return Message(
        message=(
            f"Successfully processed {processed_count} reviews. "
            f"Skipped {skipped_count} (failed validation)."
        )
    )
