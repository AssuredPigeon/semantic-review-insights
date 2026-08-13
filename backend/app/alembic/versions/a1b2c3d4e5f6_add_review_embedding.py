"""Create review table with pgvector embedding support

Revision ID: a1b2c3d4e5f6
Revises: fe56fa70289e
Create Date: 2026-08-12 00:00:00.000000

Why this migration does what it does:
- CREATE EXTENSION IF NOT EXISTS vector: pgvector must be enabled at the DB
  level before any vector column can be created. IF NOT EXISTS makes this
  idempotent — safe to run even if the extension was already installed manually.
- Creates the full 'review' table (not just the embedding column).
  The previous migration chain only covers User and Item models — Review is
  a new domain-specific model added for the NLP pipeline feature.
- VECTOR(384): dimension matches all-MiniLM-L6-v2 output exactly.
  If you change the model, you must drop and recreate this column (dimensions
  are fixed at column creation time in pgvector).
- embedding nullable: existing rows have no embedding yet. They get vectorized on
  the next /process call (backfill). NOT NULL would break the migration on
  a non-empty table.
- ivfflat index with cosine ops: enables approximate nearest-neighbor search.
  Without an index, pgvector does an exact scan (O(n)) which is fine for
  small tables but degrades at ~50k+ rows. ivfflat trades a small recall
  drop for dramatically faster queries. lists=100 is the recommended default
  for tables up to ~1M rows.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "fe56fa70289e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable the pgvector extension (idempotent).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create the review table with all columns.
    # We use raw SQL for the embedding column because SQLAlchemy's DDL
    # system doesn't know the 'vector' type natively — pgvector registers
    # it at runtime, not at DDL-generation time.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS review (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            game_id     VARCHAR(100) NOT NULL,
            review_text TEXT NOT NULL,
            author      VARCHAR(255),
            voted_up    BOOLEAN,
            playtime_hours FLOAT,
            language    VARCHAR(50) DEFAULT 'english',
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            embedding   vector(384)
        )
        """
    )

    # Index on game_id for fast per-game queries.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_review_game_id ON review (game_id)"
    )

    # IVFFlat index for approximate nearest-neighbor cosine search.
    # Only meaningful once the table has rows, but harmless on an empty table.
    op.execute(
        "CREATE INDEX IF NOT EXISTS review_embedding_idx "
        "ON review USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS review_embedding_idx")
    op.execute("DROP INDEX IF EXISTS ix_review_game_id")
    op.drop_table("review")
    # Note: we intentionally do NOT drop the vector extension on downgrade.
    # Other tables or extensions might depend on it. Dropping extensions
    # is a manual operation that requires explicit DBA intent.
