"""Add review embedding column and enable pgvector

Revision ID: a1b2c3d4e5f6
Revises: fe56fa70289e
Create Date: 2026-08-12 00:00:00.000000

Why this migration does what it does:
- CREATE EXTENSION IF NOT EXISTS vector: pgvector must be enabled at the DB
  level before any vector column can be created. IF NOT EXISTS makes this
  idempotent — safe to run even if the extension was already installed manually.
- VECTOR(384): dimension matches all-MiniLM-L6-v2 output exactly.
  If you change the model, you must drop and recreate this column (dimensions
  are fixed at column creation time in pgvector).
- nullable=True: existing rows have no embedding yet. They get vectorized on
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

    # Add the embedding column as nullable so existing rows aren't affected.
    op.add_column(
        "review",
        sa.Column(
            "embedding",
            sa.TEXT(),  # stored as text; pgvector casts automatically via the vector type
            nullable=True,
        ),
    )

    # Change column type to vector(384) using ALTER COLUMN ... USING cast.
    # We use execute() because SQLAlchemy doesn't know the 'vector' type natively —
    # pgvector registers it at runtime, not at DDL-generation time.
    op.execute(
        "ALTER TABLE review "
        "ALTER COLUMN embedding TYPE vector(384) "
        "USING embedding::vector"
    )

    # Create an IVFFlat index for approximate nearest-neighbor cosine search.
    # This index is only useful once the table has a meaningful number of rows.
    # On an empty or tiny table it has no effect but doesn't hurt either.
    op.execute(
        "CREATE INDEX IF NOT EXISTS review_embedding_idx "
        "ON review USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS review_embedding_idx")
    op.drop_column("review", "embedding")
    # Note: we intentionally do NOT drop the vector extension on downgrade.
    # Other tables or extensions might depend on it. Dropping extensions
    # is a manual operation that requires explicit DBA intent.
