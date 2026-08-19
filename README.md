<div align="center">

# 🔍 Semantic Review Insights

**Production-ready NLP API that transforms raw user reviews into semantic intelligence.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat&logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

> [!NOTE]

Semantic Review Insights is a REST API that ingests raw, unstructured user reviews and exposes them as queryable semantic data. Instead of keyword matching, it uses **sentence embeddings + pgvector** to find reviews by *meaning* — enabling similarity search, clustering, and analytics on top of natural language.

Built solo end-to-end: NLP pipeline, API layer, authentication, pagination, and testing.

---

## Architecture

```
Raw Text Input
      │
      ▼
┌─────────────────────────────────────────────┐
│              Ingestion Pipeline             │
│  1. Validation (length, language, spam)     │
│  2. Vectorization (sentence-transformers)   │
│  3. Storage (pgvector cosine index)         │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│               FastAPI Layer                 │
│  • JWT Auth (OAuth2 Bearer)                 │
│  • Pydantic v2 validation                   │
│  • Cursor-based pagination                  │
│  • Auto OpenAPI docs                        │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│              PostgreSQL + pgvector          │
│  • Vector similarity search (cosine)        │
│  • Alembic-managed migrations               │
│  • SQLModel ORM                             │
└─────────────────────────────────────────────┘
```

---

## Key Features

### 🧠 NLP Ingestion Pipeline
Raw review text is validated (length, language detection, spam heuristics), vectorized with **sentence-transformers**, and stored in **pgvector** — translating unstructured language into high-dimensional semantic vectors.

### 🔎 Semantic Search
Nearest-neighbor retrieval over `pgvector` cosine similarity index. Queries return reviews ranked by *semantic closeness*, not keyword overlap — capturing intent and meaning.

### 🔒 JWT Authentication
OAuth2 Bearer token flow with role-based access. Pydantic v2 enforces strict type validation and serialization across all request/response schemas.

### 📄 Cursor-Based Pagination
Stateless, cache-friendly pagination using opaque cursors. Handles large review collections efficiently without memory pressure or offset instability.

### 🧪 Comprehensive Test Suite
pytest + HTTPX async client covering:
- Unit tests for each pipeline stage (validation, vectorization, storage)
- Integration tests for full auth flows
- Parametrized edge-case coverage for boundary inputs

### 🗄️ Alembic Migrations
Schema changes are version-controlled, reversible, and applied deterministically — no manual SQL.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| ORM | SQLModel |
| Database | PostgreSQL + pgvector |
| Embeddings | sentence-transformers |
| Auth | JWT (OAuth2 Bearer) |
| Validation | Pydantic v2 |
| Migrations | Alembic |
| Testing | pytest + HTTPX |
| Containerization | Docker |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL with pgvector extension

### Installation

```bash
git clone https://github.com/AssuredPigeon/semantic-review-insights.git
cd semantic-review-insights

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env with your database credentials and JWT secret
```

Key variables:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/reviews_db
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Run with Docker

```bash
docker compose up --build
```

### Run Locally

```bash
# Apply migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload
```

### Run Tests

```bash
pytest -v
```

---

## API Reference

Once running, the full interactive docs are available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Core Endpoints

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `POST` | `/auth/token` | Obtain JWT token | No |
| `POST` | `/reviews/` | Ingest a new review | Yes |
| `GET` | `/reviews/` | List reviews (paginated) | Yes |
| `GET` | `/reviews/search` | Semantic similarity search | Yes |
| `GET` | `/reviews/{id}` | Get a single review | Yes |
| `DELETE` | `/reviews/{id}` | Delete a review | Yes (admin) |

### Example: Semantic Search

```bash
curl -X GET "http://localhost:8000/reviews/search?q=great+battery+life&limit=5" \
  -H "Authorization: Bearer <token>"
```

Response:
```json
{
  "results": [
    {
      "id": "abc123",
      "text": "The battery easily lasts two full days.",
      "similarity": 0.91,
      "created_at": "2026-07-15T10:30:00Z"
    }
  ],
  "next_cursor": "eyJpZCI6..."
}
```

---

## Project Structure

```
semantic-review-insights/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── core/
│   │   ├── config.py        # Settings (Pydantic BaseSettings)
│   │   └── security.py      # JWT creation/verification
│   ├── api/
│   │   └── routes/          # Route modules (auth, reviews, search)
│   ├── models/              # SQLModel table definitions
│   ├── schemas/             # Pydantic request/response schemas
│   ├── services/
│   │   ├── pipeline.py      # NLP ingestion pipeline
│   │   └── search.py        # pgvector similarity queries
│   └── db/
│       └── session.py       # Database session management
├── alembic/                 # Migration files
├── tests/                   # pytest test suite
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Design Decisions

**Why pgvector over a dedicated vector DB?**
Keeping vectors in PostgreSQL eliminates an extra service, simplifies transactions, and lets relational and vector queries run in the same database — sufficient at this scale with no operational overhead.

**Why cursor-based pagination?**
Offset pagination breaks under concurrent inserts (rows shift). Cursors are stable, stateless, and play well with caching layers.

**Why sentence-transformers over OpenAI embeddings?**
Local inference eliminates API costs, removes the external dependency, and keeps data on-premise — important for user-generated content.

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
Built by <a href="https://github.com/AssuredPigeon">Daniel Tornero</a>
</div>
