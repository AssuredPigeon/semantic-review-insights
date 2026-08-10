from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, delete
from sqlmodel.pool import StaticPool

from app.api.deps import get_db
from app.core.config import settings
from app.core.db import engine as postgres_engine, init_db
from app.main import app
from app.models import Item, Review, User
from tests.utils.user import authentication_token_from_email
from tests.utils.utils import get_superuser_token_headers

import socket

# SQLite in-memory engine for local test environment
sqlite_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


def is_postgres_available() -> bool:
    try:
        sock = socket.create_connection(("localhost", 5432), timeout=0.5)
        sock.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def db() -> Generator[Session, None, None]:
    if is_postgres_available():
        active_engine = postgres_engine
    else:
        active_engine = sqlite_engine
        SQLModel.metadata.create_all(active_engine)

    def get_test_db() -> Generator[Session, None, None]:
        with Session(active_engine) as session:
            yield session

    app.dependency_overrides[get_db] = get_test_db

    with Session(active_engine) as session:
        init_db(session)
        yield session
        session.execute(delete(Item))
        session.execute(delete(Review))
        session.execute(delete(User))
        session.commit()

    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def superuser_token_headers(client: TestClient) -> dict[str, str]:
    return get_superuser_token_headers(client)


@pytest.fixture(scope="module")
def normal_user_token_headers(client: TestClient, db: Session) -> dict[str, str]:
    return authentication_token_from_email(
        client=client, email=settings.EMAIL_TEST_USER, db=db
    )
