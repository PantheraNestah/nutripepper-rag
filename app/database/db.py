# app/database/db.py

import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.database.models import Base

# SQLite database file lives at the project root
# aiosqlite gives SQLAlchemy an async driver for SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./nutripepper.db")

engine = create_async_engine(
    DATABASE_URL,
    echo=False,          # Set True temporarily to log all SQL for debugging
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevents DetachedInstanceError after commit
)


async def init_db() -> None:
    """
    Create all tables on startup if they don't exist.
    Called once from main.py lifespan.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """
    FastAPI dependency that yields a database session per request.
    Not used for background logging (which manages its own session),
    but available for any future read endpoints.
    """
    async with AsyncSessionLocal() as session:
        yield session