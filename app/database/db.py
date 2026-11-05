# app/database/db.py

import os
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.database.models import Base

# SQLite database file lives at the project root
# aiosqlite gives SQLAlchemy an async driver for SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./nutripepper.db")

_IS_SQLITE = DATABASE_URL.startswith("sqlite")

# SQLite allows one writer at a time. Concurrent background log writes must wait for each
# other rather than fail: wait up to 30 s for the lock (the driver default is 5 s).
engine = create_async_engine(
    DATABASE_URL,
    echo=False,          # Set True temporarily to log all SQL for debugging
    future=True,
    connect_args={"timeout": 30} if _IS_SQLITE else {},
)

if _IS_SQLITE:
    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        # WAL: readers never block the writer and vice versa. NORMAL sync is safe under WAL.
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

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