import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.config import settings
from app.middleware.timer import ProcessTimeMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.db import init_db
from app.services.warmup import preload_enabled, warm_up

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup before the server begins accepting requests.
    Creates all database tables if they don't already exist, then loads the retrieval
    and verification models so the first request is fast (disable with PRELOAD_MODELS=false).
    """
    await init_db()
    if preload_enabled():
        await asyncio.to_thread(warm_up)
    yield
    # Any shutdown cleanup goes here (after yield)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ProcessTimeMiddleware)

#app.include_router(router, prefix="/api")
app.include_router(router)
