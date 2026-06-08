from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.config import settings
from app.middleware.timer import ProcessTimeMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.db import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on startup before the server begins accepting requests.
    Creates all database tables if they don't already exist.
    """
    await init_db()
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
