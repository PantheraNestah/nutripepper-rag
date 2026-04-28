from fastapi import FastAPI
from app.api.routes import router
from app.core.config import settings
from app.middleware.timer import ProcessTimeMiddleware

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)

app.add_middleware(ProcessTimeMiddleware)
app.include_router(router)