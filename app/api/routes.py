# app/api/routes.py

import time
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from app.schemas.query import QueryRequest, LearnRequest
from app.schemas.response import QueryResponse
from app.dependencies.pipeline import get_pipeline
from app.services.interaction_logger import log_query_interaction, log_learn_interaction

router = APIRouter()


def _get_session_id(request: Request) -> str | None:
    """
    Extract session_id from request headers if the client sends one.
    The Android tablet app should send X-Session-ID on every request
    to group multi-turn conversations. Falls back to None gracefully.
    """
    return request.headers.get("X-Session-ID")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: Request,
    body: QueryRequest,
    background_tasks: BackgroundTasks,
    pipeline=Depends(get_pipeline),
):
    start = time.monotonic()
    result = await pipeline.run_query(body.user_query)
    latency_ms = (time.monotonic() - start) * 1000

    background_tasks.add_task(
        log_query_interaction,
        session_id=_get_session_id(request),
        original_query=result["original_query"],
        processed_query=result["processed_query"],
        retrieved_context=result["retrieved_context"],
        generated_response=result["generated_response"],
        latency_ms=latency_ms,
        is_conversational=len(result["retrieved_context"]) == 0,
    )

    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": result,
    }


@router.post("/learn")
async def learn(
    request: Request,
    body: LearnRequest,
    background_tasks: BackgroundTasks,
    pipeline=Depends(get_pipeline),
):
    start = time.monotonic()
    result = await pipeline.run_lesson(body.concept)
    latency_ms = (time.monotonic() - start) * 1000

    background_tasks.add_task(
        log_learn_interaction,
        session_id=_get_session_id(request),
        concept=body.concept,
        retrieved_context=result.get("retrieved_context", []),
        lesson_data=result,
        latency_ms=latency_ms,
    )

    return {
        "status": "success",
        "message": "Lesson generated successfully",
        "data": result,
    }