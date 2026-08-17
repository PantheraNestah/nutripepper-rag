# app/api/routes.py

import time
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from app.schemas.query import QueryRequest, LearnRequest, TextLearnRequest
from app.schemas.response import QueryResponse
from app.dependencies.pipeline import get_pipeline
from app.services.interaction_logger import log_query_interaction, log_learn_interaction

router = APIRouter()


def _get_session_id(request: Request) -> str | None:
    return request.headers.get("X-Session-ID")


# ── /query ────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(
    request: Request,
    body: QueryRequest,
    background_tasks: BackgroundTasks,
    pipeline=Depends(get_pipeline),
):
    start  = time.monotonic()
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
        citations_present=result.get("citations_present"),
        citation_count=result.get("citation_count"),
    )

    # Return clean response — strip internal evaluation fields
    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": {
            "original_query":    result["original_query"],
            "processed_query":   result["processed_query"],
            "retrieved_context": result["retrieved_context"],
            "generated_response": result["generated_response"],
            "citations_present": result.get("citations_present"),
        },
    }


# ── /learn/structured ─────────────────────────────────────────────────────────

@router.post("/learn/structured")
async def learn_structured(
    request: Request,
    body: LearnRequest,
    background_tasks: BackgroundTasks,
    pipeline=Depends(get_pipeline),
):
    start  = time.monotonic()
    result = await pipeline.run_structured_lesson(body.concept)
    latency_ms = (time.monotonic() - start) * 1000

    background_tasks.add_task(
        log_learn_interaction,
        session_id=_get_session_id(request),
        concept=body.concept,
        retrieved_context=result.pop("retrieved_context", []),
        lesson_data=result,
        latency_ms=latency_ms,
        lesson_mode="structured",
    )

    return {"status": "success", "message": "Structured lesson generated", "data": result}


# ── /learn/text ───────────────────────────────────────────────────────────────

@router.post("/learn/text")
async def learn_text(
    request: Request,
    body: TextLearnRequest,
    background_tasks: BackgroundTasks,
    pipeline=Depends(get_pipeline),
):
    start  = time.monotonic()
    result = await pipeline.run_text_lesson(body.concept, body.description)
    latency_ms = (time.monotonic() - start) * 1000

    background_tasks.add_task(
        log_learn_interaction,
        session_id=_get_session_id(request),
        concept=body.concept if not body.description else f"{body.concept} | {body.description}",
        retrieved_context=result.pop("retrieved_context", []),
        lesson_data=result,
        latency_ms=latency_ms,
        lesson_mode="text",
        description=body.description,
    )

    return {"status": "success", "message": "Text lesson generated", "data": result}