# app/services/interaction_logger.py

import json
import logging
from datetime import datetime, timezone

from app.database.db import AsyncSessionLocal
from app.database.models import (
    Interaction, QueryDetail, LearnDetail,
    RetrievedChunk, EndpointType
)
from app.core.verifier import verify_response

logger = logging.getLogger(__name__)


async def log_query_interaction(
    session_id: str | None,
    original_query: str,
    processed_query: str,
    retrieved_context: list,
    generated_response: str,
    latency_ms: float | None,
    is_conversational: bool,
    citations_present: bool | None = None,
    citation_count: int | None = None,
) -> None:
    """
    Background task: verifies grounding then persists a /query interaction.

    Verification runs here — after the response is already sent to the client —
    so NLI latency never affects Pepper's response time.
    """
    # ── Phase 2: grounding verification (async, post-response) ───────────────
    grounding_score    = None
    grounding_label    = None
    has_contradiction  = None
    claims_checked     = None
    claims_supported   = None

    try:
        report = verify_response(generated_response, retrieved_context)
        grounding_score   = report.grounding_score
        grounding_label   = report.grounding_label
        has_contradiction = report.has_contradiction
        claims_checked    = report.claims_checked
        claims_supported  = report.claims_supported
        logger.info(
            f"[Logger] Verification complete: "
            f"score={grounding_score} label={grounding_label} "
            f"supported={claims_supported}/{claims_checked}"
        )
    except Exception as e:
        logger.error(f"[Logger] Verification failed (non-fatal): {e}", exc_info=True)

    # ── Persist to DB ─────────────────────────────────────────────────────────
    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():

                interaction = Interaction(
                    session_id=session_id,
                    endpoint=EndpointType.query,
                    created_at=datetime.now(timezone.utc),
                    latency_ms=latency_ms,
                    is_conversational=is_conversational,
                )
                session.add(interaction)
                await session.flush()

                session.add(QueryDetail(
                    interaction_id=interaction.id,
                    original_query=original_query,
                    processed_query=processed_query,
                    generated_response=generated_response,
                    citations_present=citations_present,
                    citation_count=citation_count,
                    grounding_score=grounding_score,
                    grounding_label=grounding_label,
                    has_contradiction=has_contradiction,
                    claims_checked=claims_checked,
                    claims_supported=claims_supported,
                ))

                for i, chunk in enumerate(retrieved_context):
                    text = chunk.get("text", "")
                    session.add(RetrievedChunk(
                        interaction_id=interaction.id,
                        chunk_index=i,
                        source=chunk.get("source"),
                        authority=chunk.get("authority"),
                        domain=chunk.get("domain"),
                        text_preview=text[:500] if text else None,
                    ))

        logger.info(
            f"[Logger] Query saved "
            f"(id={interaction.id} grounding={grounding_label} score={grounding_score})"
        )

    except Exception as e:
        logger.error(f"[Logger] Failed to save query: {e}", exc_info=True)


async def log_learn_interaction(
    session_id: str | None,
    concept: str,
    retrieved_context: list,
    lesson_data: dict,
    latency_ms: float | None,
    lesson_mode: str,
    description: str | None = None,
) -> None:
    """Background task: persists a /learn/structured or /learn/text interaction."""
    try:
        sources = ",".join(lesson_data.get("sources", []))

        if lesson_mode == "structured":
            beats      = lesson_data.get("beats", [])
            beat_types = ",".join(b.get("type", "") for b in beats)
            extra = {
                "total_beats": lesson_data.get("total_beats", len(beats)),
                "beat_types":  beat_types,
                "lesson_json": json.dumps(beats),
            }
        else:  # text
            hints = lesson_data.get("quiz_hints", [])
            extra = {
                "lesson_text": lesson_data.get("lesson_text", ""),
                "quiz_hints":  ",".join(hints) if hints else None,
                "description": description,
            }

        async with AsyncSessionLocal() as session:
            async with session.begin():

                interaction = Interaction(
                    session_id=session_id,
                    endpoint=EndpointType.learn,
                    created_at=datetime.now(timezone.utc),
                    latency_ms=latency_ms,
                    is_conversational=False,
                )
                session.add(interaction)
                await session.flush()

                session.add(LearnDetail(
                    interaction_id=interaction.id,
                    concept=concept,
                    lesson_mode=lesson_mode,
                    sources_used=sources,
                    **extra,
                ))

                for i, chunk in enumerate(retrieved_context):
                    text = chunk.get("text", "")
                    session.add(RetrievedChunk(
                        interaction_id=interaction.id,
                        chunk_index=i,
                        source=chunk.get("source"),
                        authority=chunk.get("authority"),
                        domain=chunk.get("domain"),
                        text_preview=text[:500] if text else None,
                    ))

        logger.info(
            f"[Logger] Learn ({lesson_mode}) saved "
            f"(id={interaction.id}, concept='{concept}')"
        )

    except Exception as e:
        logger.error(f"[Logger] Failed to save learn interaction: {e}", exc_info=True)