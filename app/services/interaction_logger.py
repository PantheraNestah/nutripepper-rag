# app/services/interaction_logger.py

import logging
from datetime import datetime, timezone

from app.database.db import AsyncSessionLocal
from app.database.models import (
    Interaction, QueryDetail, LearnDetail,
    RetrievedChunk, EndpointType
)

logger = logging.getLogger(__name__)


async def log_query_interaction(
    session_id: str | None,
    original_query: str,
    processed_query: str,
    retrieved_context: list,
    generated_response: str,
    latency_ms: float | None,
    is_conversational: bool,
) -> None:
    """
    Background task: persists a /query interaction to SQLite.
    Runs after the response has already been sent to the client.
    Any exception here is caught and logged — it must never crash the app.
    """
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
                await session.flush()  # Populates interaction.id before child inserts

                query_detail = QueryDetail(
                    interaction_id=interaction.id,
                    original_query=original_query,
                    processed_query=processed_query,
                    generated_response=generated_response,
                )
                session.add(query_detail)

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

        logger.info(f"[Logger] Query interaction saved (id={interaction.id})")

    except Exception as e:
        # Log but never re-raise — background tasks must not affect the response
        logger.error(f"[Logger] Failed to save query interaction: {e}", exc_info=True)


async def log_learn_interaction(
    session_id: str | None,
    concept: str,
    retrieved_context: list,
    lesson_data: dict,
    latency_ms: float | None,
) -> None:
    """
    Background task: persists a /learn interaction to SQLite.
    Runs after the response has already been sent to the client.
    """
    try:
        beats = lesson_data.get("beats", [])
        beat_types = ",".join(b.get("type", "") for b in beats)
        sources_used = ",".join(lesson_data.get("sources", []))

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

                learn_detail = LearnDetail(
                    interaction_id=interaction.id,
                    concept=concept,
                    total_beats=lesson_data.get("total_beats"),
                    beat_types=beat_types,
                    sources_used=sources_used,
                )
                session.add(learn_detail)

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

        logger.info(f"[Logger] Learn interaction saved (id={interaction.id})")

    except Exception as e:
        logger.error(f"[Logger] Failed to save learn interaction: {e}", exc_info=True)