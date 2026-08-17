# app/services/pipeline.py

import logging
from app.services.retriever import retrieve
from app.services.generator import generate_response, generate_lesson, generate_text_lesson
from app.utils.preprocess import preprocess
from app.core.instrumentation import instrument_response

logger = logging.getLogger(__name__)

CONVERSATIONAL_TRIGGERS = {
    "hello", "hi", "hey", "who are you", "what are you",
    "what can you do", "help", "thanks", "thank you", "bye", "goodbye"
}


def is_conversational(query: str) -> bool:
    q = query.lower().strip().rstrip("?!")
    return any(trigger in q for trigger in CONVERSATIONAL_TRIGGERS)


class RAGPipeline:

    # ── /query ────────────────────────────────────────────────────────────────

    async def run_query(self, query: str) -> dict:
        processed_query = preprocess(query)
        logger.info(f"[Pipeline] Query: '{processed_query}'")

        if is_conversational(processed_query):
            logger.info("[Pipeline] Conversational — skipping retrieval")
            response        = generate_response(processed_query, context=[])
            instrumentation = instrument_response(response)
            return {
                "original_query":    query,
                "processed_query":   processed_query,
                "retrieved_context": [],
                "generated_response": response,
                "citations_present": instrumentation.citations_present,
                "citation_count":    instrumentation.citation_count,
            }

        context  = retrieve(processed_query)
        response = generate_response(processed_query, context)

        # Phase 1 — citation instrumentation only (fast, no NLI model)
        instrumentation = instrument_response(response)

        return {
            "original_query":    query,
            "processed_query":   processed_query,
            "retrieved_context": context,
            "generated_response": response,
            "citations_present": instrumentation.citations_present,
            "citation_count":    instrumentation.citation_count,
            # Grounding fields are intentionally absent here —
            # verification runs asynchronously in the background task
        }

    # ── /learn/structured ─────────────────────────────────────────────────────

    async def run_structured_lesson(self, concept: str) -> dict:
        processed_concept = preprocess(concept)
        logger.info(f"[Pipeline] Structured lesson: '{processed_concept}'")

        context     = retrieve(processed_concept, top_k=6)
        lesson_data = generate_lesson(processed_concept, context)
        lesson_data["retrieved_context"] = context
        return lesson_data

    # ── /learn/text ───────────────────────────────────────────────────────────

    async def run_text_lesson(self, concept: str, description: str | None = None) -> dict:
        processed_concept = preprocess(concept)
        logger.info(f"[Pipeline] Text lesson: '{processed_concept}'")

        context     = retrieve(processed_concept, top_k=10)
        lesson_data = generate_text_lesson(processed_concept, context, description)
        lesson_data["retrieved_context"] = context
        return lesson_data