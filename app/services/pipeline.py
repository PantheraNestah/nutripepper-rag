from app.utils.preprocess import preprocess_query
from app.services.retriever import retrieve_documents
from app.services.generator import generate_response, generate_lesson
from app.core.logger import logger

CONVERSATIONAL_TRIGGERS = {
    "hello", "hi", "hey", "who are you", "what are you",
    "what can you do", "help", "thanks", "thank you", "bye", "goodbye"
}

def is_conversational(query: str) -> bool:
    q = query.lower().strip().rstrip("?!")
    return any(trigger in q for trigger in CONVERSATIONAL_TRIGGERS)

class RAGPipeline:
 
    # ── /query flow ───────────────────────────────────────────────────────────
 
    async def run_query(self, query: str) -> dict:
        processed_query = preprocess_query(query)
        logger.info(f"[Pipeline] Query received: '{processed_query}'")
 
        if is_conversational(processed_query):
            logger.info("[Pipeline] Conversational query detected — skipping retrieval")
            response = generate_response(processed_query, context=[])
            return {
                "original_query": query,
                "processed_query": processed_query,
                "retrieved_context": [],
                "generated_response": response,
            }
 
        context = retrieve_documents(processed_query)
        logger.info(f"[Pipeline] Retrieved {len(context)} chunks for query")
 
        response = generate_response(processed_query, context)
        return {
            "original_query": query,
            "processed_query": processed_query,
            "retrieved_context": context,
            "generated_response": response,
        }
 
    # ── /learn flow ───────────────────────────────────────────────────────────
 
    async def run_lesson(self, concept: str) -> dict:
        """
        Retrieve relevant knowledge chunks for a concept and generate
        a structured teaching lesson for Pepper.
 
        Args:
            concept: The nutritional concept entered via the tablet.
 
        Returns:
            A dict matching the LearnData schema.
        """
        processed_concept = preprocess_query(concept)
        logger.info(f"[Pipeline] Lesson requested for concept: '{processed_concept}'")
 
        # Retrieve more chunks for lessons than for queries — lessons need
        # broader context to cover introduction, local foods, facts, and a quiz
        context = retrieve_documents(processed_concept, top_k=6)
        logger.info(f"[Pipeline] Retrieved {len(context)} chunks for lesson")
 
        lesson_data = generate_lesson(processed_concept, context)
        return lesson_data
 
