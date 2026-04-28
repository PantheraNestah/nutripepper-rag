from app.utils.preprocess import preprocess_query
from app.services.retriever import retrieve_documents
from app.services.generator import generate_response
from app.core.logger import logger

def run_basic_pipeline(user_query: str):
    logger.info(f"Incoming Query: {user_query}")

    processed_query = preprocess_query(user_query)
    logger.info(f"Processed Query: {processed_query}")

    retrieved_context = retrieve_documents(processed_query)
    logger.info(f"Retrieved {len(retrieved_context)} context docs")

    generated_response = generate_response(processed_query, retrieved_context)
    logger.info("Generated response successfully")

    return {
        "original_query": user_query,
        "processed_query": processed_query,
        "retrieved_context": retrieved_context,
        "generated_response": generated_response
    }