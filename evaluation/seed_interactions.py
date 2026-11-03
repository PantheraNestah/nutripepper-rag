
"""
evaluation/seed_interactions.py

Runs all test queries through the live /query endpoint to populate
the database with verified interactions for grounding evaluation.

Run AFTER starting the uvicorn server:
    uvicorn app.main:app --reload &
    python evaluation/seed_interactions.py

This gives the verifier enough data to compute meaningful grounding
metrics when evaluate.py is run afterward.
"""

import json
import time
import logging
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("seed_interactions")

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TEST_SET_PATH = PROJECT_ROOT / "evaluation" / "test_set.json"

API_BASE_URL  = "http://localhost:8000"
DELAY_SECONDS = 1.5   # Polite delay between requests — avoids hammering the LLM API


def run_query(query: str, session_id: str = "eval_seed") -> dict | None:
    """POST a single query to /query and return the response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"user_query": query},
            headers={
                "Content-Type":  "application/json",
                "X-Session-ID":  session_id,
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Non-200 response: {response.status_code} for query: '{query[:50]}'")
            return None
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout for query: '{query[:50]}'")
        return None
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return None


def main():
    logger.info("=== NutriPepper Interaction Seeder ===")

    # Load test set
    if not TEST_SET_PATH.exists():
        logger.error(f"Test set not found at {TEST_SET_PATH}")
        logger.error("Run: python evaluation/generate_test_set.py first")
        return

    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)

    logger.info(f"Loaded {len(test_set)} test queries")
    logger.info(f"API base URL: {API_BASE_URL}")
    logger.info(f"Delay between requests: {DELAY_SECONDS}s")
    logger.info("Starting in 3 seconds — make sure uvicorn is running...")
    time.sleep(3)

    # Check server is up
    try:
        requests.get(f"{API_BASE_URL}/docs", timeout=5)
    except Exception:
        logger.error(f"Cannot reach {API_BASE_URL} — is uvicorn running?")
        return

    success_count = 0
    fail_count    = 0

    for i, item in enumerate(test_set, start=1):
        query_id = item["query_id"]
        query    = item["query"]

        logger.info(f"[{i:02d}/{len(test_set)}] {query_id}: '{query[:70]}'")

        result = run_query(query, session_id=f"eval_seed_{query_id}")

        if result and result.get("status") == "success":
            response_preview = result["data"]["generated_response"][:80]
            logger.info(f"  ✓ Response: '{response_preview}...'")
            success_count += 1
        else:
            logger.warning(f"  ✗ Failed")
            fail_count += 1

        # Delay between requests
        if i < len(test_set):
            time.sleep(DELAY_SECONDS)

    logger.info("=== Seeding Complete ===")
    logger.info(f"Successful: {success_count}/{len(test_set)}")
    logger.info(f"Failed:     {fail_count}/{len(test_set)}")
    logger.info("")
    logger.info("Background verification is now running for each interaction.")
    logger.info(f"Wait ~60 seconds for all verifications to complete, then run:")
    logger.info("  python evaluation/evaluate.py")


if __name__ == "__main__":
    main()
