# app/services/warmup.py
"""Load every model and index at startup so the first request is not slow."""

import logging
import os
import time

logger = logging.getLogger(__name__)


def preload_enabled() -> bool:
    """Set PRELOAD_MODELS=false to skip preloading (fast start for dev, or tests)."""
    return os.getenv("PRELOAD_MODELS", "true").lower() not in ("0", "false", "no")


def warm_up() -> None:
    """
    Blocking: loads the retriever (vector store, BM25, embedding model, reranker) and the NLI
    verifier. Call from a worker thread. A failure is logged and swallowed; the models then
    load lazily on first use, as they did before preloading existed.
    """
    # Imported here so merely importing this module does not pull in torch.
    from app.services import retriever
    from app.core import verifier

    for name, fn in (("retriever", retriever.preload), ("verifier", verifier.preload)):
        start = time.monotonic()
        try:
            fn()
            logger.info(f"[Warmup] {name} ready in {time.monotonic() - start:.1f}s")
        except Exception:
            logger.exception(f"[Warmup] {name} preload failed; it will load on first use instead")
