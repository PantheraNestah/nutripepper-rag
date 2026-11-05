"""Regression tests for the 'database is locked' failures.

Cause: blocking work (NLI verification, LLM calls) ran on the event loop, so write
transactions in other background tasks could not finish and later writers timed out.
"""
import asyncio
import time

from sqlalchemy import text

from app.database import db
from app.services import interaction_logger as il
from app.services import pipeline as pipeline_mod


async def _max_loop_stall(coro, tick=0.02):
    """Run `coro` while a heartbeat measures the longest gap the event loop was unresponsive."""
    gaps, done = [], False

    async def heartbeat():
        last = time.monotonic()
        while not done:
            await asyncio.sleep(tick)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(tick * 3)            # let the heartbeat start ticking before the work begins
    try:
        await coro
    finally:
        done = True
        await hb
    return max(gaps)


def test_sqlite_uses_wal_and_long_busy_timeout():
    async def run():
        await db.init_db()
        async with db.engine.connect() as conn:
            mode = (await conn.execute(text("PRAGMA journal_mode"))).scalar()
            timeout_ms = (await conn.execute(text("PRAGMA busy_timeout"))).scalar()
        return mode, timeout_ms

    mode, timeout_ms = asyncio.run(run())
    assert mode.lower() == "wal"
    assert timeout_ms >= 30000


def test_verification_does_not_block_the_event_loop(monkeypatch):
    from types import SimpleNamespace as NS

    def slow_verify(response, context):
        time.sleep(0.6)                      # stands in for the CPU-heavy NLI step
        return NS(grounding_score=1.0, grounding_label="grounded", has_contradiction=False,
                  claims_checked=1, claims_supported=1)

    monkeypatch.setattr(il, "verify_response", slow_verify)

    async def run():
        await db.init_db()
        task = il.log_query_interaction("s", "q", "q", [{"text": "t"}], "resp", 1.0, False)
        stall = await _max_loop_stall(task)
        async with db.AsyncSessionLocal() as s:
            rows = (await s.execute(text("select count(*) from interactions"))).scalar()
        return stall, rows

    stall, rows = asyncio.run(run())
    assert stall < 0.3, f"event loop was blocked for {stall:.2f}s during verification"
    assert rows >= 1


def test_concurrent_verifications_run_one_at_a_time(monkeypatch):
    from types import SimpleNamespace as NS
    active, peak = 0, 0

    def slow_verify(response, context):
        nonlocal active, peak
        active += 1; peak = max(peak, active)
        time.sleep(0.2)
        active -= 1
        return NS(grounding_score=1.0, grounding_label="grounded", has_contradiction=False,
                  claims_checked=1, claims_supported=1)

    monkeypatch.setattr(il, "verify_response", slow_verify)

    async def run():
        await db.init_db()
        await asyncio.gather(*[
            il.log_query_interaction("s", f"q{i}", f"q{i}", [{"text": "t"}], "r", 1.0, False)
            for i in range(3)
        ])

    asyncio.run(run())
    assert peak == 1


def test_pipeline_llm_calls_do_not_block_the_event_loop(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "retrieve", lambda *a, **k: [{"text": "t", "source": "s"}])

    def slow_generate(query, context):
        time.sleep(0.6)                      # stands in for a 10-40 s network call
        return "answer [1]"

    monkeypatch.setattr(pipeline_mod, "generate_response", slow_generate)

    async def run():
        return await _max_loop_stall(pipeline_mod.RAGPipeline().run_query("what is iron?"))

    stall = asyncio.run(run())
    assert stall < 0.3, f"event loop was blocked for {stall:.2f}s during generation"


def test_retrieval_does_not_block_the_event_loop(monkeypatch):
    def slow_retrieve(query, top_k=3):
        time.sleep(0.6)                      # stands in for embedding + reranking on CPU
        return [{"text": "t", "source": "s"}]

    monkeypatch.setattr(pipeline_mod, "retrieve", slow_retrieve)
    monkeypatch.setattr(pipeline_mod, "generate_response", lambda q, c: "answer [1]")

    async def run():
        return await _max_loop_stall(pipeline_mod.RAGPipeline().run_query("what is iron?"))

    stall = asyncio.run(run())
    assert stall < 0.3, f"event loop was blocked for {stall:.2f}s during retrieval"
