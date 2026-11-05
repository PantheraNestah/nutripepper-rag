"""Startup model preloading."""
import logging

from fastapi.testclient import TestClient

from app import main
from app.core import verifier
from app.services import retriever, warmup


def test_preload_enabled_by_default_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("PRELOAD_MODELS", raising=False)
    assert warmup.preload_enabled() is True
    for off in ("false", "0", "no", "FALSE"):
        monkeypatch.setenv("PRELOAD_MODELS", off)
        assert warmup.preload_enabled() is False


def test_warm_up_runs_retriever_and_verifier(monkeypatch):
    calls = []
    monkeypatch.setattr(retriever, "preload", lambda: calls.append("retriever"))
    monkeypatch.setattr(verifier, "preload", lambda: calls.append("verifier"))
    warmup.warm_up()
    assert calls == ["retriever", "verifier"]


def test_warm_up_failure_is_logged_not_raised(monkeypatch, caplog):
    def boom():
        raise RuntimeError("no chroma collection")

    calls = []
    monkeypatch.setattr(retriever, "preload", boom)
    monkeypatch.setattr(verifier, "preload", lambda: calls.append("verifier"))
    with caplog.at_level(logging.ERROR, logger="app.services.warmup"):
        warmup.warm_up()                          # must not raise
    assert "retriever preload failed" in caplog.text
    assert calls == ["verifier"]                  # one failure does not stop the other


def test_app_startup_runs_warm_up_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "warm_up", lambda: calls.append("warm"))
    monkeypatch.delenv("PRELOAD_MODELS", raising=False)
    with TestClient(main.app):
        assert calls == ["warm"]                  # done before the app serves anything


def test_app_startup_skips_warm_up_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "warm_up", lambda: calls.append("warm"))
    monkeypatch.setenv("PRELOAD_MODELS", "false")
    with TestClient(main.app):
        pass
    assert calls == []
