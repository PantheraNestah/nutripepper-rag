# tests/test_llm.py
# Run from the project root:  python -m pytest tests -q
# Uses fake provider clients: no network, no API keys.

from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

from app.services import llm
from app.services.llm import LLMError, Tier, complete, parse_json_object, parse_spec, spec_for


# ── fakes ─────────────────────────────────────────────────────────────────────

def anthropic_response(text="hello", stop_reason="end_turn", category=None):
    return NS(
        stop_reason=stop_reason,
        stop_details=NS(category=category) if stop_reason == "refusal" else None,
        content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
    )


class FakeAnthropic:
    def __init__(self, responses=None, beta_error=None):
        self.calls = []                       # (endpoint, kwargs)
        self._responses = list(responses or [anthropic_response()])
        self._beta_error = beta_error
        self.messages = NS(create=lambda **kw: self._create("messages", kw))
        self.beta = NS(messages=NS(create=lambda **kw: self._create("beta", kw)))

    def _create(self, endpoint, kwargs):
        self.calls.append((endpoint, kwargs))
        if endpoint == "beta" and self._beta_error:
            raise self._beta_error
        return self._responses.pop(0)


def openai_response(content="hi", finish_reason="stop"):
    return NS(choices=[NS(finish_reason=finish_reason, message=NS(content=content))])


class FakeOpenAI:
    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)

        def create(**kw):
            self.calls.append(kw)
            return self._responses.pop(0)

        self.chat = NS(completions=NS(create=create))


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    llm._clients.clear()
    for name in ("LLM_MODEL_COMPLEX", "LLM_MODEL_MEDIUM", "LLM_MODEL_EASY", "ANTHROPIC_REFUSAL_FALLBACK", "ANTHROPIC_PROMPT_CACHE"):
        monkeypatch.delenv(name, raising=False)
    yield
    llm._clients.clear()


# ── spec parsing / tier mapping ───────────────────────────────────────────────

def test_parse_spec_defaults_effort_to_medium():
    s = parse_spec("anthropic:claude-haiku-5-5")
    assert (s.provider, s.model, s.effort) == ("anthropic", "claude-haiku-5-5", "medium")


@pytest.mark.parametrize("bad", ["", "claude-haiku-5-5", "mistral:x", "anthropic:x:turbo", "a:b:c:d", "anthropic::low"])
def test_parse_spec_rejects_bad_input(bad):
    with pytest.raises(LLMError):
        parse_spec(bad)


def test_default_tier_mapping():
    assert str(spec_for(Tier.COMPLEX)) == "anthropic:claude-sonnet-5-5:medium"
    assert str(spec_for(Tier.MEDIUM)) == "anthropic:claude-haiku-5-5:medium"
    assert str(spec_for(Tier.EASY)) == "anthropic:claude-haiku-5-5:medium"


def test_env_overrides_tier(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_COMPLEX", "deepseek:deepseek-flash:high")
    assert str(spec_for(Tier.COMPLEX)) == "deepseek:deepseek-flash:high"


# ── anthropic ─────────────────────────────────────────────────────────────────

def test_haiku_uses_plain_endpoint_without_sampling_params():
    fake = FakeAnthropic()
    llm._clients["anthropic"] = fake
    out = complete(Tier.MEDIUM, system="S", user="U", max_tokens=1234, temperature=0.2, json_mode=True)

    assert out == "hello"                                  # thinking block skipped, text joined
    endpoint, kw = fake.calls[0]
    assert endpoint == "messages"
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["max_tokens"] == 1234
    assert kw["system"] == [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}]
    assert kw["messages"] == [{"role": "user", "content": "U"}]
    for forbidden in ("temperature", "top_p", "thinking", "response_format", "fallbacks"):
        assert forbidden not in kw


def test_sonnet_uses_refusal_fallback_by_default():
    fake = FakeAnthropic()
    llm._clients["anthropic"] = fake
    complete(Tier.COMPLEX, system="S", user="U", max_tokens=100)
    endpoint, kw = fake.calls[0]
    assert endpoint == "beta"
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]


def test_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_REFUSAL_FALLBACK", "false")
    fake = FakeAnthropic()
    llm._clients["anthropic"] = fake
    complete(Tier.COMPLEX, system="S", user="U", max_tokens=100)
    assert fake.calls[0][0] == "messages"


def test_fallback_rejection_retries_without_it():
    err = anthropic.BadRequestError(
        "nope",
        response=httpx2.Response(400, request=httpx2.Request("POST", "http://x")),
        body=None,
    )
    fake = FakeAnthropic(responses=[anthropic_response("ok")], beta_error=err)
    llm._clients["anthropic"] = fake
    assert complete(Tier.COMPLEX, system="S", user="U", max_tokens=100) == "ok"
    assert [c[0] for c in fake.calls] == ["beta", "messages"]
    assert "fallbacks" not in fake.calls[1][1]


def test_anthropic_effort_off_becomes_low(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_EASY", "anthropic:claude-haiku-5-5:off")
    fake = FakeAnthropic()
    llm._clients["anthropic"] = fake
    complete(Tier.EASY, system="S", user="U", max_tokens=100)
    assert fake.calls[0][1]["output_config"] == {"effort": "low"}


def test_refusal_raises_llmerror():
    llm._clients["anthropic"] = FakeAnthropic([anthropic_response(stop_reason="refusal", category="bio")])
    with pytest.raises(LLMError, match="declined"):
        complete(Tier.MEDIUM, system="S", user="U", max_tokens=100)


def test_empty_anthropic_output_raises():
    llm._clients["anthropic"] = FakeAnthropic([anthropic_response(text="")])
    with pytest.raises(LLMError, match="empty"):
        complete(Tier.MEDIUM, system="S", user="U", max_tokens=100)


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        complete(Tier.MEDIUM, system="S", user="U", max_tokens=100)


def test_cache_control_only_on_system_block_never_top_level_or_user():
    fake = FakeAnthropic()
    llm._clients["anthropic"] = fake
    complete(Tier.MEDIUM, system="S", user="U", max_tokens=100)
    kw = fake.calls[0][1]
    assert "cache_control" not in kw                              # no top-level auto caching
    assert kw["messages"] == [{"role": "user", "content": "U"}]   # unique tail stays uncached


def test_sonnet_fallback_path_also_caches():
    fake = FakeAnthropic()
    llm._clients["anthropic"] = fake
    complete(Tier.COMPLEX, system="S", user="U", max_tokens=100)
    endpoint, kw = fake.calls[0]
    assert endpoint == "beta"
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_prompt_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_PROMPT_CACHE", "false")
    fake = FakeAnthropic()
    llm._clients["anthropic"] = fake
    complete(Tier.MEDIUM, system="S", user="U", max_tokens=100)
    assert fake.calls[0][1]["system"] == "S"


def test_cache_usage_is_logged(caplog):
    resp = anthropic_response()
    resp.usage = NS(input_tokens=40, cache_creation_input_tokens=0,
                    cache_read_input_tokens=800, output_tokens=120)
    llm._clients["anthropic"] = FakeAnthropic([resp])
    with caplog.at_level("INFO", logger="app.services.llm"):
        complete(Tier.MEDIUM, system="S", user="U", max_tokens=100)
    assert "cache_read=800" in caplog.text and "cache_write=0" in caplog.text


# ── deepseek (openai-format) ──────────────────────────────────────────────────

def test_deepseek_thinking_on_with_reasoning_effort(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_COMPLEX", "deepseek:deepseek-flash:high")
    fake = FakeOpenAI([openai_response('{"a": 1}')])
    llm._clients["deepseek"] = fake
    complete(Tier.COMPLEX, system="S", user="U", max_tokens=500, json_mode=True, temperature=0.4)
    kw = fake.calls[0]
    assert kw["model"] == "deepseek-flash"
    assert kw["reasoning_effort"] == "high"
    assert kw["extra_body"] == {"thinking": {"type": "enabled"}}
    assert kw["response_format"] == {"type": "json_object"}
    assert "temperature" not in kw                           # ignored in thinking mode
    assert kw["messages"][0] == {"role": "system", "content": "S"}


def test_deepseek_flash_low_effort(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_EASY", "deepseek:deepseek-flash:low")
    fake = FakeOpenAI([openai_response("answer")])
    llm._clients["deepseek"] = fake
    complete(Tier.EASY, system="S", user="U", max_tokens=500, temperature=0.7)
    kw = fake.calls[0]
    assert kw["model"] == "deepseek-flash"
    assert kw["reasoning_effort"] == "low"
    assert kw["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in kw


def test_deepseek_usage_logged_with_cache_and_reasoning(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace as NS
    monkeypatch.setenv("LLM_MODEL_EASY", "deepseek:deepseek-flash:low")
    resp = openai_response("answer")
    resp.usage = NS(prompt_tokens=900, prompt_cache_hit_tokens=640, prompt_cache_miss_tokens=260,
                    completion_tokens=300, completion_tokens_details=NS(reasoning_tokens=120))
    llm._clients["deepseek"] = FakeOpenAI([resp])
    with caplog.at_level(logging.INFO, logger="app.services.llm"):
        complete(Tier.EASY, system="S", user="U", max_tokens=500)
    assert "cache_hit=640" in caplog.text and "reasoning=120" in caplog.text


def test_deepseek_off_disables_thinking_and_keeps_temperature(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_MEDIUM", "deepseek:deepseek-flash:off")
    fake = FakeOpenAI([openai_response("answer")])
    llm._clients["deepseek"] = fake
    complete(Tier.MEDIUM, system="S", user="U", max_tokens=500, temperature=0.2)
    kw = fake.calls[0]
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kw["temperature"] == 0.2
    assert "reasoning_effort" not in kw and "response_format" not in kw


def test_deepseek_blank_json_is_retried_once(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_MEDIUM", "deepseek:deepseek-flash:off")
    fake = FakeOpenAI([openai_response("   "), openai_response('{"ok": true}')])
    llm._clients["deepseek"] = fake
    out = complete(Tier.MEDIUM, system="S", user="U", max_tokens=500, json_mode=True)
    assert parse_json_object(out) == {"ok": True}
    assert len(fake.calls) == 2


def test_mixed_providers_per_tier(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_EASY", "deepseek:deepseek-flash:off")
    a, d = FakeAnthropic(), FakeOpenAI([openai_response("from deepseek")])
    llm._clients.update(anthropic=a, deepseek=d)
    assert complete(Tier.MEDIUM, system="S", user="U", max_tokens=10) == "hello"
    assert complete(Tier.EASY, system="S", user="U", max_tokens=10) == "from deepseek"


# ── json parsing ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    'Here you go:\n{"a": 1}\nHope that helps!',
])
def test_parse_json_object_tolerates_wrappers(text):
    assert parse_json_object(text) == {"a": 1}


def test_parse_json_object_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_json_object("no json here")


# ── generator wiring ──────────────────────────────────────────────────────────

def test_generator_tiers_and_fallbacks(monkeypatch):
    from app.services import generator

    seen = []

    def fake_complete(tier, **kw):
        seen.append((tier, kw["max_tokens"], kw.get("json_mode", False)))
        return '{"concept": "x", "total_beats": 1, "beats": []}' if kw.get("json_mode") else "answer"

    monkeypatch.setattr(generator, "complete", fake_complete)
    ctx = [{"source": "s", "authority": "a", "text": "t"}]

    assert generator.generate_response("hi", []) == "answer"          # no context -> EASY
    assert generator.generate_response("q", ctx) == "answer"          # grounded -> MEDIUM
    generator.generate_lesson("x", ctx)                                # structured -> MEDIUM
    generator.generate_text_lesson("x", ctx)                           # long-form -> COMPLEX

    assert [s[0] for s in seen] == [Tier.EASY, Tier.MEDIUM, Tier.MEDIUM, Tier.COMPLEX]
    assert seen[3][1] == generator.TEXT_LESSON_MAX_TOKENS and seen[3][2] is True

    def boom(*a, **kw):
        raise LLMError("down")

    monkeypatch.setattr(generator, "complete", boom)
    assert "sorry" in generator.generate_response("q", ctx).lower()
    assert generator.generate_lesson("x", ctx)["total_beats"] == 1
    assert generator.generate_text_lesson("x", ctx)["quiz_hints"] == []
