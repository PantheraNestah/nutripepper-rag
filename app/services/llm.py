# app/services/llm.py
"""
Provider-agnostic LLM access.

Callers ask for a *tier* (how demanding the task is), not for a model:

    from app.services.llm import Tier, complete
    text = complete(Tier.MEDIUM, system=SYSTEM, user=PROMPT, max_tokens=2000)

Each tier maps to a "provider:model[:effort]" spec, configurable via env vars:

    LLM_MODEL_COMPLEX   default  anthropic:claude-sonnet-5-5:medium
    LLM_MODEL_MEDIUM    default  anthropic:claude-haiku-5-5:medium
    LLM_MODEL_EASY      default  anthropic:claude-haiku-5-5:medium

Providers
    anthropic  native Anthropic SDK          key: ANTHROPIC_API_KEY
    deepseek   OpenAI-format API (openai SDK) key: DEEPSEEK_API_KEY   (DEEPSEEK_BASE_URL optional)
    openai     OpenAI API (openai SDK)        key: OPENAI_API_KEY

Effort is one of: off | low | medium | high | max
    anthropic  low/medium/high/max -> output_config.effort ("off" is treated as "low":
               Sonnet/Haiku 5.5 cannot switch thinking off).
    deepseek   off -> thinking disabled; otherwise thinking on with reasoning_effort
               low | high | max (DeepSeek maps minimal -> low, medium -> high itself).
               Model id "deepseek-flash" is served by DeepSeek-V4.1-Flash-1017.
               Reasoning tokens count against max_tokens; temperature is ignored while thinking.
    openai     effort is ignored.

Prompt caching
    anthropic  the system prompt is sent with a cache_control breakpoint (disable with
               ANTHROPIC_PROMPT_CACHE=false). Cache hits show up in the "[LLM] ... usage" log line.
    deepseek / openai  cache the shared prompt prefix automatically; no request parameter exists.
               The system prompt is always first, so the stable part leads.

Clients are created lazily, so a provider whose key is missing only fails when a
tier that actually uses it is called.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── Public types ──────────────────────────────────────────────────────────────

class Tier(str, Enum):
    COMPLEX = "complex"   # long-form or high-stakes generation
    MEDIUM  = "medium"    # grounded answers, structured output
    EASY    = "easy"      # short, simple generations


class LLMError(RuntimeError):
    """The model returned nothing usable (refusal, empty output, bad config)."""


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model:    str
    effort:   str = "medium"

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}:{self.effort}"


DEFAULT_SPECS = {
    Tier.COMPLEX: "anthropic:claude-sonnet-5-5:medium",
    Tier.MEDIUM:  "anthropic:claude-haiku-5-5:medium",
    Tier.EASY:    "anthropic:claude-haiku-5-5:medium",
}

_EFFORTS   = {"off", "low", "medium", "high", "max"}
_PROVIDERS = {"anthropic", "deepseek", "openai"}

# Models for which Anthropic's server-side refusal fallback is available.
_FALLBACK_MODELS = {"claude-sonnet-5-5"}
_FALLBACK_BETA   = "server-side-fallback-2026-07-01"


def parse_spec(raw: str) -> ModelSpec:
    """Parse 'provider:model[:effort]'. Model ids never contain ':'."""
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) not in (2, 3) or not all(parts):
        raise LLMError(f"Bad model spec '{raw}'. Expected 'provider:model[:effort]'.")
    provider, model = parts[0].lower(), parts[1]
    effort = parts[2].lower() if len(parts) == 3 else "medium"
    if provider not in _PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}' in '{raw}'. Use one of {sorted(_PROVIDERS)}.")
    if effort not in _EFFORTS:
        raise LLMError(f"Unknown effort '{effort}' in '{raw}'. Use one of {sorted(_EFFORTS)}.")
    return ModelSpec(provider, model, effort)


def spec_for(tier: Tier) -> ModelSpec:
    raw = os.getenv(f"LLM_MODEL_{tier.name}") or DEFAULT_SPECS[tier]
    return parse_spec(raw)


# ── Client factories (lazy, cached) ───────────────────────────────────────────

_clients: dict = {}


def _require_key(env_var: str, provider: str) -> str:
    key = os.getenv(env_var)
    if not key:
        raise LLMError(f"{env_var} is not set but a tier is configured to use '{provider}'.")
    return key


def _anthropic_client():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY", "anthropic"))
    return _clients["anthropic"]


def _openai_compatible_client(provider: str):
    if provider not in _clients:
        from openai import OpenAI
        if provider == "deepseek":
            _clients[provider] = OpenAI(
                api_key=_require_key("DEEPSEEK_API_KEY", "deepseek"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
        else:
            _clients[provider] = OpenAI(api_key=_require_key("OPENAI_API_KEY", "openai"))
    return _clients[provider]


# ── Provider calls ────────────────────────────────────────────────────────────

def _call_anthropic(spec: ModelSpec, system: str, user: str, max_tokens: int) -> str:
    client = _anthropic_client()
    effort = "low" if spec.effort == "off" else spec.effort

    # Prompt caching: one explicit breakpoint at the end of the system prompt, which is
    # identical on every call of a given task. The user message (retrieved context +
    # question) is unique per request, so top-level automatic caching is deliberately NOT
    # used: its breakpoint would land after that unique tail and only add write cost.
    # Prefixes under the model minimum (512 tokens on Sonnet/Haiku 5.5) silently don't
    # cache; _log_usage() reports what actually happened.
    if os.getenv("ANTHROPIC_PROMPT_CACHE", "true").lower() not in ("0", "false", "no"):
        system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        system_param = system

    # Sonnet/Haiku 5.5 take no temperature/top_p, and thinking is always adaptive.
    # Thinking tokens count against max_tokens, so callers must leave headroom.
    kwargs = dict(
        model=spec.model,
        max_tokens=max_tokens,
        system=system_param,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort},
    )

    use_fallback = (
        spec.model in _FALLBACK_MODELS
        and os.getenv("ANTHROPIC_REFUSAL_FALLBACK", "true").lower() not in ("0", "false", "no")
    )
    if use_fallback:
        try:
            response = client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
        except Exception as e:
            import anthropic
            if not isinstance(e, anthropic.BadRequestError):
                raise
            logger.warning(f"[LLM] Refusal-fallback request rejected ({e}); retrying without it. "
                           "Set ANTHROPIC_REFUSAL_FALLBACK=false to stop trying.")
            response = client.messages.create(**kwargs)
    else:
        response = client.messages.create(**kwargs)

    _log_usage(spec, response)

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMError(f"{spec.model} declined the request (category: {getattr(details, 'category', None)}).")
    if response.stop_reason == "max_tokens":
        logger.warning(f"[LLM] {spec.model} hit max_tokens={max_tokens}; output may be truncated.")

    return "".join(b.text for b in response.content if b.type == "text").strip()


def _log_usage(spec: ModelSpec, response) -> None:
    """Log token usage, including prompt-cache writes/reads (the only proof caching works)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    logger.info(
        f"[LLM] {spec.model} usage: input={getattr(usage, 'input_tokens', None)} "
        f"cache_write={getattr(usage, 'cache_creation_input_tokens', None)} "
        f"cache_read={getattr(usage, 'cache_read_input_tokens', None)} "
        f"output={getattr(usage, 'output_tokens', None)}"
    )


def _call_openai_compatible(spec: ModelSpec, system: str, user: str, max_tokens: int,
                            json_mode: bool, temperature: float | None) -> str:
    client = _openai_compatible_client(spec.provider)
    kwargs = dict(
        model=spec.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    if spec.provider == "deepseek":
        if spec.effort == "off":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            if temperature is not None:
                kwargs["temperature"] = temperature      # honoured only with thinking off
        else:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = spec.effort     # DeepSeek maps medium -> high
    elif temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    _log_usage_openai(spec, response)
    choice = response.choices[0]
    if choice.finish_reason == "length":
        logger.warning(f"[LLM] {spec.model} hit max_tokens={max_tokens}; output may be truncated "
                       "(with thinking on, reasoning tokens count against max_tokens).")
    return (choice.message.content or "").strip()


def _log_usage_openai(spec: ModelSpec, response) -> None:
    """Log token usage; DeepSeek also reports automatic prefix-cache hits/misses."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    details = getattr(usage, "completion_tokens_details", None)
    logger.info(
        f"[LLM] {spec.model} usage: prompt={getattr(usage, 'prompt_tokens', None)} "
        f"cache_hit={getattr(usage, 'prompt_cache_hit_tokens', None)} "
        f"cache_miss={getattr(usage, 'prompt_cache_miss_tokens', None)} "
        f"completion={getattr(usage, 'completion_tokens', None)} "
        f"reasoning={getattr(details, 'reasoning_tokens', None)}"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def complete(tier: Tier, *, system: str, user: str, max_tokens: int,
             json_mode: bool = False, temperature: float | None = None) -> str:
    """
    Run one completion on the model configured for `tier` and return its text.

    json_mode    request JSON output where the provider supports it (DeepSeek/OpenAI use
                 response_format; Anthropic relies on the prompt). Use parse_json_object()
                 on the result either way. The prompt must itself ask for JSON.
    temperature  honoured only by OpenAI-format providers (and DeepSeek with effort=off);
                 ignored for Anthropic, whose current models do not accept it.
    max_tokens   include headroom for reasoning tokens.

    Raises LLMError on refusal, empty output or misconfiguration; provider SDK
    exceptions (auth, rate limit, network) propagate unchanged.
    """
    spec = spec_for(tier)
    logger.info(f"[LLM] tier={tier.value} -> {spec}")

    attempts = 2 if (json_mode and spec.provider != "anthropic") else 1   # DeepSeek can return blank JSON
    for attempt in range(1, attempts + 1):
        if spec.provider == "anthropic":
            text = _call_anthropic(spec, system, user, max_tokens)
        else:
            text = _call_openai_compatible(spec, system, user, max_tokens, json_mode, temperature)
        if text:
            return text
        logger.warning(f"[LLM] Empty response from {spec} (attempt {attempt}/{attempts})")

    raise LLMError(f"{spec} returned an empty response.")


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and stray prose."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start:end + 1])
