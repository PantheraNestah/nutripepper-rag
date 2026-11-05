# core/verifier.py

import re
import logging
from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder

from app.core.logger import logger as app_logger

logger = logging.getLogger(__name__)

# ── NLI model ─────────────────────────────────────────────────────────────────
# cross-encoder/nli-MiniLM2-L6-H768 reliably returns 3-class softmax scores
# in the order [contradiction, entailment, neutral] via predict() with
# activation_fct=None. Lightweight enough for CPU inference.

_NLI_MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
_nli_model: CrossEncoder | None = None


def _get_nli_model() -> CrossEncoder:
    global _nli_model
    if _nli_model is None:
        app_logger.info(f"[Verifier] Loading NLI model: {_NLI_MODEL_NAME}")
        _nli_model = CrossEncoder(_NLI_MODEL_NAME)
        app_logger.info("[Verifier] NLI model loaded")
    return _nli_model


def preload() -> None:
    """Load the NLI model now (and run one tiny inference) instead of on first verification."""
    _get_nli_model().predict([("warm up", "warm up")])


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ClaimVerificationResult:
    claim: str
    label: str          # "entailed" | "neutral" | "contradicted"
    confidence: float
    best_source: str
    method: str         # "nli" | "keyword_overlap" | "no_context"


@dataclass
class GroundingReport:
    claims_checked: int
    claims_supported: int
    claims_neutral: int
    claims_contradicted: int
    grounding_score: float
    grounding_label: str
    has_contradiction: bool
    claim_results: list[ClaimVerificationResult] = field(default_factory=list)


# ── Claim extraction ──────────────────────────────────────────────────────────

def extract_claims(response_text: str) -> list[str]:
    """
    Split a response into individual verifiable claims.
    Filters out short sentences and meta-commentary.
    """
    cleaned = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', response_text).strip()
    raw_sentences = re.split(r'(?<=[.!?])\s+', cleaned)

    meta_prefixes = (
        "based on", "according to the", "the context",
        "i am sorry", "please note", "it is important to note",
        "in summary", "to summarise", "in conclusion", "remember,",
        "practical takeaway",
    )

    claims = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 6:
            continue
        if sentence.lower().startswith(meta_prefixes):
            continue
        claims.append(sentence)

    return claims


# ── Keyword overlap fallback ──────────────────────────────────────────────────

def _keyword_overlap_score(claim: str, chunk_text: str) -> float:
    """
    Compute simple keyword overlap between a claim and a chunk.
    Used as a fallback when NLI confidence is low (e.g. tabular data chunks).

    Returns a score 0.0–1.0 representing proportion of claim content words
    found in the chunk.
    """
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "for", "of", "in", "to", "and", "or", "but", "with", "as",
        "it", "its", "this", "that", "which", "by", "at", "from",
        "can", "may", "when", "how", "what", "your", "you", "we",
        "they", "their", "our", "also", "some", "all", "more", "on",
    }

    claim_words = {
        w.lower().strip(".,;:()\"'") for w in claim.split()
        if w.lower().strip(".,;:()\"'") not in stopwords
        and len(w) > 2
    }
    chunk_words = {
        w.lower().strip(".,;:()\"'") for w in chunk_text.split()
        if len(w) > 2
    }

    if not claim_words:
        return 0.0

    overlap = claim_words & chunk_words
    return len(overlap) / len(claim_words)


# ── NLI scoring ───────────────────────────────────────────────────────────────

def _score_claim_against_chunks(
    claim: str,
    chunks: list[dict],
    model: CrossEncoder,
) -> ClaimVerificationResult:
    """
    Score a single claim against all retrieved chunks.

    Primary method: NLI via CrossEncoder (3-class softmax).
    Fallback: keyword overlap when NLI max confidence is below threshold
              (handles tabular/list-style chunks poorly suited for NLI).

    Label indices for cross-encoder/nli-MiniLM2-L6-H768:
        0 → contradiction
        1 → entailment
        2 → neutral
    """
    if not chunks:
        return ClaimVerificationResult(
            claim=claim, label="neutral", confidence=0.0,
            best_source="no_context", method="no_context",
        )

    pairs  = [(chunk.get("text", ""), claim) for chunk in chunks]
    scores = model.predict(pairs, apply_softmax=True)

    # scores shape: (n_chunks, 3) — [contradiction, entailment, neutral] per chunk
    best_entail_score  = 0.0
    best_contra_score  = 0.0
    best_entail_source = ""
    best_contra_source = ""
    max_any_confidence = 0.0

    for i, score_row in enumerate(scores):
        contra_score  = float(score_row[0])
        entail_score  = float(score_row[1])
        source        = chunks[i].get("source", "unknown")
        max_score     = max(contra_score, entail_score, float(score_row[2]))

        if max_score > max_any_confidence:
            max_any_confidence = max_score

        if entail_score > best_entail_score:
            best_entail_score  = entail_score
            best_entail_source = source

        if contra_score > best_contra_score:
            best_contra_score  = contra_score
            best_contra_source = source

    # ── NLI confidence thresholds ─────────────────────────────────────────────
    # If NLI is confident enough, trust it directly
    if best_contra_score > 0.60:
        return ClaimVerificationResult(
            claim=claim, label="contradicted",
            confidence=best_contra_score,
            best_source=best_contra_source,
            method="nli",
        )

    if best_entail_score > 0.45:
        return ClaimVerificationResult(
            claim=claim, label="entailed",
            confidence=best_entail_score,
            best_source=best_entail_source,
            method="nli",
        )

    # ── Keyword overlap fallback ──────────────────────────────────────────────
    # NLI was not confident (common with tabular/list-style chunks).
    # Use keyword overlap as a softer grounding signal.
    best_overlap_score  = 0.0
    best_overlap_source = ""

    for chunk in chunks:
        overlap = _keyword_overlap_score(claim, chunk.get("text", ""))
        if overlap > best_overlap_score:
            best_overlap_score  = overlap
            best_overlap_source = chunk.get("source", "unknown")

    # Keyword overlap thresholds:
    # ≥ 0.35 → treat as weakly entailed (grounded in the knowledge base)
    # < 0.35 → neutral (can't verify)
    if best_overlap_score >= 0.35:
        return ClaimVerificationResult(
            claim=claim, label="entailed",
            confidence=best_overlap_score,
            best_source=best_overlap_source,
            method="keyword_overlap",
        )

    return ClaimVerificationResult(
        claim=claim, label="neutral",
        confidence=best_overlap_score,
        best_source=best_overlap_source or "unknown",
        method="keyword_overlap",
    )


# ── Grounding label ───────────────────────────────────────────────────────────

def _assign_grounding_label(score: float, has_contradiction: bool) -> str:
    if has_contradiction:
        return "partial"
    if score >= 0.75:
        return "grounded"
    if score >= 0.45:
        return "partial"
    return "ungrounded"


# ── Public interface ──────────────────────────────────────────────────────────

def verify_response(response_text: str, retrieved_context: list[dict]) -> GroundingReport:
    """
    Verify a generated response against its retrieved context.
    Runs after response is sent — never blocks Pepper's answer.
    """
    if not retrieved_context:
        logger.info("[Verifier] No context — skipping (conversational)")
        return GroundingReport(
            claims_checked=0, claims_supported=0, claims_neutral=0,
            claims_contradicted=0, grounding_score=1.0,
            grounding_label="grounded", has_contradiction=False,
        )

    claims = extract_claims(response_text)
    if not claims:
        logger.info("[Verifier] No verifiable claims extracted")
        return GroundingReport(
            claims_checked=0, claims_supported=0, claims_neutral=0,
            claims_contradicted=0, grounding_score=1.0,
            grounding_label="grounded", has_contradiction=False,
        )

    model = _get_nli_model()
    logger.info(f"[Verifier] Verifying {len(claims)} claims against {len(retrieved_context)} chunks")

    claim_results = [
        _score_claim_against_chunks(claim, retrieved_context, model)
        for claim in claims
    ]

    # Log method breakdown for transparency
    nli_count     = sum(1 for r in claim_results if r.method == "nli")
    keyword_count = sum(1 for r in claim_results if r.method == "keyword_overlap")
    logger.info(f"[Verifier] Methods used — NLI: {nli_count}, keyword_overlap: {keyword_count}")

    supported    = sum(1 for r in claim_results if r.label == "entailed")
    neutral      = sum(1 for r in claim_results if r.label == "neutral")
    contradicted = sum(1 for r in claim_results if r.label == "contradicted")
    total        = len(claim_results)
    has_contra   = contradicted > 0
    score        = supported / total if total > 0 else 0.0
    label        = _assign_grounding_label(score, has_contra)

    logger.info(
        f"[Verifier] Result: score={score:.2f} label={label} "
        f"supported={supported}/{total} contradicted={contradicted}"
    )

    return GroundingReport(
        claims_checked=total,
        claims_supported=supported,
        claims_neutral=neutral,
        claims_contradicted=contradicted,
        grounding_score=round(score, 4),
        grounding_label=label,
        has_contradiction=has_contra,
        claim_results=claim_results,
    )