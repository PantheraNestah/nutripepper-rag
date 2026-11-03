# core/instrumentation.py

import re
from dataclasses import dataclass


@dataclass
class ResponseInstrumentation:
    """
    Lightweight metrics computed from a generated response at response time.
    Zero extra LLM calls — purely rule-based.
    Populated immediately after generation, stored in QueryDetail.
    """
    citations_present: bool
    citation_count: int


def instrument_response(response_text: str) -> ResponseInstrumentation:
    """
    Extract citation instrumentation from a generated response.

    Looks for inline citations in the form [1], [2], [1,2], [1, 2].
    These are inserted by the LLM when it follows the grounding prompt instructions.

    Args:
        response_text: The raw generated response string.

    Returns:
        ResponseInstrumentation with citation presence and count.
    """
    # Match patterns like [1], [2], [1,2], [1, 2], [10]
    citation_pattern = re.compile(r'\[\d+(?:,\s*\d+)*\]')
    matches = citation_pattern.findall(response_text)

    # Count unique source indices referenced across all citation tags
    unique_indices = set()
    for match in matches:
        # Extract all numbers from e.g. "[1, 2]" → {1, 2}
        indices = re.findall(r'\d+', match)
        unique_indices.update(indices)

    return ResponseInstrumentation(
        citations_present=len(matches) > 0,
        citation_count=len(unique_indices),
    )