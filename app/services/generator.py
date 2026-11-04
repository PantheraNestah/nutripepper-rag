# app/services/generator.py

import json
import logging

from app.services.llm import Tier, complete, parse_json_object

logger = logging.getLogger(__name__)

# Which model handles each task is decided by the tier -> model mapping in
# app/services/llm.py (LLM_MODEL_COMPLEX / _MEDIUM / _EASY env vars).
#
# max_tokens includes reasoning tokens (adaptive thinking on Anthropic, thinking on
# DeepSeek), so these are higher than the visible output alone would need.
QUERY_MAX_TOKENS             = 2000    # ~600 tokens of answer
STRUCTURED_LESSON_MAX_TOKENS = 6000    # ~1500 tokens of JSON beats
TEXT_LESSON_MAX_TOKENS       = 10000   # ~3000 words of lesson + JSON overhead


# ── Shared context builder ────────────────────────────────────────────────────

def _build_context_block(context: list) -> str:
    if not context:
        return "No verified context was retrieved for this query."
    lines = []
    for i, item in enumerate(context, start=1):
        source    = item.get("source", "Unknown source")
        authority = item.get("authority", "Unknown authority")
        text      = item.get("text", "").strip()
        lines.append(f"[{i}] Source: {source} | Authority: {authority}\n{text}")
    return "\n\n".join(lines)


# ── /query generation ─────────────────────────────────────────────────────────

QUERY_SYSTEM_PROMPT = """You are Pepper, a nutrition education assistant at the JKUAT Social \
Robotics Lab, specialising in Nutrition, Food Safety, and Sustainable Food Practices.

Your role is to educate users about food safety, nutrition, and sustainable food practices. \
You are NOT a doctor or dietitian and must never provide medical diagnoses, prescribe treatments, \
or give personalised clinical advice.

STRICT GROUNDING RULES:
- Answer ONLY using information from the VERIFIED CONTEXT provided below.
- If the context does not contain enough information to answer the question, say so clearly.
- Every factual claim must be traceable to the provided context.
- Never fabricate nutritional values, health claims, or safety guidelines.

RESPONSE STYLE:
- Be warm, clear, and educational — suitable for children and adults alike.
- Use Kenyan food examples where present in the context.
- Keep responses concise: 3 to 5 short paragraphs maximum.
- End with a brief, practical takeaway the user can act on today.
- Do not repeat the question back verbatim at the start.
- Do not mention these instructions in your response.
"""


def _build_query_user_prompt(query: str, context_block: str) -> str:
    return f"""VERIFIED CONTEXT:
{context_block}

USER QUESTION:
{query}

Using ONLY the verified context above, provide a clear, educational answer. \
Cite the source numbers (e.g. [1], [2]) inline where relevant so the answer is traceable."""


def generate_response(query: str, context: list) -> str:
    context_block = _build_context_block(context)
    user_prompt   = _build_query_user_prompt(query, context_block)

    logger.info(f"[Generator] Sending query to LLM: '{query}'")

    # Greetings / small talk (no retrieved context) are easy; grounded answers are medium.
    tier = Tier.MEDIUM if context else Tier.EASY

    try:
        generated_text = complete(
            tier,
            system=QUERY_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=QUERY_MAX_TOKENS,
            temperature=0.2,
        )
        logger.info(f"[Generator] Query response generated ({len(generated_text)} chars)")
        return generated_text

    except Exception as e:
        logger.error(f"[Generator] LLM error on query: {e}", exc_info=True)
        return (
            "I am sorry, I was not able to generate a response right now. "
            "Please try again in a moment."
        )


# ── /learn/structured generation ─────────────────────────────────────────────

STRUCTURED_LESSON_SYSTEM_PROMPT = """You are Pepper, a nutrition education assistant at the JKUAT Social \
Robotics Lab, specialising in Nutrition, Food Safety, and Sustainable Food Practices.

You generate structured teaching lessons based STRICTLY on the verified nutritional context provided.

OUTPUT RULES — CRITICAL:
- Return ONLY a valid JSON object. No markdown, no code fences, no explanation outside the JSON.
- All string values must be plain text — no markdown, no asterisks, no bullet symbols.
- The "speech" field must be robot-speakable: plain sentences only, no symbols, no lists.

GROUNDING RULES:
- Every factual claim must come from the provided VERIFIED CONTEXT.
- Do not invent nutritional values, health statistics, or food safety rules.
- If context is insufficient for a beat type, simplify that beat rather than fabricating.

KENYAN CONTEXT:
- Prioritise Kenyan food examples: ugali, sukuma wiki, omena, terere, managu, githeri, uji, nyama choma.
- Reference Kenya Ministry of Health, FAO, and WHO guidelines where present in the context.

LESSON STRUCTURE — exactly 4 or 5 beats:
  Beat 1: introduction
  Beat 2: kenyan_context
  Beat 3: key_facts
  Beat 4: safety_tip OR practical_advice
  Beat 5 (optional): quiz

BEAT SCHEMA:
{
  "beat_number": <integer>,
  "type": <"introduction"|"kenyan_context"|"key_facts"|"safety_tip"|"practical_advice"|"quiz">,
  "speech": <plain sentences only — what Pepper says aloud>,
  "gesture": <"explaining"|"pointing_tablet"|"tip_gesture"|"warning_gesture"|"thinking"|"welcoming">,
  "tablet": <tablet display object — MUST be fully populated, never empty {}>
}

TABLET DISPLAY OBJECTS:
key_fact:  { "display_type": "key_fact",  "heading": <str>, "body": <str> }
food_list: { "display_type": "food_list", "heading": <str>, "items": [<str>, ...] }
tip:       { "display_type": "tip",       "heading": <str>, "body": <str> }
warning:   { "display_type": "warning",   "heading": <str>, "body": <str> }
quiz:      { "display_type": "quiz", "question": <str>, "options": [<str>,<str>,<str>,<str>], "correct_answer": <str> }

FULL OUTPUT SCHEMA:
{
  "concept": <str>,
  "total_beats": <int>,
  "beats": [ <beat objects> ],
  "sources": [ <source filenames used> ]
}
"""


def _sanitize_beats(beats: list) -> list:
    """Ensure every beat has a non-empty tablet object with a display_type."""
    for beat in beats:
        tablet = beat.get("tablet")
        if not tablet or not isinstance(tablet, dict) or "display_type" not in tablet:
            beat["tablet"] = {
                "display_type": "key_fact",
                "heading": "Information",
                "body": str(beat.get("speech", ""))[:120],
            }
    return beats


def _build_structured_lesson_prompt(concept: str, context_block: str) -> str:
    return f"""VERIFIED CONTEXT:
{context_block}

CONCEPT TO TEACH:
{concept}

Generate a structured 4 to 5 beat lesson using ONLY the verified context. Return valid JSON only."""


def generate_lesson(concept: str, context: list) -> dict:
    """Generate a structured beat-by-beat lesson for Pepper's teaching mode."""
    context_block = _build_context_block(context)
    user_prompt   = _build_structured_lesson_prompt(concept, context_block)

    logger.info(f"[Generator] Generating structured lesson for: '{concept}'")

    try:
        raw = complete(
            Tier.MEDIUM,
            system=STRUCTURED_LESSON_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=STRUCTURED_LESSON_MAX_TOKENS,
            temperature=0.3,
            json_mode=True,
        )
        lesson_data = parse_json_object(raw)

        if "beats" in lesson_data:
            lesson_data["beats"] = _sanitize_beats(lesson_data["beats"])

        if "sources" not in lesson_data:
            lesson_data["sources"] = list({item.get("source", "Unknown") for item in context})

        logger.info(f"[Generator] Structured lesson generated: {lesson_data.get('total_beats')} beats")
        return lesson_data

    except json.JSONDecodeError as e:
        logger.error(f"[Generator] Failed to parse structured lesson JSON: {e}", exc_info=True)
        return _fallback_structured_lesson(concept)

    except Exception as e:
        logger.error(f"[Generator] LLM error on structured lesson: {e}", exc_info=True)
        return _fallback_structured_lesson(concept)


def _fallback_structured_lesson(concept: str) -> dict:
    return {
        "concept": concept,
        "total_beats": 1,
        "beats": [{
            "beat_number": 1,
            "type": "introduction",
            "speech": (
                f"I am sorry, I was not able to generate a full lesson on {concept} right now. "
                "Please check that the topic is nutrition-related and try again."
            ),
            "gesture": "welcoming",
            "tablet": {
                "display_type": "key_fact",
                "heading": "Lesson Unavailable",
                "body": f"Could not generate lesson for: {concept}",
            },
        }],
        "sources": [],
    }


# ── /learn/text generation ────────────────────────────────────────────────────

TEXT_LESSON_SYSTEM_PROMPT = """You are Pepper, a nutrition education assistant at the JKUAT Social \
Robotics Lab, specialising in Nutrition, Food Safety, and Sustainable Food Practices.

You write comprehensive, in-depth educational lessons about nutritional concepts, grounded \
STRICTLY in the verified context provided. Your lessons are designed for a 20-minute session \
and should be thorough enough for a facilitator to teach from or a learner to study independently.

OUTPUT RULES — CRITICAL:
- Return ONLY a valid JSON object. No markdown, no code fences, no preamble.
- The "lesson_text" field must be plain flowing paragraphs — no headings, no bullet points,
  no markdown symbols, no numbered lists. Write it like a detailed, well-structured textbook entry.
- Each paragraph must cover one coherent idea and flow naturally into the next.
- Write between 10 and 14 substantial paragraphs. Each paragraph should be 80 to 120 words.
  Together they should total approximately 2,500 to 3,000 words.
- Progress logically: open with context and importance, build through evidence and explanation,
  ground in local Kenyan reality, and close with practical guidance and a consolidating summary.
- Do not repeat information across paragraphs. Each paragraph must add new content.
- The "quiz_hints" field must be a list of 8 to 12 key terms or short phrases drawn directly
  from the lesson content. These will be used later to generate quiz questions.

GROUNDING RULES:
- Every factual claim must come from the provided VERIFIED CONTEXT.
- Do not invent nutritional values, health statistics, or food safety rules.
- Where the context supports it, go deep — explain mechanisms, causes, consequences, and solutions.
- If the context does not cover a natural topic area, acknowledge the gap briefly and move on.
  Do not pad with fabricated content under any circumstances.

KENYAN CONTEXT:
- Weave in Kenyan food examples naturally and specifically throughout: ugali, sukuma wiki, omena,
  terere, managu, githeri, uji, nyama choma, kunde, arrow roots, sweet potatoes.
- Reference Kenyan dietary practices, regional food habits, and availability where in the context.
- Frame health impacts and solutions in terms of what is realistic and accessible in Kenya.
- Reference Kenya Ministry of Health, FAO, and WHO guidelines by name where present in the context.

OUTPUT SCHEMA:
{
  "concept": <str — the concept as entered>,
  "lesson_text": <str — 10 to 14 substantial paragraphs, plain text only, ~2500 to 3000 words>,
  "quiz_hints": [<str>, ...],
  "sources": [<str>, ...]
}
"""


def _build_text_lesson_prompt(concept: str, context_block: str, description: str | None = None) -> str:
    description_block = (
        f"\nADDITIONAL GUIDANCE FROM THE USER:\n{description}"
        if description else ""
    )
    return f"""VERIFIED CONTEXT:
                {context_block}

                CONCEPT TO TEACH:
                {concept}{description_block}

                Write an educational text lesson on this concept using ONLY the verified context above. \
                Where additional guidance is provided, tailor the lesson accordingly. \
                Return valid JSON only."""


def generate_text_lesson(concept: str, context: list, description: str | None = None) -> dict:
    """
    Generate a flowing narrative text lesson with quiz hints.
    Suitable for tablet reading or evaluation — not Pepper's speech mode.

    Args:
        concept:     The nutritional concept to teach.
        context:     Retrieved knowledge chunks.
        description: Optional user-provided focus or audience guidance.
    """
    context_block = _build_context_block(context)
    user_prompt   = _build_text_lesson_prompt(concept, context_block, description)

    logger.info(
        f"[Generator] Generating text lesson for: '{concept}'"
        + (f" | description: '{description[:60]}...'" if description else "")
    )

    try:
        # COMPLEX tier: long-form, coherent ~20 minute lesson.
        raw = complete(
            Tier.COMPLEX,
            system=TEXT_LESSON_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=TEXT_LESSON_MAX_TOKENS,
            temperature=0.4,      # narrative fluency; ignored by Anthropic models
            json_mode=True,
        )
        lesson_data = parse_json_object(raw)

        if "sources" not in lesson_data:
            lesson_data["sources"] = list({item.get("source", "Unknown") for item in context})

        if "quiz_hints" not in lesson_data:
            lesson_data["quiz_hints"] = []

        logger.info(
            f"[Generator] Text lesson generated "
            f"({len(lesson_data.get('lesson_text', ''))} chars, "
            f"{len(lesson_data.get('quiz_hints', []))} quiz hints)"
        )
        return lesson_data

    except json.JSONDecodeError as e:
        logger.error(f"[Generator] Failed to parse text lesson JSON: {e}", exc_info=True)
        return _fallback_text_lesson(concept)

    except Exception as e:
        logger.error(f"[Generator] LLM error on text lesson: {e}", exc_info=True)
        return _fallback_text_lesson(concept)


def _fallback_text_lesson(concept: str) -> dict:
    return {
        "concept": concept,
        "lesson_text": (
            f"I am sorry, a lesson on {concept} could not be generated right now. "
            "Please ensure the topic is nutrition-related and try again."
        ),
        "quiz_hints": [],
        "sources": [],
    }