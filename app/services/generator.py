import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Shared context builder (used by both query and lesson) ────────────────────

def _build_context_block(context: list) -> str:
    if not context:
        return "No verified context was retrieved for this query."
    lines = []
    for i, item in enumerate(context, start=1):
        source = item.get("source", "Unknown source")
        authority = item.get("authority", "Unknown authority")
        text = item.get("text", "").strip()
        lines.append(f"[{i}] Source: {source} | Authority: {authority}\n{text}")
    return "\n\n".join(lines)

def _sanitize_beats(beats: list) -> list:
    """Ensure every beat has a valid, non-empty tablet object."""
    for beat in beats:
        tablet = beat.get("tablet")
        if not tablet or not isinstance(tablet, dict) or "display_type" not in tablet:
            beat["tablet"] = {
                "display_type": "key_fact",
                "heading": "Information",
                "body": str(beat.get("speech", ""))[:120],
            }
    return beats

# ── /query generation ─────────────────────────────────────────────────────────

QUERY_SYSTEM_PROMPT = """You are NutriPepper, a friendly and knowledgeable nutrition education \
assistant for the Pepper humanoid robot deployed in Kenyan communities, schools, and health centres.

Your role is to educate users about food safety, nutrition, and sustainable food practices. \
You are NOT a doctor or dietitian and must never provide medical diagnoses, prescribe treatments, \
or give personalised clinical advice.

STRICT GROUNDING RULES:
- Answer ONLY using information from the VERIFIED CONTEXT provided below.
- If the context does not contain enough information to answer the question, say so clearly — \
do not guess or use general knowledge.
- Every factual claim you make must be traceable to the provided context.
- Never fabricate nutritional values, health claims, or safety guidelines.

RESPONSE STYLE:
- Be warm, clear, and educational — suitable for children and adults alike.
- Use Kenyan food examples where present in the context (ugali, sukuma wiki, nyama choma, etc.).
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
    user_prompt = _build_query_user_prompt(query, context_block)

    logger.info(f"[Generator] Sending query to LLM: '{query}'")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
            top_p=0.9,
        )
        generated_text = response.choices[0].message.content.strip()
        logger.info(f"[Generator] Query response generated ({len(generated_text)} chars)")
        return generated_text

    except Exception as e:
        logger.error(f"[Generator] LLM error on query: {e}", exc_info=True)
        return (
            "I am sorry, I was not able to generate a response right now. "
            "Please try again in a moment."
        )


# ── /learn generation ─────────────────────────────────────────────────────────

LESSON_SYSTEM_PROMPT = """You are NutriPepper, a nutrition education assistant for the Pepper \
humanoid robot deployed in Kenyan communities, schools, and health centres.

You generate structured teaching lessons based STRICTLY on the verified nutritional context provided.

OUTPUT RULES — CRITICAL:
- You must return ONLY a valid JSON object. No markdown, no code fences, no explanation outside the JSON.
- Do not include any text before or after the JSON object.
- All values must be plain text strings — no markdown, no asterisks, no bullet symbols, no newlines inside strings.
- The "speech" field must be robot-speakable: plain sentences only, no symbols, no lists.

GROUNDING RULES:
- Every factual claim in every beat must come from the provided VERIFIED CONTEXT.
- Do not invent nutritional values, health statistics, or food safety rules.
- If the context is insufficient for a beat type, simplify that beat rather than fabricating content.

KENYAN CONTEXT:
- Prioritise Kenyan food examples: ugali, sukuma wiki, omena, terere, managu, githeri, uji, nyama choma.
- Reference Kenya Ministry of Health, FAO, and WHO guidelines where present in the context.
- Frame advice to be practical and accessible for Kenyan communities.

LESSON STRUCTURE:
Generate exactly 4 or 5 teaching beats. Use this sequence as a guide:
  Beat 1: introduction — introduce the concept clearly and simply
  Beat 2: kenyan_context — relate the concept to Kenyan foods and local practices
  Beat 3: key_facts — share 2 to 3 important facts grounded in the context
  Beat 4: safety_tip OR practical_advice — one actionable tip the user can apply today
  Beat 5 (optional): quiz — a single multiple-choice question to check understanding

BEAT SCHEMA — each beat must follow this exact structure:
{
  "beat_number": <integer>,
  "type": <one of: "introduction", "kenyan_context", "key_facts", "safety_tip", "practical_advice", "quiz">,
  "speech": <string — what Pepper says aloud, plain sentences only>,
  "gesture": <one of: "explaining", "pointing_tablet", "tip_gesture", "warning_gesture", "thinking", "welcoming">,
  "tablet": <one of the tablet display objects below — MUST be fully populated, never empty {}>
}

"CRITICAL: Every beat MUST include a fully populated tablet object. "
"An empty tablet object {} is NEVER acceptable. "
"If unsure which display type to use, default to key_fact with a heading and body.",

TABLET DISPLAY OBJECTS — use the correct one for each beat type:

key_fact:  { "display_type": "key_fact", "heading": <string>, "body": <string> }
food_list: { "display_type": "food_list", "heading": <string>, "items": [<string>, ...] }
tip:       { "display_type": "tip", "heading": <string>, "body": <string> }
warning:   { "display_type": "warning", "heading": <string>, "body": <string> }
quiz:      { "display_type": "quiz", "question": <string>, "options": [<string>, <string>, <string>, <string>], "correct_answer": <string> }

FULL OUTPUT SCHEMA:
{
  "concept": <string — the concept as entered>,
  "total_beats": <integer — number of beats generated>,
  "beats": [ <beat objects> ],
  "sources": [ <list of source document names used> ]
}
"""


def _build_lesson_user_prompt(concept: str, context_block: str) -> str:
    return f"""VERIFIED CONTEXT:
{context_block}

CONCEPT TO TEACH:
{concept}

Generate a structured 4 to 5 beat lesson on this concept using ONLY the verified context above. \
Return valid JSON only."""


def generate_lesson(concept: str, context: list) -> dict:
    """
    Generate a structured teaching lesson for Pepper based on a nutritional concept.

    Args:
        concept: The nutritional concept entered via the tablet.
        context: Retrieved knowledge chunks relevant to the concept.

    Returns:
        A dict matching the LearnData schema, or a safe fallback dict on failure.
    """
    context_block = _build_context_block(context)
    user_prompt = _build_lesson_user_prompt(concept, context_block)

    logger.info(f"[Generator] Generating lesson for concept: '{concept}'")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": LESSON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,       # Slightly higher than query — allows natural lesson phrasing
            max_tokens=1500,       # Lessons are longer than single query responses
            top_p=0.9,
            response_format={"type": "json_object"},  # Force JSON mode — OpenAI guarantees valid JSON
        )

        raw = response.choices[0].message.content.strip()
        logger.info(f"[Generator] Lesson JSON received ({len(raw)} chars)")

        lesson_data = json.loads(raw)

        # Sanitize any empty tablet objects the LLM may have produced
        if "beats" in lesson_data:
            lesson_data["beats"] = _sanitize_beats(lesson_data["beats"])

        if "sources" not in lesson_data:
            lesson_data["sources"] = list({
                item.get("source", "Unknown") for item in context
            })

        logger.info(f"[Generator] Lesson parsed: {lesson_data.get('total_beats')} beats")
        return lesson_data

    except json.JSONDecodeError as e:
        logger.error(f"[Generator] Failed to parse lesson JSON: {e}", exc_info=True)
        return _fallback_lesson(concept)

    except Exception as e:
        logger.error(f"[Generator] LLM error on lesson: {e}", exc_info=True)
        return _fallback_lesson(concept)


def _fallback_lesson(concept: str) -> dict:
    """Returns a minimal safe lesson when generation fails."""
    return {
        "concept": concept,
        "total_beats": 1,
        "beats": [
            {
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
            }
        ],
        "sources": [],
    }