from pydantic import BaseModel, Field
from typing import Annotated, List, Any, Dict, Literal, Optional, Union

class QueryPayload(BaseModel):
    original_query: str
    processed_query: str
    retrieved_context: List[Dict]
    generated_response: str

class StandardResponse(BaseModel):
    status: str
    message: str
    data: Any

# ── /query response ───────────────────────────────────────────────────────────
 
class QueryData(BaseModel):
    original_query: str
    processed_query: str
    retrieved_context: list
    generated_response: str
 
 
class QueryResponse(BaseModel):
    status: str
    message: str
    data: QueryData
 
 
# ── /learn response — tablet display types ────────────────────────────────────
 
class TabletKeyFact(BaseModel):
    display_type: Literal["key_fact"]
    heading: str
    body: str
 
 
class TabletFoodList(BaseModel):
    display_type: Literal["food_list"]
    heading: str
    items: List[str]
 
 
class TabletTip(BaseModel):
    display_type: Literal["tip"]
    heading: str
    body: str
 
 
class TabletWarning(BaseModel):
    display_type: Literal["warning"]
    heading: str
    body: str
 
 
class TabletQuiz(BaseModel):
    display_type: Literal["quiz"]
    question: str
    options: List[str]
    correct_answer: str
 
 
TabletContent = Annotated[
    Union[TabletKeyFact, TabletFoodList, TabletTip, TabletWarning, TabletQuiz],
    Field(discriminator="display_type"),
]
 
 
# ── /learn response — beat structure ─────────────────────────────────────────
 
BEAT_TYPES = Literal[
    "introduction",
    "kenyan_context",
    "key_facts",
    "safety_tip",
    "practical_advice",
    "quiz",
]
 
GESTURE_TYPES = Literal[
    "explaining",
    "pointing_tablet",
    "tip_gesture",
    "warning_gesture",
    "thinking",
    "welcoming",
]
 
 
class TeachingBeat(BaseModel):
    beat_number: int
    type: BEAT_TYPES
    speech: str                  # What Pepper says aloud — no symbols or markdown
    gesture: GESTURE_TYPES       # Maps to a NAOqi animation tag
    tablet: TabletContent        # What the tablet displays during this beat
 
 
class LearnData(BaseModel):
    concept: str
    total_beats: int
    beats: List[TeachingBeat]
    sources: List[str]           # Source documents used to ground the lesson
 
 
class LearnResponse(BaseModel):
    status: str
    message: str
    data: LearnData
 
