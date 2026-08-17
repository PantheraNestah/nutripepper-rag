# app/schemas/query.py

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    user_query: str = Field(..., min_length=1, description="The user's nutrition question")


class LearnRequest(BaseModel):
    """Used by /learn/structured — concept only."""
    concept: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="The nutritional concept to generate a lesson for",
        examples=["iron deficiency in children", "food safety during pregnancy"],
    )


class TextLearnRequest(BaseModel):
    """
    Used by /learn/text — concept plus an optional description.
    The description lets the user or facilitator steer the lesson
    toward a specific audience, focus area, or local context.
    """
    concept: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="The nutritional concept to generate a lesson for",
        examples=["iron deficiency in children", "food safety during pregnancy"],
    )
    description: str | None = Field(
        default=None,
        min_length=5,
        max_length=1000,
        description="Optional brief description or focus to guide lesson generation",
        examples=[
            "Focus on school-age children aged 6 to 12 in rural Kenya",
            "Emphasise practical cooking tips and locally available foods",
            "Target caregivers at a community health centre",
        ],
    )