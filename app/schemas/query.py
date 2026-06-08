from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    user_query: str = Field(..., min_length=1, description="The user's nutrition question")

class LearnRequest(BaseModel):
    concept: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="The nutritional concept to generate a lesson for",
        examples=["iron deficiency in children", "food safety during pregnancy"],
    )