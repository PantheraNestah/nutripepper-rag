from pydantic import BaseModel
from typing import List, Any

class QueryPayload(BaseModel):
    original_query: str
    processed_query: str
    retrieved_context: List[str]
    generated_response: str

class StandardResponse(BaseModel):
    status: str
    message: str
    data: Any