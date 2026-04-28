from fastapi import APIRouter, Depends
from app.schemas.query import QueryRequest
from app.schemas.response import StandardResponse
from app.dependencies.pipeline import get_pipeline_service
from app.core.exceptions import internal_server_error

router = APIRouter()

@router.post("/query", response_model=StandardResponse)
def process_query(
    request: QueryRequest,
    pipeline_service=Depends(get_pipeline_service)
):
    try:
        result = pipeline_service(request.user_query)

        return {
            "status": "success",
            "message": "Query processed successfully",
            "data": result
        }

    except Exception:
        return internal_server_error()