from fastapi import APIRouter, Depends
from app.schemas.query import QueryRequest, LearnRequest
from app.schemas.response import StandardResponse, QueryResponse, LearnResponse
from app.dependencies.pipeline import get_pipeline_service
from app.core.exceptions import internal_server_error

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    pipeline_service=Depends(get_pipeline_service)
):
    try:
        result = await pipeline_service.run_query(request.user_query)

        return {
            "status": "success",
            "message": "Query processed successfully",
            "data": result
        }

    except Exception:
        return internal_server_error()
    
@router.post("/learn")
async def learn(request: LearnRequest, pipeline=Depends(get_pipeline_service)):
    result = await pipeline.run_lesson(request.concept)
    return {
        "status": "success",
        "message": "Lesson generated successfully",
        "data": result,
    }
