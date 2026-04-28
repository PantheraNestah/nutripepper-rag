from fastapi.responses import JSONResponse

def internal_server_error():
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal Server Error"
        }
    )