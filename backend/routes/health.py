from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_google_genai import ChatGoogleGenerativeAI
import os

router = APIRouter()

class HealthCheckPayload(BaseModel):
    api_key: str | None = None

@router.post("/healthcheck")
async def smart_health_check(payload: HealthCheckPayload):
    """
    Validates Gemini key by performing a cheap dummy request.
    - If api_key is provided → use it
    - Else → use os.getenv fallback key
    """

    key_to_use = payload.api_key or os.getenv("GOOGLE_API_KEY")
    if not key_to_use:
        return JSONResponse(status_code=400, content={"status": "invalid_key", "message": "No API key provided."})

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=key_to_use,
            temperature=0.2
        )
        response = llm.invoke("Say hello")
        if "hello" in response.content.lower():
            return {"status": "ok"}
        else:
            return {"status": "unexpected_behavior", "message": "Gemini response was unexpected."}

    except Exception as e:
        if "quota" in str(e).lower():
            return JSONResponse(status_code=429, content={"status": "quota_exceeded", "message": str(e)})
        elif "permission" in str(e).lower() or "invalid" in str(e).lower():
            return JSONResponse(status_code=403, content={"status": "invalid_key", "message": str(e)})
        else:
            return JSONResponse(status_code=500, content={"status": "backend_unavailable", "message": str(e)})
