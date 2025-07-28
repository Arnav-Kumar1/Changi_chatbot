from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.rag_pipeline import rag_pipeline

router = APIRouter()

# --- Request Schema ---
class QARequest(BaseModel):
    user_query: str = Field(..., description="The user's input question.")
    api_key: str = Field(..., description="Google Gemini API key for this session.")

# --- RAG Endpoint ---
@router.post("/qa", summary="Query the RAG pipeline")
def query_rag(request: QARequest):
    """
    Accepts a user query and an API key, then returns an answer and sources using the RAG pipeline.
    """
    if not request.api_key.strip() or not request.user_query.strip():
        raise HTTPException(status_code=400, detail="Both 'user_query' and 'api_key' must be provided.")

    try:
        result = rag_pipeline(user_query=request.user_query, api_key=request.api_key)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {str(e)}")

# --- Healthcheck ---
@router.get("/ping", summary="Healthcheck")
def ping():
    return {"status": "ok"}
