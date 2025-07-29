from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import qa


app = FastAPI(
    title="Changi RAG Chatbot",
    description="Ask questions based on Changi & Jewel website content.",
    version="1.0.0"
)

# CORS (for local frontend dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register your route(s)
app.include_router(qa.router, prefix="/api")
