from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import document_router, translation_router, ai_router

app = FastAPI(
    title="AI-Powered Handwriting Recognition & Translation API",
    version="2.0.0",
    description="Advanced AI-driven document processing with NLP, LLM, and GenAI capabilities"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document_router.router, prefix="/api/documents", tags=["documents"])
app.include_router(translation_router.router, prefix="/api/translation", tags=["translation"])
app.include_router(ai_router.router, prefix="/api/ai", tags=["ai-features"])

@app.get("/")
async def root():
    return {
        "message": "AI-Powered Handwriting Recognition & Translation API is running",
        "version": "2.0.0",
        "features": [
            "Advanced OCR with AI enhancement",
            "Context-aware translation",
            "NLP text analysis",
            "Document intelligence",
            "Sentiment analysis",
            "Entity extraction",
            "Text summarization",
            "Quality assessment"
        ],
        "endpoints": {
            "documents": "/api/documents",
            "translation": "/api/translation",
            "ai_features": "/api/ai",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)