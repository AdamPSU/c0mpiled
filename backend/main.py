from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router as paper_router
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI(title="C0mpiled Paper Search API")

# Enable CORS for local development (standard for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.add_api_route("/", lambda: {"status": "ok", "message": "Welcome to C0mpiled Paper Search API"})
app.include_router(paper_router, prefix="/api", tags=["papers"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
