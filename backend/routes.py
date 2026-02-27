import os
from fastapi import APIRouter, HTTPException, Query
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter()

SH_API_KEY = os.getenv("SH_API_KEY")
SEMANTIC_SCHOLAR_BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

@router.get("/search")
async def search_papers(query: str = Query(..., description="The search query for papers")):
    """
    Search for papers using the Semantic Scholar Bulk Search API.
    Results are sorted by citation count in descending order.
    """
    headers = {}
    if SH_API_KEY:
        headers["x-api-key"] = SH_API_KEY

    params = {
        "query": query,
        "fields": "paperId,title,year,citationCount"
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                SEMANTIC_SCHOLAR_BULK_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract paper data
            papers = data.get("data", [])
            
            # Sort by citation count in descending order
            # citationCount can be None, so handle that case
            sorted_papers = sorted(
                papers,
                key=lambda x: x.get("citationCount") if x.get("citationCount") is not None else 0,
                reverse=True
            )
            
            return {
                "total": len(sorted_papers),
                "query": query,
                "papers": sorted_papers
            }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Semantic Scholar API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
