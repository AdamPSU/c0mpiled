import os
from fastapi import APIRouter, HTTPException, Query
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter()

SH_API_KEY = os.getenv("SH_API_KEY")
SEMANTIC_SCHOLAR_BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

def filter_papers(papers: list) -> dict:
    """
    Groups papers by year, sorts each year by citation count descending,
    and keeps only the top 5 papers per year.
    """
    grouped_papers = {}
    
    for paper in papers:
        year = paper.get("year")
        if year is None:
            continue
            
        if year not in grouped_papers:
            grouped_papers[year] = []
        grouped_papers[year].append(paper)
        
    filtered_data = {}
    for year, year_papers in grouped_papers.items():
        # Sort by citation count descending
        sorted_year_papers = sorted(
            year_papers,
            key=lambda x: x.get("citationCount") if x.get("citationCount") is not None else 0,
            reverse=True
        )
        # Keep top 5
        filtered_data[year] = sorted_year_papers[:5]
        
    return filtered_data

@router.get("/search")
async def search_papers(
    query: str = Query(..., description="The search query for papers"),
    min_citation_count: int = Query(50, alias="minCitationCount", description="Minimum number of citations required")
):
    """
    Search for papers using the Semantic Scholar Bulk Search API.
    Results are sorted by citation count in descending order.
    """
    
    headers = {}
    if SH_API_KEY:
        headers["x-api-key"] = SH_API_KEY

    params = {
        "query": query,
        "fields": "paperId,title,year,citationCount",
        "minCitationCount": min_citation_count
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                SEMANTIC_SCHOLAR_BULK_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=15.0
            )
            response.raise_for_status()
            data = response.json()
            
            papers = data.get("data", [])
            
            # Group and filter papers by year (top 5 per year)
            filtered_papers = filter_papers(papers)
            
            # Sort the final dictionary by year descending
            sorted_years = sorted(filtered_papers.keys(), reverse=True)
            final_data = {str(year): filtered_papers[year] for year in sorted_years}
            
            return {
                "total_unique_years": len(final_data),
                "query": query,
                "papers_by_year": final_data
            }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Semantic Scholar API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
