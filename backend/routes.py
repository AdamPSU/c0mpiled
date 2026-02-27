import os
import math
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
import httpx
from dotenv import load_dotenv

from backend.ml_utils import HybridSearcher

# Load environment variables
load_dotenv()

router = APIRouter()

SH_API_KEY = os.getenv("SH_API_KEY")
SEMANTIC_SCHOLAR_BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Initialize ML components once
hybrid_searcher = HybridSearcher()

def calculate_hybrid_score(paper: dict) -> float:
    """
    Computes a hybrid score (Historical * Momentum * Semantic Similarity) for a paper.
    TreeScorer = (Historical Weight) * (Momentum Weight) * (Semantic Similarity)
    """
    
    total_citations = paper.get("citationCount", 0) or 0
    published_year = paper.get("year")
    current_year = datetime.now().year
    
    # Semantic Similarity is now part of the paper metadata after reranking
    semantic_similarity = paper.get("semantic_similarity", 0.0)
    
    if published_year is None:
        return 0.0
        
    # Historical Weight = log_10(Total Citations + 1)
    historical_weight = math.log10(total_citations + 1)
    
    # Momentum Weight = (Total Citations) / (Current Year - Published Year + 1)
    # Adding 1 to the denominator to prevent division by zero for current year papers
    momentum_weight = total_citations / (current_year - published_year + 1)
    
    # TreeScorer Formula
    return historical_weight * momentum_weight * semantic_similarity

def filter_papers(papers: list) -> dict:
    """
    Groups papers by year, sorts each year by hybrid score (TreeScorer) descending,
    and keeps only the top 5 papers per year.
    """
    grouped_papers = {}
    
    for paper in papers:
        year = paper.get("year")
        if year is None:
            continue
            
        if year not in grouped_papers:
            grouped_papers[year] = []
        
        # Calculate hybrid score before adding to group
        paper["hybrid_score"] = calculate_hybrid_score(paper)
        grouped_papers[year].append(paper)
        
    filtered_data = {}
    for year, year_papers in grouped_papers.items():
        # Sort by hybrid score (TreeScorer) descending
        sorted_year_papers = sorted(
            year_papers,
            key=lambda x: x.get("hybrid_score", 0.0),
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
        "fields": "paperId,title,year,citationCount,openAccessPdf,abstract",
        "minCitationCount": min_citation_count
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
            
            papers = data.get("data", [])
            
            # Step 1: Semantic Reranking with RRF (BM25 + Bi-Encoder)
            # This handles deduplication and adds 'semantic_similarity' to papers with abstracts
            reranked_papers = await hybrid_searcher.hybrid_rerank(query, papers)
            
            # Step 2: Group, Score with TreeScorer, and Filter by Year
            filtered_papers = filter_papers(reranked_papers)
            
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
