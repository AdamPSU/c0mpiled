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
    Computes a hybrid TreeScorer as a weighted combination of raw values.
    
    Formula:
    Score = (0.4 * Historical) + (0.4 * Momentum) + (0.2 * Semantic)
    
    - Historical: log10(Total Citations + 1)
    - Momentum: Total Citations / (Current Year - Published Year + 1)
    - Semantic: Raw RRF score from hybrid search
    """
    total_citations = paper.get("citationCount", 0) or 0
    published_year = paper.get("year")
    current_year = datetime.now().year
    
    # Semantic Similarity is now part of the paper metadata after reranking
    semantic_similarity = paper.get("semantic_similarity", 0.0)
    
    if published_year is None:
        return 0.0
        
    # 1. Historical Weight (Raw log10)
    historical_raw = math.log10(total_citations + 1)
    
    # 2. Momentum Weight (Raw citations/year)
    years_active = (current_year - published_year) + 1
    momentum_raw = total_citations / years_active
    
    # 3. Semantic Similarity (Raw RRF score)
    semantic_raw = semantic_similarity
    
    # TreeScorer Formula (Weighted Linear Combination)
    tree_scorer = (0.4 * historical_raw) + (0.4 * momentum_raw) + (0.2 * semantic_raw)
    
    return tree_scorer

def filter_papers(papers: list) -> list:
    """
    Calculates hybrid score (TreeScorer) for all papers and returns the top 30 
    highest scoring papers across all years.
    """
    for paper in papers:
        # Calculate hybrid score
        paper["hybrid_score"] = calculate_hybrid_score(paper)
        
    # Sort all papers by hybrid score descending
    sorted_papers = sorted(
        papers,
        key=lambda x: x.get("hybrid_score", 0.0),
        reverse=True
    )
    
    # Return the 30 most relevant papers
    return sorted_papers[:30]

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
            
            # Step 2: Score with TreeScorer and pick top 30
            top_30_papers = filter_papers(reranked_papers)
            
            # Step 3: Group by year for discretization
            final_data = {}
            for paper in top_30_papers:
                year = str(paper.get("year", "Unknown"))
                if year not in final_data:
                    final_data[year] = []
                final_data[year].append(paper)
            
            # Sort the final dictionary by year descending
            sorted_years = sorted(final_data.keys(), reverse=True)
            sorted_final_data = {year: final_data[year] for year in sorted_years}
            
            return {
                "total_papers": len(top_30_papers),
                "query": query,
                "papers_by_year": sorted_final_data
            }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Semantic Scholar API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
