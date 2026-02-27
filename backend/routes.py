import os
import math
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
import httpx
from dotenv import load_dotenv

from backend.ai_utils import AncestryTreeGenerator

# Load environment variables
load_dotenv()

router = APIRouter()

SH_API_KEY = os.getenv("SH_API_KEY")
SEMANTIC_SCHOLAR_BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Initialize AI components once
tree_generator = AncestryTreeGenerator()

def calculate_hybrid_score(paper: dict) -> float:
    """
    Computes a hybrid TreeScorer as a weighted combination of raw values.
    
    Formula:
    Score = (0.5 * Historical) + (0.5 * Momentum)
    
    - Historical: log10(Total Citations + 1)
    - Momentum: Total Citations / (Current Year - Published Year + 1)
    """
    total_citations = paper.get("citationCount", 0) or 0
    published_year = paper.get("year")
    current_year = datetime.now().year
    
    if published_year is None:
        return 0.0
        
    # 1. Historical Weight (Raw log10)
    historical_raw = math.log10(total_citations + 1)
    
    # 2. Momentum Weight (Raw citations/year)
    years_active = (current_year - published_year) + 1
    momentum_raw = total_citations / years_active
    
    # TreeScorer Formula (Balanced Linear Combination)
    tree_scorer = (0.5 * historical_raw) + (0.5 * momentum_raw)
    
    return tree_scorer

def filter_papers(papers: list) -> list:
    """
    Calculates hybrid score (TreeScorer) for all papers and returns the top 30 
    highest scoring papers across all years.
    Handles deduplication based on paperId.
    """
    seen_ids = set()
    unique_papers = []
    
    for paper in papers:
        paper_id = paper.get("paperId")
        if paper_id and paper_id not in seen_ids:
            seen_ids.add(paper_id)
            # Calculate hybrid score
            paper["hybrid_score"] = calculate_hybrid_score(paper)
            unique_papers.append(paper)
        
    # Sort all papers by hybrid score descending
    sorted_papers = sorted(
        unique_papers,
        key=lambda x: x.get("hybrid_score", 0.0),
        reverse=True
    )
    
    # Return the 30 most relevant papers
    return sorted_papers[:30]

@router.get("/search")
async def search_papers(
    query: str = Query(..., description="The search query for papers")
):
    """
    Search for papers using the Semantic Scholar Bulk Search API.
    The top 30 papers by TreeScorer (Historical + Momentum) are returned and grouped by year.
    """

    headers = {}
    if SH_API_KEY:
        headers["x-api-key"] = SH_API_KEY

    params = {
        "query": query,
        "fields": "paperId,title,year,citationCount,openAccessPdf,abstract",
        "minCitationCount": 50
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
            
            # Step 1: Score with TreeScorer and pick top 30
            # This also handles deduplication
            top_30_papers = filter_papers(papers)
            
            # Step 3: Group and format by year for discretization
            final_data = {}
            for paper in top_30_papers:
                year_val = paper.get("year", "Unknown")
                year_key = str(year_val)
                
                if year_key not in final_data:
                    final_data[year_key] = []
                
                # Format to the new schema: {year, name, link, abstract, score}
                formatted_paper = {
                    "year": year_val,
                    "name": paper.get("title"),
                    "link": (paper.get("openAccessPdf") or {}).get("url") if paper.get("openAccessPdf") else None,
                    "abstract": paper.get("abstract"),
                    "score": paper.get("hybrid_score")
                }
                final_data[year_key].append(formatted_paper)
            
            # Sort the final dictionary by year descending
            sorted_years = sorted(final_data.keys(), reverse=True)
            sorted_final_data = {year: final_data[year] for year in sorted_years}
            
            # Step 4: Call AI to generate the ancestry tree
            # This uses the system prompt from prompts/system_prompt.txt
            ai_tree_result = await tree_generator.generate_tree(query, sorted_final_data)
            
            return {
                "query": query,
                "total_papers_retrieved": len(top_30_papers),
                "ancestry_tree_reasoning": ai_tree_result.get("reasoning"),
                "ancestry_tree": ai_tree_result.get("output"),
                "raw_papers_by_year": sorted_final_data
            }

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Semantic Scholar API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
