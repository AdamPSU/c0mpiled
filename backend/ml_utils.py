import asyncio
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Dict

class HybridSearcher:
    def __init__(self, model_name: str = "allenai/specter2_base"):
        # Configure global settings for LlamaIndex
        Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
        # We don't need a default LLM for this task as we are just retrieving
        Settings.llm = None 

    async def hybrid_rerank(self, query: str, papers: List[Dict]) -> List[Dict]:
        """
        Reranks a list of papers using LlamaIndex QueryFusionRetriever with RRF.
        """
        # 1. Standardize Deduplication
        seen_ids = set()
        unique_papers = []
        for p in papers:
            if p['paperId'] not in seen_ids:
                seen_ids.add(p['paperId'])
                unique_papers.append(p)
        
        # Filter for papers with abstracts
        valid_papers = [p for p in unique_papers if p.get('abstract')]
        other_papers = [p for p in unique_papers if not p.get('abstract')]
        
        if not valid_papers:
            for p in other_papers:
                p['semantic_similarity'] = 0.0
            return other_papers

        # 2. Convert to LlamaIndex Documents
        # We store the full paper dict in metadata for easy retrieval
        documents = [
            Document(
                text=p['abstract'], 
                metadata={"paperId": p['paperId'], "original_data": p}
            ) for p in valid_papers
        ]

        # 3. Setup Index & Retrievers
        # In-memory index for the current batch
        index = VectorStoreIndex.from_documents(documents)
        
        vector_retriever = index.as_retriever(similarity_top_k=len(documents))
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=index.docstore, 
            similarity_top_k=len(documents)
        )

        # 4. Fusion Magic (RRF)
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            num_queries=1,
            mode="reciprocal_rerank",
            similarity_top_k=len(documents),
            use_async=True
        )

        # 5. Execute search
        nodes = await fusion_retriever.aretrieve(query)
        
        # 6. Reconstruct final list with fused scores
        fused_results = []
        for node in nodes:
            paper_data = node.metadata["original_data"].copy()
            paper_data['semantic_similarity'] = node.score
            fused_results.append(paper_data)

        # Add papers that didn't have abstracts with 0 score
        for p in other_papers:
            p_copy = p.copy()
            p_copy['semantic_similarity'] = 0.0
            fused_results.append(p_copy)
            
        return fused_results
