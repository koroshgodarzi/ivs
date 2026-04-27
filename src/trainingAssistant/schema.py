from typing import List, TypedDict


class AgentState(TypedDict):
    query: str
    query_embedding: List[float]
    selected_sources: str
    candidate_paths: List[List[str]]
    selected_paths: List[List[str]]
    retrieved_chunks: List[dict]
    reranked_chunks: List[dict]
    answer: str