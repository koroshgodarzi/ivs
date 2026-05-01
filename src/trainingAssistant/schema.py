from typing import List, TypedDict


class AgentState(TypedDict):
    query: str
    query_embedding: List[float]
    selected_sources: List[str] 
    candidate_paths: List[str]
    selected_paths: List[str]
    retrieved_chunks: List[dict]
    reranked_chunks: List[dict]
    answer: str
    steps_taken: int
    low_confidence: bool
    last_generated_segment: str