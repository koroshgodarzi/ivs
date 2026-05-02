from typing import List, TypedDict, Optional

class AgentState(TypedDict):
    query: str
    query_embedding: List[float]
    selected_sources: List[str] 
    candidate_paths: List[str]
    selected_paths: List[str]
    retrieved_chunks: List[dict]
    reranked_chunks: List[dict]
    answer: str # This will be the cumulative answer
    steps_taken: int
    low_confidence: bool
    is_finished: bool # New: To track if LLM hit stop sequence
    last_generated_segment: str
    # Tracking the best attempt for fallback
    best_answer: str
    best_is_finished: bool
    best_min_logprob: float