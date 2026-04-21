from typing import List, TypedDict


class AgentState(TypedDict):
    query: str
    query_embedding: List[float]
    selected_tag: str
    candidate_paths: List[List[str]]
    selected_paths: List[List[str]]
    retrieved_chunks: List[dict]
    answer: str