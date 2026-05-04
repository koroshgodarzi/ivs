from typing import List, TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str  # The latest user message
    query_embedding: List[float]
    selected_sources: List[str] 
    retrieved_chunks: List[dict]
    reranked_chunks: List[dict]
    answer: str
    steps_taken: int
    low_confidence: bool
    is_finished: bool # New: To track if LLM hit stop sequence
    last_generated_segment: str
    # Tracking the best attempt for fallback
    best_answer: str
    best_is_finished: bool
    best_min_logprob: float