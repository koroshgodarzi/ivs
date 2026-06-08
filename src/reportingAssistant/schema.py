"""Pydantic models for API and LangGraph state definitions."""

from typing import TypedDict, List, Optional, Annotated
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    """State managed by the LangGraph workflow."""
    messages: Annotated[List[dict], "Chat history messages"]
    generated_query: Optional[List[List[str]]] 
    query_results: Optional[str]  # Results from executing the query
    error_message: Optional[List[str]]
    summary_context: Optional[str]  # Summary from previous session (for reset feature)
    retry_count: int  # Track number of retries to limit loops
    validation_result: Optional[str]  # Result from query validation
    retrieved_schema: Optional[str]
    query_explanation: Optional[str]
    retrieved_columns: Optional[str] # Added field to store schema metadata
    retrieved_values: Optional[str]
    keywords: str
    query_generation_user_prompt: str
    chat_history: str
    to_plot: bool
    viz_config: Optional[dict]