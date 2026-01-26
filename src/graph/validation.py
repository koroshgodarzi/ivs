"""Query validation using LLM to check if a question can be answered from the database schema."""

from langchain_core.messages import HumanMessage, SystemMessage

from graph.schema import GraphState
from graph.utils import get_llm, extract_json_from_text
from typing import Literal

import os
import json


def validate_query(state: GraphState) -> GraphState:
    """
    Use LLM to determine if a question can be answered from the database schema.
    This function works as a LangGraph node and updates the state.
    
    Args:
        state: The GraphState containing messages with the user's question
        
    Returns:
        Updated GraphState with validation_result field
    """

    # Get the LLM instance
    llm = get_llm()
    
    # Extract the user's question from the last user message
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    if not user_messages:
        state["validation_result"] = "No user question found."
        return state
    
    # Get user prompt for schema retrieval
    # user_prompt = user_messages[-1]["content"]
    
    # Read the database schema using embeddings based on user prompt
    schema = ""
    for view in state.get("retrieved_schema"):
        with open(os.path.join('..', 'data', 'short_schema', f'{view}.txt')) as f:
            s = f.read()
        schema += s

    
    with open(os.path.join('..', 'prompt_template', 'query_validation_system_prompt.txt')) as f:
        system_prompt = f.read()

    with open(os.path.join('..', 'prompt_template', 'query_validation_user_prompt.txt')) as f:
        user_prompt = f.read().format(schema, user_messages)
    
    # Build messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    num_tokens_msg = llm.get_num_tokens_from_messages(messages)
    print(f"Tokens in validation messages: {num_tokens_msg}")
    
    # Invoke the LLM
    response = llm.invoke(messages)
    
    # Store the validation result in state
    validation_result = response.content.strip()
    state["validation_result"] = validation_result
    
    return state


def should_proceed_with_query(state: GraphState) -> Literal["proceed", "halt"]:
    """Conditional edge: Decide whether to proceed with SQL generation based on validation."""
    validation_result = state.get("validation_result", "")
    
    if not validation_result:
        # If no validation result, proceed (shouldn't happen, but safe fallback)
        return "proceed"
    
    # Check if the response starts with "Yes" (case-insensitive)
    # validation_result_upper = validation_result.strip().upper()
    
    # Extract JSON from validation_result if it's a string
    if isinstance(validation_result, str):
        try:
            validation_result = extract_json_from_text(validation_result)
            # Update state with parsed JSON
            state["validation_result"] = validation_result
        except (json.JSONDecodeError, ValueError) as e:
            # If parsing fails, treat as invalid and halt
            print(f"Failed to parse validation_result as JSON: {e}")
            return "halt"

    if validation_result.get("short answer", "").strip().upper().startswith("YES"):
        return "proceed"
    else:
        return "halt"


def handle_validation_failure(state: GraphState) -> GraphState:
    """Handle case when validation fails - set query_results and add assistant message."""
    validation_result = state.get("validation_result", "The question cannot be answered with the available database schema.")
    
    # Extract message from validation_result if it's a dict
    if isinstance(validation_result, dict):
        validation_message = validation_result.get("instructions", "The question cannot be answered with the available database schema.")
    elif isinstance(validation_result, str):
        # Try to parse if it's a string with JSON
        try:
            parsed = extract_json_from_text(validation_result)
            validation_message = parsed.get("instructions", validation_result)
        except (json.JSONDecodeError, ValueError):
            validation_message = validation_result
    else:
        validation_message = str(validation_result)
    
    # Store validation result in query_results as a dict
    state["query_results"] = [{"validation_message": validation_message}]
    
    # Add assistant response to messages
    state["messages"].append({
        "role": "assistant",
        "content": validation_message
    })
    
    return state


if __name__ == "__main__":
    # with open(os.path.join('..', 'data', 'short_schema', 'vw_Contract.txt')) as f:
    #     r_schema = f.read()
    state = GraphState(messages=[{"role": "user", "content": "Status of how many contracts are canceled?"}], generated_query=None, query_results=None, error_message=None, summary_context=None, retry_count=0, validation_result=None, retrieved_schema=['vw_Contracts'])
    state = validate_query(state)
    print(state)

