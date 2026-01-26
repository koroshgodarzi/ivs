from graph.schema import GraphState


def format_final_response(state: GraphState) -> GraphState:
    """
    Surrogate for format_final_response that stores query and results directly
    without prompting the LLM.
    
    If query execution was successful: stores generated_query and query_results as content.
    If query execution was unsuccessful: stores error_message as content.
    """
    query = state.get("generated_query", "")
    results = state.get("query_results")
    error = state.get("error_message")
    
    if error:
        # Query execution was unsuccessful - store error_message
        response_text = error
    elif results is not None:
        # Query execution was successful - store generated_query and query_results
        response_text = results
    else:
        # Fallback case
        response_text = "I was unable to generate a valid SQL query."
    
    # Add assistant response to messages (same as original function at lines 246-249)
    state["messages"].append({
        "role": "assistant",
        "content": response_text
    })

    return state