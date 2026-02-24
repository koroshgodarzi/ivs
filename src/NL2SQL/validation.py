"""Query validation using LLM to check if a question can be answered from the database schema."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from graph.schema import GraphState
from utils import get_llm, extract_json_from_text, ommiting_think_block, count_chat_tokens
from typing import Literal

import os
import json


def validate_user_question(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Use LLM to determine if a question can be answered from the database schema.
    This function works as a LangGraph node and updates the state.
    
    Args:
        state: The GraphState containing messages with the user's question
        
    Returns:
        Updated GraphState with validation_result field
    """
    model_name = config.get("configurable", {}).get("model_name", "gpt")
    llm = get_llm(model_id=model_name)
    
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    if not user_messages:
        state["validation_result"] = "No user question found."
        return state
    
    schema = ""
    views = state.get("retrieved_schema")

    for v in views:
        schema += v + ':\n'
        with open(os.path.join('..', 'data', 'short_schema', f'{v}.txt')) as f:
            s = f.read()
        schema += s
        schema += '\n'
    
    with open(os.path.join('..', 'prompt_template', 'query_validation_system_prompt.txt')) as f:
        system_prompt = f.read()

    with open(os.path.join('..', 'prompt_template', 'query_validation_user_prompt.txt')) as f:
        user_prompt = f.read().format(schema, user_messages)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # print(messages)

    num_tokens_msg = count_chat_tokens(messages)
    print(num_tokens_msg)

    response = llm.invoke(messages)
    
    try:
        validation_result = response.content.strip()
        validation_result = ommiting_think_block(validation_result)
        validation_result = extract_json_from_text(validation_result)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse validation_result as JSON: {e}")

    state["validation_result"] = validation_result
    print(f"validation result is: {validation_result}")

    return state


def should_proceed_with_user_question(state: GraphState, config: RunnableConfig) -> Literal["proceed", "halt"]:
    """Conditional edge: Decide whether to proceed with SQL generation based on validation."""
    validation_result = state.get("validation_result", "")

    if not validation_result:
        return "proceed"

    if validation_result.get("short answer", "").strip().upper().startswith("YES"):
        return "proceed"
    else:
        return "halt"


def handle_validation_failure(state: GraphState, config: RunnableConfig) -> GraphState:
    """Handle case when validation fails - set query_results and add assistant message."""
    validation_result = state.get("validation_result", "Failed to validate user's question.")
    
    if isinstance(validation_result, dict):
        validation_message = validation_result.get("instructions", "The question cannot be answered with the available database schema.")
    else:
        validation_message = str(validation_result)

    state["query_results"] = [{"validation_message": validation_message}]
    
    state["messages"].append({
        "role": "assistant",
        "content": validation_message
    })
    
    return state


if __name__ == "__main__":
    # with open(os.path.join('..', 'data', 'short_schema', 'vw_Contract.txt')) as f:
    #     r_schema = f.read()
    state = GraphState(messages=[{"role": "user", "content": "Status of how many contracts are canceled?"}], generated_query=None, query_results=None, error_message=None, summary_context=None, retry_count=0, validation_result=None, retrieved_schema=['vw_Contracts', 'vw_Projects'])
    state = validate_user_question(state)
    print(state)

