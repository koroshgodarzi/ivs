"""Query validation using LLM to check if a question can be answered from the database schema."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from NL2SQL.schema import GraphState
from utils import get_llm, extract_json_from_text, ommiting_think_block, count_chat_tokens
from typing import Literal

import os
import json


def validate_user_question(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Use LLM to determine if a question can be answered from the database schema.
    Iterates through 3, 5, and 7 table candidates until a 'Yes' is found.
    """
    model_name = config.get("configurable", {}).get("model_name", "gpt")
    llm = get_llm(model_id=model_name)
    
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    if not user_messages:
        state["validation_result"] = "No user question found."
        return state
    
    views = state.get("retrieved_schema", {})
    table_candidates = views.get('Table Candidates', [])
    
    # Define the increments for the loop
    limits = [3, 5, 7]
    validation_result = {}
    last_limit_tried = 0

    with open(os.path.join('..', 'prompt_template', 'query_validation_system_prompt.txt')) as f:
        system_prompt = f.read()
    with open(os.path.join('..', 'prompt_template', 'query_validation_user_prompt.txt')) as f:
        user_prompt_template = f.read()

    for limit in limits:
        # Optimization: If the total tables available are less than the previous limit, stop.
        # Or if we've already tried all available tables in the previous step.
        actual_tables_to_use = table_candidates[:limit]
        if len(actual_tables_to_use) <= last_limit_tried and last_limit_tried != 0:
            break
        
        last_limit_tried = len(actual_tables_to_use)
        print(f"Attempting validation with first {len(actual_tables_to_use)} tables...")

        # 1. Build schema for current subset
        schema = ""
        for v in actual_tables_to_use:
            schema += v + ':\n'
            try:
                with open(os.path.join('..', 'data', 'short_schema', f'{v}.txt')) as f:
                    s = f.read()
                schema += s + '\n'
            except FileNotFoundError:
                print(f"Warning: Schema file for {v} not found.")

        # 2. Prepare messages
        formatted_user_prompt = user_prompt_template.format(schema, user_messages)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=formatted_user_prompt)
        ]

        response = llm.invoke(messages)
        
        try:
            raw_content = response.content.strip()
            clean_content = ommiting_think_block(raw_content)
            parsed_json = extract_json_from_text(clean_content)
            validation_result = parsed_json
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse validation_result as JSON at limit {limit}: {e}")
            continue # Try next limit if parsing fails

        # 5. Check condition to break
        # We check for "Yes" (case-insensitive usually safer)
        short_answer = validation_result.get("short answer", "No")
        if str(short_answer).strip().upper().startswith("YES"):
            print(f"Validation successful with {limit} tables.")
            break
        else:
            print(f"Validation failed with {limit} tables. Trying next increment...")

    # Final state update
    state["validation_result"] = validation_result
    print(f"Final validation result: {validation_result}")

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

