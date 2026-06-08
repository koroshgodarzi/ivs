from reportingAssistant.schema import GraphState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils import get_llm, create_column_names_for_schemas, extract_json_from_text, ommiting_think_block
from typing import Literal
from langchain_core.runnables import RunnableConfig

import json
import os


def explain_query_error(state: GraphState, config: RunnableConfig) -> GraphState:
    """Use LLM to explain the reason for a query execution error."""
    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name", "gpt")
    llm = get_llm(model_id=model_name)

    prompt_dir = configurable.get("prompt_template_dir", os.path.join("..", "prompt_template"))

    with open(os.path.join(prompt_dir, 'error_handling_system_prompt.txt'), 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    all_turns = state.get("generated_query", [])
    current_turn_queries = all_turns[-1] if all_turns else []
    error_messages = state.get("error_message", [])

    user_prompt_text = state["query_generation_user_prompt"]

    # Use the current turn's queries and corresponding errors
    for i, (q, err) in enumerate(zip(current_turn_queries, error_messages)):
        with open(os.path.join(prompt_dir, 'error_handling_user_prompt_part_2.txt'), 'r', encoding='utf-8') as f:
            p = f.read().format(i, q, err)
        user_prompt_text += "\n" + p

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt_text)
    ]

    response = llm.invoke(messages)
    try:
        content = response.content
        
        if isinstance(content, str):
            content = content.strip()
        elif isinstance(content, list):
            # Join text fields from all content blocks in the list
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            content = ""
            
    except Exception as e:
        # Handle parsing errors here
        content = ""
        print(f"Error extracting keywords: {e}")
    response = ommiting_think_block(content)
    response = extract_json_from_text(response)

    explanation_list = state["query_explanation"]
    explanation_list.append(response["corrected_query_explanation"])
    state["query_explanation"] = explanation_list

    all_turns[-1].append(response["corrected_query"])
    state["generated_query"] = all_turns
    
    return response


def should_retry(state: GraphState, config: RunnableConfig) -> Literal["retry", "end"]:
    error = state.get("error_message")
    retry_count = state.get("retry_count", 0)
    
    if error and retry_count < 3:
        return "retry"
    
    return "end"


def error_handler(state: GraphState, config: RunnableConfig) -> GraphState:
    error = state.get("error_message")
    if error:
        retry_count = state.get("retry_count", 0)
        state["retry_count"] = retry_count + 1
    return state