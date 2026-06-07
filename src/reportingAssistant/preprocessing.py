from reportingAssistant.schema import GraphState
import os
from langchain_core.runnables import RunnableConfig
import json
from utils import get_llm, extract_json_from_text, ommiting_think_block
from pathlib import Path


def keyword_view_extraction(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Extracts keywords, attributes, and actual DB view names from the user's question.
    """
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    user_question = user_messages[-1]["content"] if user_messages else ""

    # Retrieve parameters from the config
    configurable = config.get("configurable", {})
    data_dir = configurable.get("data_dir", os.path.join("..", "data", "noisy_inclusive"))
    prompt_dir = configurable.get("prompt_template_dir", os.path.join("..", "prompt_template"))

    # Load view descriptions
    view_desc_path = os.path.join(data_dir, "view_descriptions.json")
    try:
        with open(view_desc_path, 'r', encoding='utf-8') as f:
            view_descriptions = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find view descriptions at {view_desc_path}")
        view_descriptions = {}

    view_descriptions_str = json.dumps(view_descriptions, indent=4, ensure_ascii=False)

    model_name = configurable.get("model_name", "gpt")
    llm = get_llm(model_name)

    # Load prompt template using global path
    prompt_template_path = os.path.join(prompt_dir, 'keyword_view_extraction_prompt.txt')
    try:
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template not found at {prompt_template_path}")
        return {"keywords": None}

    prompt = prompt.replace("[VIEW_DESCRIPTIONS_HERE]", view_descriptions_str)

    history = format_chat_history(state)
    prompt = prompt.replace("[HISTORY_HERE]", history)
    prompt = prompt.replace("[USER_QUESTION_HERE]", user_question)
    
    response = llm.invoke(prompt)
    
    keywords = None
    try:
        content = response.content
    
        if isinstance(content, str):
            keywords_text = content.strip()
        elif isinstance(content, list):
            # Join text fields from all content blocks in the list
            keywords_text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            keywords_text = ""
        keywords_text = ommiting_think_block(keywords_text)
        keywords = extract_json_from_text(keywords_text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse keywords as JSON: {e}")

    print(f"---Extracted Keywords---\n{keywords}")

    return {"keywords": keywords}


def format_chat_history(state: GraphState) -> str:
    history_str = ""
    messages = state.get("messages", [])
    queries = state.get("generated_query", [])
    
    # We match messages to query attempts
    # Assuming user asks, agent runs, then maybe retries
    turn_idx = 0
    for msg in messages:
        if msg["role"] == "user":
            history_str += f"User Question: {msg['content']}\n"
            # Get the list of queries for this turn
            if turn_idx < len(queries):
                attempted_queries = queries[turn_idx]
                if attempted_queries:
                    history_str += f"System (Previous SQL attempt): {attempted_queries[-1]}\n"
            history_str += "---\n"
            turn_idx += 1
            
    return history_str