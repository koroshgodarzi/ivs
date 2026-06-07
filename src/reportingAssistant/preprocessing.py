from reportingAssistant.schema import GraphState
import os
from langchain_core.runnables import RunnableConfig
import json
from utils import get_llm, extract_json_from_text, ommiting_think_block, count_chat_tokens
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

    return {"keywords": keywords, "chat_history": history}


def format_chat_history(state: GraphState, max_tokens: int = 2000) -> str:
    """
    Formats chat history into a string, truncating the oldest turns
    to ensure the total token count stays below max_tokens.
    """
    messages = state.get("messages", [])
    queries = state.get("generated_query", [])
    
    # List to store formatted strings for each turn
    formatted_turns = []
    
    # We rebuild the turns to check token counts
    # We iterate backwards (from newest to oldest) to keep the most recent context
    # and stop when we exceed the threshold
    
    turn_idx = 0
    # Group interactions into (User + System) pairs
    turns = []
    for msg in messages:
        if msg["role"] == "user":
            # Start a new turn
            turns.append({"user": msg["content"], "query": None})
            if turn_idx < len(queries) and queries[turn_idx]:
                turns[-1]["query"] = queries[turn_idx][-1] # Get last attempt
            turn_idx += 1

    # Process from newest to oldest
    final_history_str = ""
    current_token_count = 0
    
    for turn in reversed(turns):
        turn_str = f"User Question: {turn['user']}\n"
        if turn['query']:
            turn_str += f"System (Previous SQL attempt): {turn['query']}\n"
        turn_str += "---\n"
        
        # Calculate tokens for this specific turn
        turn_tokens = count_chat_tokens([{"role": "user", "content": turn_str}])
        
        if current_token_count + turn_tokens <= max_tokens:
            final_history_str = turn_str + final_history_str
            current_token_count += turn_tokens
        else:
            # Threshold reached, stop adding older context
            break
            
    return final_history_str