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
    rephrased_quest = state.get("rephrased_query", "")

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
    prompt = prompt.replace("[USER_REPHRASED_QUESTION]", rephrased_quest)
    
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
    Formats chat history into a string, keeping the MOST RECENT turns
    up to the max_token limit, returned in chronological order.
    """
    messages = state.get("messages", [])
    
    current_token_count = 0
    history_turns = []

    # Iterate through messages from newest to oldest
    for msg in reversed(messages):
        # Format the specific turn
        if msg["role"] == "user":
            turn_str = f"User Question: {msg['content']}\n---\n"
        elif msg["role"] == "assistant":
            turn_str = f"System (Previous SQL attempt): {msg['content']}\n---\n"
        else:
            continue # Skip roles that aren't defined
        
        # Calculate tokens for this specific turn
        turn_tokens = count_chat_tokens([{"role": "user", "content": turn_str}])
        
        # Check if adding this message exceeds the limit
        if current_token_count + turn_tokens <= max_tokens:
            # Add to the beginning of our list to maintain chronological order
            history_turns.insert(0, turn_str)
            current_token_count += turn_tokens
        else:
            # Limit reached: stop processing older messages
            break

    return "".join(history_turns)