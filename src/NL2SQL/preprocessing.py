from NL2SQL.schema import GraphState
import os
from langchain_core.runnables import RunnableConfig
import json
from utils import get_llm, extract_json_from_text, ommiting_think_block


def keyword_extraction(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Extracts keywords from the user's question using an LLM.
    """
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    user_question = user_messages[-1]["content"] if user_messages else ""

    model_name = config.get("configurable", {}).get("model_name", "gpt")
    llm = get_llm(model_name)  # Assumes you have a function to get your LLM instance

    with open(os.path.join('..', 'prompt_template', 'keyword_extraction_prompt.txt')) as f:
        prompt = f.read().replace("[USER_QUESTION_HERE]", user_question)
    
    # This is a simplified call; you might need to adapt it based on your LLM's API
    response = llm.invoke(prompt)
    
    try:
        keywords = response.content.strip()
        keywords = ommiting_think_block(keywords)
        keywords = extract_json_from_text(keywords)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse keywords as JSON: {e}")

    print(f"---Extracted Keywords---\n{keywords}")

    return {"keywords": keywords}