from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from reportingAssistant.schema import GraphState
from reportingAssistant.date import today_date  # Added import for today's date
from utils import get_llm, extract_json_from_text, ommiting_think_block, count_chat_tokens
import os
import json


def intent_router(state: GraphState, config):
    """
    Analyzes the user's input, decides if a query is needed, 
    and generates a response immediately if no query is needed.
    If a query is needed, it rephrases the quest based on chat history.
    """
    # 1. Get current user question
    user_messages = [msg for msg in state.get("master_messages", []) if msg["role"] == "user"]
    user_question = user_messages[-1]["content"] if user_messages else ""

    # Retrieve parameters from the config
    configurable = config.get("configurable", {})
    data_dir = configurable.get("data_dir", os.path.join("..", "data", "noisy_inclusive"))
    prompt_dir = configurable.get("prompt_template_dir", os.path.join("..", "prompt_template"))
    model_name = configurable.get("model_name", "gpt")

    llm = get_llm(model_name)
    
    # 2. Fetch today's date
    current_date = today_date()

    # 3. Format chat history from master_messages
    master_messages = state.get("master_messages", [])
    chat_history_str = ""
    for msg in master_messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        chat_history_str += f"{role}: {content}\n"
        
    if not chat_history_str.strip():
        chat_history_str = "No previous chat history."

    # 4. Load the prompt template from prompt_dir / master_LLM.txt
    prompt_file_path = os.path.join(prompt_dir, "master_LLM.txt")
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as file:
            prompt_template = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the prompt template at {prompt_file_path}")

    # Inject variables safely using `.replace()` instead of `.format()` 
    # to avoid KeyError if the text file contains JSON curly braces {}.
    system_prompt = prompt_template.replace("{current_date}", str(current_date))
    system_prompt = system_prompt.replace("{chat_history}", chat_history_str)
    
    # 5. Make the single LLM call
    decision_response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ])
    
    # 6. Extract and parse the response
    try:
        content = decision_response.content
    
        if isinstance(content, str):
            decision_text = content.strip()
        elif isinstance(content, list):
            decision_text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            decision_text = ""
            
        decision_text = ommiting_think_block(decision_text)
        decision = extract_json_from_text(decision_text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse keywords as JSON: {e}")
        # Fallback decision structure in case of parsing errors
        decision = {"needs_query": False, "direct_response": "I encountered an error understanding your request."}

    current_master_messages = state.get("master_messages", [])

    if decision.get("needs_query"):
        # We capture the rephrased query if generated; if it failed, fallback to original user question
        state["generated_query"].append([])
        rephrased = decision.get("rephrased_query", user_question)
        print(rephrased)
        
        # Scenario 2 (Start): Append only the user's question to master_messages
        current_messages = state.get("messages", [])
        updated_messages = current_messages + [{"role": "user", "content": user_question}]
        
        return {
            "intent": "data_query",
            "rephrased_query": rephrased, # Store the rephrased query in the state for the SQL generator
            "messages": updated_messages,
            "final_response": None
        }
    else:
        # Extract the direct response
        direct_response = decision.get("direct_response", "")
        
        # Scenario 1: intent_router updates the master_messages itself
        updated_master_messages = current_master_messages + [
            {"role": "assistant", "content": direct_response}
        ]
        
        return {
            "intent": "chit_chat", 
            "final_response": direct_response,
            "master_messages": updated_master_messages
        }