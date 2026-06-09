from utils import (
    get_llm, extract_json_from_text, format_join_info_for_llm, 
    ommiting_think_block, count_chat_tokens, create_ddl_for_schemas, 
    create_data_context_for_schemas, fix_sql_wildcards, get_join_relationships
)
from reportingAssistant.schema import GraphState
from reportingAssistant.date import today_date
from reportingAssistant.preprocessing import format_chat_history
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

import json
import os


def sql_generator_column_based(state: GraphState, config: RunnableConfig) -> GraphState:
    """Node: Generate SQL query from user input using selected columns and retrieved values."""
    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name", "gpt")
    llm = get_llm(model_id=model_name, reasoning=True, max_tokens=12000)

    # Extract configurations
    prompt_dir = configurable.get("prompt_template_dir", os.path.join("..", "prompt_template"))
    data_dir = configurable.get("data_dir", os.path.join("..", "data", "noisy_inclusive"))
    docs_dir = configurable.get("docs_dir", os.path.join("..", "docs", "noisy_inclusive"))

    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    user_question = user_messages[-1]["content"] if user_messages else ""

    needed_columns_dict = state["retrieved_columns"]
    
    # Pass explicit path definitions down to helper functions
    join_info = get_join_relationships(state["keywords"]["Views"], csv_file_path=os.path.join(docs_dir, 'IDColumns.csv'))
    all_schemas_metadata = create_ddl_for_schemas(needed_columns_dict, data_dir=data_dir)
    data_context = create_data_context_for_schemas(needed_columns_dict, data_dir=data_dir)
    join_info = format_join_info_for_llm(join_info)
    today = today_date(["Persian"])

    date = '\n'.join(f"in {calendar} is {date}" for calendar, date in today.items())

    history = state["chat_history"]
    
    retrieved_values = state.get("retrieved_values", {})
    
    # Load and format the prompt files using global path
    with open(os.path.join(prompt_dir, 'query_generation_user_prompt_column_based.txt'), 'r', encoding='utf-8') as f:
        user_prompt_template = f.read()

    user_prompt = user_prompt_template.format(
        date=date,
        schemas=all_schemas_metadata,
        data_context=data_context,
        join_info=join_info,
        retrieved_values=retrieved_values,
        user_question=user_question,
        history=history
    )

    keywords = state.get("keywords", {"PM_Concepts": None, "Views": [], "Attributes": [], "Values": []})
    pm_concepts = keywords.get("PM_Concepts", None)
    if pm_concepts:
        user_prompt = "### PROJECT MANAGEMENT CONCEPTS:\n" + pm_concepts + "\n\n" + user_prompt

    state["query_generation_user_prompt"] = user_prompt

    with open(os.path.join(prompt_dir, 'query_generation_system_prompt.txt'), 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    print(f"Token count for generation: {count_chat_tokens(messages)}")
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
        raw_content = ommiting_think_block(content)
        query_generation_result = extract_json_from_text(raw_content)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse query_generation_result as JSON: {e}")
        return state

    explanation_list = state.get('query_explanation', [])
    explanation_list.append(query_generation_result.get('query_explanation', ""))
    state['query_explanation'] = explanation_list
    
    sql_query = (
        query_generation_result.get("generated_query", "")
        .strip()
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )
    sql_query = fix_sql_wildcards(sql_query)

    all_turns = state.get("generated_query") or []
    if not all_turns:
        all_turns = [[]]
    
    # Append the new query to the current turn's list
    all_turns[-1].append(sql_query)
    state["generated_query"] = all_turns

    print(f"Generated query accepted:\n{sql_query}")

    return state