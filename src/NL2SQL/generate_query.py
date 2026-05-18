from utils import get_llm, extract_json_from_text, create_column_names_for_schemas, ommiting_think_block, get_rag_context, count_chat_tokens, create_ddl_for_schemas
from NL2SQL.schema import GraphState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

import json
import os


def sql_generator(state: GraphState, config: RunnableConfig) -> GraphState:
    """Node: Generate SQL query from user input using selected columns."""
    model_name = config.get("configurable", {}).get("model_name", "gpt")
    llm = get_llm(model_id=model_name)

    # 1. Get User Question
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    user_question = user_messages[-1]["content"] if user_messages else ""

    # 2. Extract needed_columns from validation_result
    validation_result = state.get("validation_result", {})
    if isinstance(validation_result, str):
        try:
            validation_result = extract_json_from_text(validation_result)
        except (json.JSONDecodeError, ValueError):
            validation_result = {}

    # The key is now 'needed_columns' as per the new validator logic
    needed_columns_dict = validation_result.get('needed_columns', {})

    # 3. Generate DDL/Metadata context for ONLY the selected columns
    # We create a new helper function below that handles column names directly
    all_schemas_metadata = create_ddl_for_schemas(needed_columns_dict)
    
    # Store for reference/logging
    state["retrieved_columns"] = all_schemas_metadata

    # 4. Prepare Prompts
    with open(os.path.join('..', 'prompt_template', 'query_generation_user_prompt.txt')) as f:
        user_prompt_template = f.read()
    
    rag_context = [] # get_rag_context(user_question) if applicable

    user_prompt = user_prompt_template.format(all_schemas_metadata, rag_context, user_question)

    with open(os.path.join('..', 'prompt_template', 'query_generation_system_prompt.txt')) as f:
        system_prompt = f.read()

    with open(os.path.join('..', 'prompt_template', 'query_generation_shot.txt')) as f:
        system_prompt += f.read()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # 5. Invoke LLM
    print(f"Token count for generation: {count_chat_tokens(messages)}")
    response = llm.invoke(messages)
    
    try:
        raw_content = ommiting_think_block(response.content.strip())
        query_generation_result = extract_json_from_text(raw_content)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse query_generation_result as JSON: {e}")
        # Fallback or error handling logic here
        return state

    # 6. Update State
    query_explanation = state.get('query_explanation') or []
    query_explanation.append(query_generation_result.get('query_explanation', ""))
    state['query_explanation'] = query_explanation
    
    sql_query = (
        query_generation_result.get("generated_query", "")
        .strip()
        .replace("```sql", "")
        .replace("```", "")
        .strip()
    )

    queries = state.get("generated_query") or []
    queries.append(sql_query)
    state["generated_query"] = queries

    print(f"Generated query accepted:\n{sql_query}")

    return state

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1", "model_name": 'qwen-api'}}
    validation_result = {'short answer': 'Yes', 'Needed table and categories': {'vw_Projects': [2, 4]}}
    state = GraphState(messages=[{"role": "user", "content": "Status of how many contracts are canceled?"}], generated_query=None, query_results=None, error_message=None, summary_context=None, retry_count=0, validation_result=validation_result, retrieved_schema=['vw_Contracts', 'vw_Projects'])
    state = sql_generator(state, config)
    print(state)