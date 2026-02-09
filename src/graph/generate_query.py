from graph.utils import get_llm, extract_json_from_text, create_column_names_for_schemas, ommiting_think_block
from graph.schema import GraphState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import json
import os


def sql_generator(state: GraphState) -> GraphState:
    """Node 1: Generate SQL query from user input using LLM."""
    llm = get_llm(max_tokens=2048)

    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    user_question = user_messages[-1]["content"] if user_messages else ""

    validation_result = state.get("validation_result", {})
    if isinstance(validation_result, str):
        try:
            validation_result = extract_json_from_text(validation_result)
        except (json.JSONDecodeError, ValueError):
            validation_result = {}

    all_schemas_metadata = create_column_names_for_schemas(validation_result['Needed table and categories'])

    with open(os.path.join('..', 'prompt_template', 'query_generation_user_prompt.txt')) as f:
        user_prompt = f.read()
    
    user_prompt = user_prompt.format(all_schemas_metadata, user_question)

    with open(os.path.join('..', 'prompt_template', 'query_generation_system_prompt.txt')) as f:
        system_prompt = f.read()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)

    try:
        query_generation_result = response.content.strip()
        query_generation_result = ommiting_think_block(query_generation_result)
        query_generation_result = extract_json_from_text(query_generation_result)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse query_generation_result as JSON: {e}")

    query_explanation = []
    query_explanation.append(query_generation_result['query_explanation'])
    state['query_explanation'] = query_explanation
    
    sql_query = (
        query_generation_result["generated_query"]
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
    validation_result = {'short answer': 'Yes', 'Needed table and categories': {'vw_Projects': [2, 4]}}
    state = GraphState(messages=[{"role": "user", "content": "Status of how many contracts are canceled?"}], generated_query=None, query_results=None, error_message=None, summary_context=None, retry_count=0, validation_result=validation_result, retrieved_schema=['vw_Contracts', 'vw_Projects'])
    state = sql_generator(state)
    print(state)