from graph.utils import get_llm, extract_json_from_text, create_column_names_for_schemas, ommiting_think_block
from graph.schema import GraphState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import json
import os


def get_system_prompt(schema_list: list, needed_categories: str) -> str:
    """Build the system prompt with schema and optional summary context."""
    with open(os.path.join('..', 'prompt_template', 'query_generation_system_prompt.txt')) as f:
        system_prompt_template = f.read()
    
    all_schemas_metadata = create_column_names_for_schemas(schema_list, needed_categories)
    base_prompt = system_prompt_template.format(all_schemas_metadata)
    
    return base_prompt


def sql_generator(state: GraphState) -> GraphState:
    """Node 1: Generate SQL query from user input using LLM with human correction loop."""
    
    llm = get_llm(max_tokens=2048)

    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    user_prompt = user_messages[-1]["content"] if user_messages else ""

    validation_result = state.get("validation_result", {})
    if isinstance(validation_result, str):
        try:
            validation_result = extract_json_from_text(validation_result)
        except (json.JSONDecodeError, ValueError):
            validation_result = {}

    column_categories = (
        validation_result.get("Needed categories", "")
        if isinstance(validation_result, dict)
        else ""
    )

    schema_list = state.get("retrieved_schema", [])
    if isinstance(schema_list, str):
        schema_list = [schema_list]

    system_prompt = get_system_prompt(schema_list, column_categories)

    # Initial message stack
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    while True:
        response = llm.invoke(messages)

        try:
            query_generation_result = response.content.strip()
            query_generation_result = ommiting_think_block(query_generation_result)
            query_generation_result = extract_json_from_text(query_generation_result)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse query_generation_result as JSON: {e}")
            return state

        user_command = input(
            f"{query_generation_result['query_explanation']}\n"
            "Do you approve of the procedure? (y / n / h): "
        ).lower()

        if user_command == "y":
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
            break

        elif user_command == "n":
            user_instructions = input("Type your corrections or instructions: ")

            # Add the model's previous response FIRST
            messages.append(
                AIMessage(content=response.content)
            )

            # Then add the user's correction
            messages.append(
                HumanMessage(
                    content=(
                        "Please revise the previous SQL query using these corrections:\n"
                        f"{user_instructions}"
                    )
                )
            )

        elif user_command == "h":
            print("SQL generation halted by user.")
            break

        else:
            print("Invalid input. Please type y, n, or h.")

    return state


if __name__ == "__main__":
    validation_result = '```json\n{\n  "short answer": "Yes",\n  "Needed categories": [2]\n}\n```'
    state = GraphState(messages=[{"role": "user", "content": "Status of how many contracts are canceled?"}], generated_query=None, query_results=None, error_message=None, summary_context=None, retry_count=0, validation_result=validation_result, retrieved_schema=['vw_Contracts'])
    state = sql_generator(state)
    print(state)