import pandas as pd
from NL2SQL.schema import GraphState
from langchain_core.runnables import RunnableConfig

def format_final_response(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Formats the final response:
    1. Creates a Markdown table via Pandas for the chat history.
    2. Preserves raw 'query_results' (JSON) for UI/FastAPI usage.
    """
    query_explanation = state.get("query_explanation")
    results = state.get("query_results")
    error = state.get("error_message")
    
    response_content = ""

    if results is not None:
        if isinstance(query_explanation, list):
            df = pd.DataFrame(results)
            markdown_table = df.to_markdown(index=False)

            explanation_str = '\n\n'.join(str(i) for i in query_explanation) if isinstance(query_explanation, list) else query_explanation
            
            response_content = (
                f"### Query Explanation\n{explanation_str}\n\n"
                f"### Results\n{markdown_table}"
            )
        else:            
            response_content = f"### Query Explanation\n{query_explanation}\n\n**Result:** The query returned no matching records."

            if len(results) > 15:
                response_content += f"\n\n*(Showing first {len(results)} rows)*"

    elif error:
        error_detail = "\n".join(error) if isinstance(error, list) else str(error)
        response_content = f"### ❌ Query Error\nI encountered an issue while running the query:\n\n`{error_detail}`"

    else:
        response_content = "I'm sorry, I was unable to generate a valid response for that query."

    state["messages"].append({
        "role": "assistant",
        "content": response_content
    })

    return state