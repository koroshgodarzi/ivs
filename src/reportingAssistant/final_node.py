import pandas as pd
from NL2SQL.schema import GraphState
from langchain_core.runnables import RunnableConfig

def format_final_response(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Formats the final response by processing SQL results, errors, or chit-chat.
    Updates response lists for both the UI and internal history.
    """
    # 1. Extract raw data from state
    results = state.get("query_results")
    error = state.get("error_message")
    explanation = state.get("query_explanation", [])
    chit_chat = state.get("final_response")
    
    # Safely extract the last generated SQL query
    all_turns = state.get("generated_query", [])
    last_query = all_turns[-1][-1] if all_turns and all_turns[-1] else ""

    # 2. Determine content based on the scenario
    ui_content = ""
    internal_content = ""

    # SCENARIO A: Chit-chat / General Response
    if chit_chat:
        ui_content = chit_chat
        internal_content = chit_chat

    # SCENARIO B: SQL Query Success
    elif results is not None:
        exp_text = explanation[-1] if isinstance(explanation, list) and explanation else str(explanation)
        
        if results:
            df = pd.DataFrame(results)
            markdown_table = df.to_markdown(index=False)
            limit_msg = f"\n\n*(Showing first {len(results)} rows)*" if len(results) > 15 else ""
            
            ui_content = f"### Query Explanation\n{exp_text}\n\n### Results\n{markdown_table}{limit_msg}"
        else:
            ui_content = f"### Query Explanation\n{exp_text}\n\n**Result:** The query returned no matching records."
        
        internal_content = last_query

    # SCENARIO C: SQL Query Error
    elif error:
        error_detail = error[-1] if isinstance(error, list) else str(error)
        ui_content = f"### ❌ Query Error\nI encountered an issue while running the query:\n\n`{error_detail}`"
        internal_content = f"SQL Query:\n{last_query}\n\nError:\n{error_detail}"

    # SCENARIO D: Fallback
    else:
        ui_content = "I'm sorry, I was unable to generate a valid response."
        internal_content = ui_content

    # 3. Update State
    # UI-specific response
    state["response"].append(ui_content)

    # Standard chat history
    state["messages"].append({
        "role": "assistant", 
        "content": internal_content
    })

    # Master messages (only for SQL scenarios, as per original logic)
    if not chit_chat:
        state["master_messages"].append({
            "role": "assistant",
            "content": ui_content if ui_content else "Failed to generate query."
        })

    return state