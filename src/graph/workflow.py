from graph.schema import GraphState
from graph.embedding import schema_retriever
from graph.validation import validate_query, should_proceed_with_query, handle_validation_failure
from graph.generate_query import sql_generator
from graph.execute_query import execute_query
from graph.error_handling import error_handler, should_retry, explain_query_error
from graph.final_node import format_final_response
from graph.utils import get_llm

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


def build_graph():
    """Build and compile the LangGraph workflow."""
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("schema_retriever", schema_retriever)
    workflow.add_node("validate_query", validate_query)
    workflow.add_node("handle_validation_failure", handle_validation_failure)
    workflow.add_node("sql_generator", sql_generator)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("error_handler", error_handler)
    workflow.add_node("explain_query_error", explain_query_error)
    workflow.add_node("format_response", format_final_response)
    
    # --- FLOW DEFINITION ---
    
    # 1. Start with schema retrieval
    workflow.set_entry_point("schema_retriever")
    
    # 2. Move from retrieval to validation
    workflow.add_edge("schema_retriever", "validate_query")
    
    # 3. Validation conditional routing
    workflow.add_conditional_edges(
        "validate_query",
        should_proceed_with_query,
        {
            "proceed": "sql_generator",
            "halt": "handle_validation_failure"
        }
    )
    
    # 4. Standard edges
    workflow.add_edge("handle_validation_failure", END)
    workflow.add_edge("sql_generator", "execute_query")
    workflow.add_edge("execute_query", "error_handler")
    
    # 5. Error retry logic
    workflow.add_conditional_edges(
        "error_handler",
        should_retry,
        {
            "retry": "explain_query_error",
            "end": "format_response"
        }
    )

    workflow.add_edge("explain_query_error", "error_handler")
    workflow.add_edge("format_response", END)
    
    # Compile with memory
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)



import uuid

def main():
    # 1. Ensure your API token is set
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"

    # 2. Build the compiled graph
    app = build_graph()

    # 3. Define the user's question
    questions = ["Which contracts are canceles?"]

    # 4. Initialize the state
    # This matches the 'GraphState' structure expected by your nodes
    for user_question in questions:
        initial_state = {
            "messages": [
                {"role": "user", "content": user_question}
            ],
            "retry_count": 0,
            "summary_context": None,
            "generated_query": None,
            "error_message": None,
            "query_results": None,
            "validation_result": None
        }

        # 5. Config with thread_id (required for persistent memory/SqliteSaver)
        config = {"configurable": {"thread_id": "test"}}

        print("--- Starting Text-to-SQL Workflow ---")
        print(f"User Question: {user_question}\n")

        # try:
            # 6. Run the graph
            # Use .stream() if you want to see updates node-by-node, 
            # or .invoke() to just get the final result.
        final_state = app.invoke(initial_state, config=config)

    # 7. Print the results
    print("--- Workflow Complete ---")

    state_snapshot = app.get_state(config).values
    print("--- Resulting Memory ---")

    print(state_snapshot)

    print("--- Messages ---")
    for message in state_snapshot.get("messages", []):
        print(message)
    
    # Display the SQL the AI generated
    if final_state.get("generated_query"):
        print(f"Generated SQL:\n{final_state['generated_query']}\n")

    # Display the final natural language answer
    # The last message in the list should be the assistant's response
    messages = final_state.get("messages", [])
    if messages and messages[-1]["role"] == "assistant":
        print(f"Assistant Response:\n{messages[-1]['content']}")
    
    # Check if any errors occurred during the retries
    if final_state.get("error_message") and final_state.get("retry_count", 0) >= 3:
        print(f"Final Error Message: {final_state['error_message']}")

    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # llm = get_llm()

    # # For a simple string
    # with open('/Users/korosh/Desktop/ips/vw_Contracts_schema.txt', 'r') as file:
    #     text = file.read()
    # num_tokens = llm.get_num_tokens(text)
    # print(f"Token count: {num_tokens}")
    main()
