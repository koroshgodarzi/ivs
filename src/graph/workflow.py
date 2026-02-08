from graph.schema import GraphState
from graph.embedding import schema_retriever
from graph.validation import validate_user_question, should_proceed_with_user_question, handle_validation_failure, check_other_schemas
from graph.generate_query import sql_generator
from graph.execute_query import execute_query
from graph.error_handling import error_handler, should_retry, explain_query_error
from graph.final_node import format_final_response
from graph.utils import get_llm

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from graph.monitoring import configure_text_logger, with_state_logging

def build_graph():
    workflow = StateGraph(GraphState)

    logger = configure_text_logger()

    workflow.add_node("schema_retriever", with_state_logging("schema_retriever", schema_retriever, logger))
    workflow.add_node("validate_user_question", with_state_logging("validate_user_question", validate_user_question, logger))
    workflow.add_node("handle_validation_failure", with_state_logging("handle_validation_failure", handle_validation_failure, logger))
    workflow.add_node("sql_generator", with_state_logging("sql_generator", sql_generator, logger))
    workflow.add_node("execute_query", with_state_logging("execute_query", execute_query, logger))
    workflow.add_node("error_handler", with_state_logging("error_handler", error_handler, logger))
    workflow.add_node("check_other_schemas", check_other_schemas)
    workflow.add_node("explain_query_error", with_state_logging("explain_query_error", explain_query_error, logger))
    workflow.add_node("format_response", with_state_logging("format_response", format_final_response, logger))
    
    workflow.set_entry_point("schema_retriever")
    
    # 2. Move from retrieval to validation
    workflow.add_edge("schema_retriever", "validate_user_question")
    
    # 3. Validation conditional routing
    workflow.add_conditional_edges(
        "validate_user_question",
        should_proceed_with_user_question,
        {
            "proceed": "sql_generator",
            "loop": "check_other_schemas",
            "halt": "handle_validation_failure"
        }
    )
    
    # 4. Standard edges
    workflow.add_edge("handle_validation_failure", END)
    workflow.add_edge("check_other_schemas", "validate_user_question")
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
    workflow.add_edge("explain_query_error", "execute_query")
    workflow.add_edge("format_response", END)
    
    # Compile with memory
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


import uuid
import json
import os

def main():
    # 1. Ensure your API token is set
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"
    output = {}

    # 2. Build the compiled graph
    app = build_graph()

    # 3. Define the user's question
    questions = [
        "تعداد پروژه های فعال من چند تاست؟",
        "چند تا پروژه در حالت 'در حال مذاکره' دارم؟",
        "ناصر اسدی مدیر چند تا پروژه در وضعیت در حال اجراست؟",
        "ناصر اسدی مدیر چند تا پروژه فعاله؟",
        "لیست پروژه هایی که ناظر یا مشاور دارن",
        "کدوم یکی از پروژه های EPC من پیشرفت واقعی بیشتری دارن؟",
        "جمع رقم قراردادهای خاتمه یافته عمومی رو به تفکیک سال بده.",
        "لیست قراردادهای تاخیر دار رو به ترتیب از بدترین وضعیت بده",
        "یه پروژه جدید داره میاد. به نظرت بین مدیر پروژه های قبلی، به کدوم یکی بدمش؟ هم بحث تعداد پروژه هایی که نفر دستشه رو در نظر بگیر هم بحث تاخیر پروژه های قبلی",
        "آیا ارتباطی بین محل اجرای پروژه با احتمال تاخیرش دیده میشه؟",
        "لیست قراردادهای فسخ شده رو بده",
        "جمع مبلغ و تعداد قراردادهای جاری رو بده",
        "جمع قراردادهای هر سال از 90 به اینور رو بده",
        "لیست قراردادهایی که الحاقیه دارن رو بده",
        "قراردادهایی که صورت وضعیت نخوردن ولی پرداخت داشتن",
        "جمع مبالغی که صورت وضعیت شده اما هنوز پرداخت نشده برای قراردادهای جاری",
        "میانگین درصد الحاقیه ها نسبت به رقم قرارداد به تفکیک سال",
        "وضعیت کدوم قراردادم خیلی خرابه؟ میتونی از میزان پیشرفت فیزیکی نسبت به مبلغ پرداخت شده و همچنین مبلغ اولیه قرارداد برای معیار استفاده کنی",
        "کدوم مدیر پروژه قراردادهاش رو بهتر مدیریت کرده؟ میتونی یه معیار از تعداد قراردادهای خاتمه یافته به عنوان امتیاز مثبت، فسخ شده به عنوان امتیاز منفی، و انحراف رقم پرداخت شده نهایی نسبت به رقم اولیه به عنوان امتیاز منفی شکل بدی و بر اون اساس قضاوت کنی"
    ]

    # 4. Initialize the state
    # This matches the 'GraphState' structure expected by your nodes
    for i, user_question in enumerate(questions):
        if i != 1:
            continue
        initial_state = {
            "messages": [
                {"role": "user", "content": user_question}
            ],
            "retry_count": 0,
            "summary_context": None,
            "generated_query": None,
            "error_message": None,
            "query_results": None,
            "validation_result": None,
            "schema_to_check": 0,
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
        output[str(i)] = final_state
        with open(os.path.join('..', 'output',f'output{str(i)}.json'), 'w') as f:
            json.dump(final_state, f)  


    # with open('output.json', 'w') as f:
    #     json.dump(output, f)    
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
