from graph.schema import GraphState
from graph.embedding import schema_retriever
from graph.validation import validate_user_question, should_proceed_with_user_question, handle_validation_failure
from graph.generate_query import sql_generator
from graph.execute_query import execute_query
from graph.error_handling import error_handler, should_retry, explain_query_error
from graph.final_node import format_final_response
from utils import get_llm

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from graph.monitoring import configure_text_logger, with_state_logging

import time

def build_graph():
    workflow = StateGraph(GraphState)

    logger = configure_text_logger()

    workflow.add_node("schema_retriever", with_state_logging("schema_retriever", schema_retriever, logger))
    workflow.add_node("validate_user_question", with_state_logging("validate_user_question", validate_user_question, logger)) 
    workflow.add_node("handle_validation_failure", with_state_logging("handle_validation_failure", handle_validation_failure, logger))
    workflow.add_node("sql_generator", with_state_logging("sql_generator", sql_generator, logger))
    workflow.add_node("execute_query", with_state_logging("execute_query", execute_query, logger))
    workflow.add_node("error_handler", with_state_logging("error_handler", error_handler, logger))
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
    workflow.add_edge("explain_query_error", "execute_query")
    workflow.add_edge("format_response", END)
    
    # Compile with memory
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


import uuid
import json
import os
import pandas as pd
import os
import time

def main():
    # 1. Ensure your API token is set
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"
    output = {}

    # 2. Build the compiled graph
    app = build_graph()

    # 3. Define the user's question by reading from Excel
    # excel_file = "../FAQ-IPMP-1404-11-21 (1).xlsx"
    
    # try:
    #     # Reads the Excel file. 
    #     # header=None assumes the first row is data. If there is a header, remove header=None.
    #     # iloc[:, 0] takes the first column.
    #     df = pd.read_excel(excel_file, header=0) 
        
    #     # Convert the first column to a list and remove empty rows
    #     questions = df.iloc[:, 1].dropna().astype(str).tolist()
        
    #     print(f"Successfully loaded {len(questions)} questions from {excel_file}")
        
    # except FileNotFoundError:
    #     print(f"Error: The file '{excel_file}' was not found. Loading fallback questions.")
    #     # Fallback to the hardcoded list if file is missing
    #     questions = [
    #         "تعداد پروژه های فعال من چند تاست؟",
    #         "چند تا پروژه در حالت 'در حال مذاکره' دارم؟",
    #         "ناصر اسدی مدیر چند تا پروژه در وضعیت در حال اجراست؟",
    #         "ناصر اسدی مدیر چند تا پروژه فعاله؟",
    #         "لیست پروژه هایی که ناظر یا مشاور دارن",
    #         "کدوم یکی از پروژه های EPC من پیشرفت واقعی بیشتری دارن؟",
    #         "جمع رقم قراردادهای خاتمه یافته عمومی رو به تفکیک سال بده.",
    #         "لیست قراردادهای تاخیر دار رو به ترتیب از بدترین وضعیت بده",
    #         "یه پروژه جدید داره میاد. به نظرت بین مدیر پروژه های قبلی، به کدوم یکی بدمش؟ هم بحث تعداد پروژه هایی که نفر دستشه رو در نظر بگیر هم بحث تاخیر پروژه های قبلی",
    #         "آیا ارتباطی بین محل اجرای پروژه با احتمال تاخیرش دیده میشه؟",
    #         "لیست قراردادهای فسخ شده رو بده",
    #         "جمع مبلغ و تعداد قراردادهای جاری رو بده",
    #         "جمع قراردادهای هر سال از 90 به اینور رو بده",
    #         "لیست قراردادهایی که الحاقیه دارن رو بده",
    #         "قراردادهایی که صورت وضعیت نخوردن ولی پرداخت داشتن",
    #         "جمع مبالغی که صورت وضعیت شده اما هنوز پرداخت نشده برای قراردادهای جاری",
    #         "میانگین درصد الحاقیه ها نسبت به رقم قرارداد به تفکیک سال",
    #         "وضعیت کدوم قراردادم خیلی خرابه؟ میتونی از میزان پیشرفت فیزیکی نسبت به مبلغ پرداخت شده و همچنین مبلغ اولیه قرارداد برای معیار استفاده کنی",
    #         "کدوم مدیر پروژه قراردادهاش رو بهتر مدیریت کرده؟ میتونی یه معیار از تعداد قراردادهای خاتمه یافته به عنوان امتیاز مثبت، فسخ شده به عنوان امتیاز منفی، و انحراف رقم پرداخت شده نهایی نسبت به رقم اولیه به عنوان امتیاز منفی شکل بدی و بر اون اساس قضاوت کنی"
    #     ]

    questions = ['لیست پروژه های جاری «ناصر اسدی» را بده',
    'پروژه های با ارز دلاری که «Project Admin» راهبر پروژه است چند تا هست؟ نام پروژه و وضعیت جاری',
    'کدام شکست برنامه ای پروژه هایی که من راهبر پروژه هستم (من javad ahmadi هستم)، محاسبه برنامه ای ندارند؟',
    'لیست قراردادهای جاری پروژه بعثت را بده ؟',
    'لیست منابع پروژه «فاز اول- ناصری» را نیاز دارم؟']
    # 4. Initialize the state
    for i, user_question in enumerate(questions):
        if i != 0:
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
            "query_explanation": None,
        }

        # 5. Config with thread_id
        config = {"configurable": {"thread_id": f"{str(i)}", "model_name": 'qwen-api'}}

        print("--- Starting Text-to-SQL Workflow ---")
        print(f"User Question: {user_question}\n")

        # 6. Run the graph
        start_time = time.perf_counter()
        final_state = app.invoke(initial_state, config=config)
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        messages_only = final_state.get("messages", [])
        schema_only = final_state.get("retrieved_schema", [])
        query_results = final_state.get("query_results", [])
        question_data = {
            "messages": messages_only,
            "query_results": query_results,
            "schema_only": schema_only,
            "time_spent_seconds": round(duration, 4)
        }
        output[str(i)] = question_data
        # with open(os.path.join('..', 'output', 'second_question_series_gpt',f'output{str(i)}.json'), 'w') as f:
        #     json.dump(final_state, f)  

        # with open(os.path.join('..', 'output', 'second_question_series_gpt', 'output.json'), 'w') as f:
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
