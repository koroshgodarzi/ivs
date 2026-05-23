from NL2SQL.schema import GraphState
from NL2SQL.view_selection import schema_retriever
from NL2SQL.validation import validate_user_question, should_proceed_with_user_question, handle_validation_failure
from NL2SQL.generate_query import sql_generator
from NL2SQL.execute_query import execute_query
from NL2SQL.error_handling import error_handler, should_retry, explain_query_error
from NL2SQL.final_node import format_final_response
from NL2SQL.preprocessing import keyword_extraction
from NL2SQL.column_retrieval import querying

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from NL2SQL.monitoring import configure_text_logger, with_state_logging

import time
import os


def column_based_graph(output_folder: str):
    workflow = StateGraph(GraphState)

    logger = configure_text_logger(output_folder)

    # Add the new nodes
    workflow.add_node("keyword_extraction", with_state_logging("keyword_extraction", keyword_extraction, logger))
    workflow.add_node("querying", with_state_logging("querying", querying, logger))

    # Add the existing nodes
    workflow.add_node("sql_generator", with_state_logging("sql_generator", sql_generator, logger))
    workflow.add_node("execute_query", with_state_logging("execute_query", execute_query, logger))
    workflow.add_node("error_handler", with_state_logging("error_handler", error_handler, logger))
    workflow.add_node("explain_query_error", with_state_logging("explain_query_error", explain_query_error, logger))
    workflow.add_node("format_response", with_state_logging("format_response", format_final_response, logger))

    # 1. Set the entry point
    workflow.set_entry_point("keyword_extraction")

    # 2. Define the edges
    workflow.add_edge("keyword_extraction", "querying")
    workflow.add_edge("querying", "sql_generator")
    workflow.add_edge("sql_generator", "execute_query")
    workflow.add_edge("execute_query", "error_handler")

    # 3. Error retry logic
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
    
    # Set up memory
    db_path = os.path.join('..', 'output', 'NL2SQL_langgraph_state.db')
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


def view_based_graph(output_folder: str):
    workflow = StateGraph(GraphState)

    logger = configure_text_logger(output_folder)

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
    
    db_path = os.path.join('..', 'output', 'NL2SQL_langgraph_state.db')
    conn = sqlite3.connect(db_path, check_same_thread=False)
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
    output_folder = 'test'
    os.makedirs(os.path.join('..', 'output', output_folder), exist_ok=True)

    app = view_based_graph(output_folder)

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
    'کدام پروژه های «دفتر مدیریت پروژه» با وضعیت جاری، ساختار شکست(WBS) ندارند؟',
    'لیست تمام قلم کاری ها در پروژه «خط انتقال» که پیشرفت برنامه ای و پیشرفت فیزیکی ندارند رو بده',
    'لیست پروژه های از نوع EPC رو بده',
    'پروژه های با ارز دلاری که «Project Admin» راهبر پروژه است چند تا هست؟ نام پروژه و وضعیت جاری',
    'اقلام کاری که مسئول مستقیم آنها «احمد نظاری» هستند ؟',
    'کدام شکست برنامه ای پروژه هایی که من راهبر پروژه هستم، محاسبه برنامه ای ندارند؟',
    'لیست پروژه هایی که ساختار شکست هزینه به یورو دارند؟',
    'لیست قراردادهای جاری پروژه بعثت را بده ؟',
    'اقلام مهندسی که با دیسیپلین Piping مرتبط هستند؟',
    'ریسکهایی که ذینفع آنها ABB است؟',
    'لیست مشکلات و موانع پروژه های «طاهر شعبانی»‌ به همراه نام پروژه؟',
    'لیست پروژه با وضعیت در حال اجرا بدون پیشرفت که تاریخ شروع آنها نسبت تاریخ روز گذشته است',
    'لیست قراردادهای جاری پروژه بعثت که الحاقیه دارند رو بده ؟',
    'لیست صورت وضعیت هایی که مربوط به ساختار هزینه «هزینه کارگاه» پروژه «فاز اول- ناصری» است؟',
    'لیست تمام اقلام کاریهایی که تنها ۲۰ درصد پیشرفت دارند و تاریخ پایان آنها گذشته است را نیاز دارم؟',
    'لیست قراردادهایی که تعدیل دارند از پروژه «فاز اول- احمدی» بده؟',
    'لیست پروژه هایی که ریسک منفی و تاریخ شناسایی قبل از ۶ ماه پیش دارند؟',
    'لیست قراردادهای پروژه «فاز اول- ناصری» که پرداخت بدون صورت وضعیت دارند؟',
    'لیست منابع پروژه «فاز اول- ناصری» را نیاز دارم؟',
    'کدام شکست برنامه ای پروژه هایی که من راهبر پروژه هستم (من javad ahmadi هستم)، محاسبه برنامه ای ندارند؟']
    # 4. Initialize the state

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
            "query_explanation": None,
            "retrieved_columns": None,
            "keywords": None,
            "query_generation_user_prompt": None
        }

        # 5. Config with thread_id
        config = {"configurable": {"thread_id": f"{str(i)}", "model_name": 'qwen_api'}}

        if not os.path.exists(os.path.join('..', 'output', output_folder, 'info.txt')):
            with open(os.path.join('..', 'output', output_folder, 'info.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Model: {config['configurable']['model_name']}")
                f.write("Qwen3-30B-A3B-lbu2r")

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

        with open(os.path.join('..', 'output', output_folder ,f'output_{str(i)}.json'), 'w', encoding="utf-8") as f:
            json.dump(final_state, f, ensure_ascii=False, indent=2)  

        with open(os.path.join('..', 'output', output_folder, 'output.json'), 'a', encoding="utf-8") as f:
            json.dump(question_data, f, ensure_ascii=False, indent=2)    

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
