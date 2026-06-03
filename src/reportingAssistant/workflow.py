from reportingAssistant.schema import GraphState
from reportingAssistant.generate_query import sql_generator_column_based
from reportingAssistant.execute_query import execute_query
from reportingAssistant.error_handling import error_handler, should_retry, explain_query_error
from reportingAssistant.final_node import format_final_response
from reportingAssistant.preprocessing import keyword_view_extraction
from reportingAssistant.column_retrieval import querying

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from reportingAssistant.monitoring import configure_text_logger, with_state_logging

import time
import os
import uuid
import json
import pandas as pd

# Define Centralized Folder Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROMPT_TEMPLATE_DIR = os.path.join(BASE_DIR, 'prompt_template')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'test')
DOCS_DIR = os.path.join(BASE_DIR, 'docs', 'test')


def column_based_graph(output_folder: str):
    workflow = StateGraph(GraphState)

    logger = configure_text_logger(output_folder)

    # Add nodes
    workflow.add_node("keyword_extraction", with_state_logging("keyword_extraction", keyword_view_extraction, logger))
    workflow.add_node("querying", with_state_logging("querying", querying, logger))
    workflow.add_node("sql_generator", with_state_logging("sql_generator", sql_generator_column_based, logger))
    workflow.add_node("execute_query", with_state_logging("execute_query", execute_query, logger))
    workflow.add_node("error_handler", with_state_logging("error_handler", error_handler, logger))
    workflow.add_node("explain_query_error", with_state_logging("explain_query_error", explain_query_error, logger))
    workflow.add_node("format_response", with_state_logging("format_response", format_final_response, logger))

    # Define paths/edges
    workflow.set_entry_point("keyword_extraction")
    workflow.add_edge("keyword_extraction", "querying")
    workflow.add_edge("querying", "sql_generator")
    workflow.add_edge("sql_generator", "execute_query")
    workflow.add_edge("execute_query", "error_handler")

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
    
    db_path = os.path.join(BASE_DIR, 'output', 'NL2SQL_langgraph_state.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


def main():
    output = {}
    output_folder = 'test'
    os.makedirs(os.path.join(BASE_DIR, 'output', output_folder), exist_ok=True)

    app = column_based_graph(output_folder)

    # questions = ['لیست پروژه های جاری «ناصر اسدی» را بده',
    # 'کدام پروژه های «دفتر مدیریت پروژه» با وضعیت جاری، ساختار شکست(WBS) ندارند؟',
    # 'لیست تمام قلم کاری ها در پروژه «خط انتقال» که پیشرفت برنامه ای و پیشرفت فیزیکی ندارند رو بده',
    # 'لیست پروژه های از نوع EPC رو بده',
    # 'پروژه های با ارز دلاری که «Project Admin» راهبر پروژه است چند تا هست؟ نام پروژه و وضعیت جاری',
    # 'اقلام کاری که مسئول مستقیم آنها «احمد نظاری» هستند ؟',
    # 'کدام شکست برنامه ای پروژه هایی که من راهبر پروژه هستم، محاسبه برنامه ای ندارند؟',
    # 'لیست پروژه هایی که ساختار شکست هزینه به یورو دارند؟',
    # 'لیست قراردادهای جاری پروژه بعثت را بده ؟',
    # 'اقلام مهندسی که با دیسیپلین Piping مرتبط هستند؟',
    # 'ریسکهایی که ذینفع آنها ABB است؟',
    # 'لیست مشکلات و موانع پروژه های «طاهر شعبانی»‌ به همراه نام پروژه؟',
    # 'لیست پروژه با وضعیت در حال اجرا بدون پیشرفت که تاریخ شروع آنها نسبت تاریخ روز گذشته است',
    # 'لیست قراردادهای جاری پروژه بعثت که الحاقیه دارند رو بده ؟',
    # 'لیست صورت وضعیت هایی که مربوط به ساختار هزینه «هزینه کارگاه» پروژه «فاز اول- ناصری» است؟',
    # 'لیست تمام اقلام کاریهایی که تنها ۲۰ درصد پیشرفت دارند و تاریخ پایان آنها گذشته است را نیاز دارم؟',
    # 'لیست قراردادهایی که تعدیل دارند از پروژه «فاز اول- احمدی» بده؟',
    # 'لیست پروژه هایی که ریسک منفی و تاریخ شناسایی قبل از ۶ ماه پیش دارند؟',
    # 'لیست قراردادهای پروژه «فاز اول- ناصری» که پرداخت بدون صورت وضعیت دارند؟',
    # 'لیست منابع پروژه «فاز اول- ناصری» را نیاز دارم؟',
    # 'کدام شکست برنامه ای پروژه هایی که من راهبر پروژه هستم (من javad ahmadi هستم)، محاسبه برنامه ای ندارند؟']

    questions = ["وضع پیشرفت پروژه‌ها چگونه است؟"]

    for i, user_question in enumerate(questions):
        # if i <= 7: continue
        print()
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

        # Dynamically pass global paths through the configuration context
        config = {
            "configurable": {
                "thread_id": f"{str(i)}", 
                "model_name": 'open_router',
                "prompt_template_dir": PROMPT_TEMPLATE_DIR,
                "data_dir": DATA_DIR,
                "docs_dir": DOCS_DIR
            }
        }

        info_path = os.path.join(BASE_DIR, 'output', output_folder, 'info.txt')
        if not os.path.exists(info_path):
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Model: {config['configurable']['model_name']}\n")
                f.write("Qwen3-30B-A3B-lbu2r")

        print("--- Starting Text-to-SQL Workflow ---")
        print(f"User Question: {user_question}\n")

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

        with open(os.path.join(BASE_DIR, 'output', output_folder ,f'output_{str(i)}.json'), 'w', encoding="utf-8") as f:
            json.dump(final_state, f, ensure_ascii=False, indent=2)  

        with open(os.path.join(BASE_DIR, 'output', output_folder, 'output.json'), 'a', encoding="utf-8") as f:
            json.dump(question_data, f, ensure_ascii=False, indent=2)    


if __name__ == "__main__":
    main()