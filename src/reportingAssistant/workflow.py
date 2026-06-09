from reportingAssistant.schema import GraphState
from reportingAssistant.generate_query import sql_generator_column_based
from reportingAssistant.execute_query import execute_query
from reportingAssistant.error_handling import error_handler, should_retry, explain_query_error
from reportingAssistant.final_node import format_final_response
from reportingAssistant.preprocessing import keyword_view_extraction
from reportingAssistant.column_retrieval import querying
from reportingAssistant.visualization import visualization, rendering_node

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
DATA_DIR = os.path.join(BASE_DIR, 'data', 'clean')
DOCS_DIR = os.path.join(BASE_DIR, 'data', 'clean')


def column_based_graph(output_folder: str='test'):
    workflow = StateGraph(GraphState)

    logger = configure_text_logger(output_folder)

    # Add existing nodes
    workflow.add_node("keyword_extraction", with_state_logging("keyword_extraction", keyword_view_extraction, logger))
    workflow.add_node("querying", with_state_logging("querying", querying, logger))
    workflow.add_node("sql_generator", with_state_logging("sql_generator", sql_generator_column_based, logger))
    workflow.add_node("execute_query", with_state_logging("execute_query", execute_query, logger))
    workflow.add_node("error_handler", with_state_logging("error_handler", error_handler, logger))
    workflow.add_node("explain_query_error", with_state_logging("explain_query_error", explain_query_error, logger))
    
    # 1. Add the new visualization nodes
    workflow.add_node("visualization", with_state_logging("visualization", visualization, logger))
    workflow.add_node("rendering_node", with_state_logging("rendering_node", rendering_node, logger))
    
    workflow.add_node("format_response", with_state_logging("format_response", format_final_response, logger))

    # Define standard paths/edges
    workflow.set_entry_point("keyword_extraction")
    workflow.add_edge("keyword_extraction", "querying")
    workflow.add_edge("querying", "sql_generator")
    workflow.add_edge("sql_generator", "execute_query")
    workflow.add_edge("execute_query", "error_handler")
    workflow.add_edge("explain_query_error", "execute_query")

    # 3. Apply the updated conditional edges
    workflow.add_conditional_edges(
        "error_handler",
        route_after_execution,
        {
            "retry": "explain_query_error",
            "visualize": "visualization",
            "format": "format_response"
        }
    )
    
    # 4. Connect the visualization pipeline to the format_response node
    workflow.add_edge("visualization", "rendering_node")
    workflow.add_edge("rendering_node", "format_response")

    # End the graph
    workflow.add_edge("format_response", END)
    
    # DB Setup
    db_path = os.path.join(BASE_DIR, 'output', 'NL2SQL_langgraph_state.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)


def route_after_execution(state: GraphState, config):
    """
    Routes the graph after the query is executed and checked for errors.
    """
    # Call your existing retry logic
    retry_decision = should_retry(state, config)
    
    if retry_decision == "retry":
        return "retry"
    else:
        # If the query succeeded, check if we need to plot
        to_plot = state["keywords"]["Views"]
        if to_plot:
            return "visualize"
        else:
            return "format"


def main():
    output = {}
    output_folder = 'DiagramFirstTry'
    follow_up = False
    os.makedirs(os.path.join(BASE_DIR, 'output', output_folder), exist_ok=True)

    app = column_based_graph(output_folder)

    questions = ['لیست پروژه های جاری «ناصر اسدی» را بده',
    'کدام پروژه های «دفتر مدیریت پروژه» با وضعیت جاری، ساختار شکست(WBS) ندارند؟',
    'لیست تمام قلم کاری ها در پروژه «خط انتقال» که پیشرفت برنامه ای و پیشرفت فیزیکی ندارند رو بده',
    'لیست پروژه های از نوع EPC رو بده',
    'پروژه های با ارز دلاری که «Project Admin» راهبر پروژه است چند تا هست؟ نام پروژه و وضعیت جاری',
    'اقلام کاری که مسئول مستقیم آنها «احمد نظاری» هستند ؟',
    # 'کدام شکست برنامه ای پروژه هایی که javad ahmadi راهبر پروژه آن است، محاسبه برنامه ای ندارند؟',
    'لیست پروژه هایی که ساختار شکست هزینه به یورو دارند؟',
    'لیست قراردادهای جاری پروژه بعثت را بده ؟',
    'مدارک مهندسی که با دیسیپلین Piping مرتبط هستند؟',
    'ریسکهایی که ذینفع آنها ABB است؟',
    'لیست مشکلات و موانع پروژه های «طاهر شعبانی»‌ به همراه نام پروژه؟',
    'لیست پروژه با وضعیت در حال اجرا بدون پیشرفت که تاریخ شروع آنها نسبت تاریخ روز گذشته است',
    'لیست قراردادهای جاری پروژه بعثت که الحاقیه دارند رو بده ؟',
    'لیست صورت وضعیت هایی که مربوط به ساختار هزینه «هزینه کارگاه» پروژه «فاز اول- ناصری» است؟',
    'لیست تمام اقلام کاریهایی که کمتر از ۲۰ درصد پیشرفت دارند و تاریخ پایان آنها گذشته است را نیاز دارم؟',
    'لیست قراردادهایی که تعدیل دارند از پروژه «فاز اول- احمدی» بده؟',
    'لیست پروژه هایی که ریسک منفی و تاریخ شناسایی قبل از ۶ ماه پیش دارند؟',
    'لیست قراردادهای پروژه «فاز اول- ناصری» که پرداخت بدون صورت وضعیت دارند؟',
    'لیست منابع پروژه «فاز اول- ناصری» را نیاز دارم؟',
    'کدام شکست برنامه ای پروژه هایی که من راهبر پروژه هستم (من javad ahmadi هستم)، محاسبه‌ی برنامه ای ندارند؟']

    # questions = [
    #     "وضع پیشرفت پروژه‌های تهران چگونه است؟",
    #     "لیست پروژه‌های با پیشرفت برنامه‌ای بالای ۵۰ را بده.",
    #     "مجموع مطالبات پیمانکاران پروژه‌ی ناصری چه قدر است؟",
    #     "کدام یک از ردیف‌های CBS، اورباجت شده‌اند؟",
    #     "پروژه‌هایی که در یک ماه گذشته پیشرفت اکچوال نداشته‌اند؟",
    #     "پروژه‌هایی که در یک ماه گذشته ورود اطلاعات نداشته‌اند؟",
    #     "مشکلات مشترک بین پروژه‌ها را به من بگو.",
    #     "فعالیت‌های روی مسیر بحرانی پروژه‌ی ناصری کدامند؟",
    #     "کدام قراردادها روی مسیر بحرانی پروژه‌ی ناصری‌اند؟",
    #     "کل قرادادهایی که پیشرفت مالی ۹۰ درصد به بالا دارند را بده.",
    #     "کدام قراردادها پیشرفت مالی‌شان بیش از پیشرفت فیزیکی‌شان است؟",
    #     "قراردادهای چندارزی را به همراه مبالغ‌شان به‌ام بده.",
    #     "کدام پروژه‌ها در ۶ ماه گذشته ریپلن شده‌اند؟",
    #     "کدام پروژه بیشترین پیشرفت را از اول سال داشته است؟",
    #     "کدام پروژه بیشترین راندمان را از اول سال داشته است؟",
    #     "پروژه‌های EPC با تاخیر بیشتر از ۳۰ درصد را بهم بده.",
    #     "مجموع صورت وضعیت‌های در جریان گردش برای هر قرارداد را بهم بده.",
    #     "تضامین قرارداهایی که در یک ماه آینده منقضی می‌شوند را بهم بده.",
    #     "تضامین قرارداهایی که در یک ماه آینده سررسید می‌شوند را بهم بده.",
    #     "کدام پروژه بیشترین ریسک‌ها را دارد؟",
    #     "مدارک مهندسی که تاخیر دارند را بهم بده.",
    #     "مدارک مهندسی دیسیپلین الکتریکال که تاخیر دارند را بهم بده.",
    #     "فعالیت‌های روی مسیر بحرانی پروژه‌ی ناصری را بهم بده.",
    #     "پروژه‌هایی که پارسال شروع شدند را بهم بده.",
    #     "پروژه‌هایی که امسال باید تمام بشوند را بهم بده.",
    #     "بیشترین تاخیر پروژه‌ی ناصری کجای برنامه است؟",
    #     "پیشرفت فعالیت‌های سطح یک پروژه‌ی ناصری را بهم بده.",
    #     "تا الآن توی پروژه‌ی ناصری چه قدر پول خرج کرده‌ام؟",
    #     "برنامه‌ی جریان نقدی پروژه‌ی ناصری را با هزینه‌کردش مقایسه کن.",
    #     "لیست فعالیت‌های اتمام یافته‌ی پروژه‌ی ناصری رو بهم بده.",
    #     "فعالیت‌های اتمام یافته‌ی پروژه‌ی ناصری چند تاست.",
    #     "چند درصد فعالیت‌های پروژه‌ی ناصری اتمام یافته است؟"
    # ]
    # questions = [
    #     "نمودار میزان پیشرفت پروژه‌های تهران را رسم کن.",
    #     "نمودار پروژه‌هایی که پیشرفت برنامه‌ای بالای ۵۰ درصد دارند را رسم کن.",
    #     "نمودار مجموع مطالبات پروژه‌ی ناصری را به تفکیک پیمانکار رسم کن.",
    #     "نمودار مبالغ ردیف‌های اورباجت شده‌ی CBS را نشان بده.",
    #     "نمودار پیشرفت مالی قراردادهایی که بالای ۹۰ درصد پیشرفت دارند را رسم کن.",
    #     "نمودار مبالغ قراردادهای چندارزی را به تفکیک ارز و شماره قرارداد رسم کن.",
    #     "نمودار میزان تاخیر پروژه‌های EPC که بالای ۳۰ درصد تاخیر دارند را نشان بده.",
    #     "نمودار مجموع مبالغ صورت‌وضعیت‌های در جریان گردش را به تفکیک قرارداد رسم کن.",
    #     "نمودار میزان تاخیر مدارک مهندسی را به تفکیک پروژه رسم کن.",
    #     "نمودار میزان تاخیر مدارک مهندسی دیسیپلین الکتریکال را به تفکیک پروژه نشان بده.",
    #     "نمودار زمان‌بندی (تاریخ شروع) پروژه‌هایی که پارسال آغاز شده‌اند را رسم کن.",
    #     "نمودار زمان‌بندی (تاریخ اتمام) پروژه‌هایی که امسال باید تمام شوند را رسم کن.",
    #     "نمودار میزان تاخیر فعالیت‌ها (اقلام کاری) در پروژه‌ی ناصری را رسم کن.",
    #     "نمودار میزان پیشرفت فعالیت‌های سطح یک (اقلام کاری) پروژه‌ی ناصری را نشان بده.",
    #     "نمودار مقایسه‌ای برنامه‌ی جریان نقدی و هزینه‌کرد پروژه‌ی ناصری را رسم کن.",
    #     "نمودار تاریخ اتمام فعالیت‌های پایان‌یافته‌ی پروژه‌ی ناصری را رسم کن.",
    #     "نمودار درصد فعالیت‌های اتمام‌یافته در برابر درصد باقیمانده‌ی پروژه‌ی ناصری را رسم کن."

    # ]
    if not follow_up:
        for i, user_question in enumerate(questions):
            if i == 19: continue
            print()
            initial_state = {
                "messages": [
                    {"role": "user", "content": user_question}
                ],
                "retry_count": 0,
                "summary_context": None,
                "generated_query": [],
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

# 1. Initialize the state ONCE outside the loop to persist the session
    else:
        for i, user_question in enumerate(questions):

            current_state = {
                "messages": [],
                "retry_count": 0,
                "summary_context": None,
                "generated_query": [],        # Now List[List[str]]
                "error_message": [],          # Changed to list to track errors per attempt
                "query_results": None,
                "validation_result": None,
                "query_explanation": [],      # Changed to list
                "retrieved_columns": None,
                "keywords": None,
                "query_generation_user_prompt": None
            }

            # 2. Iterate through questions in the same session
            for i, user_question in enumerate(questions):
                print(f"\n--- Processing Turn {i}: {user_question} ---")

                # Append new question to messages
                current_state["messages"].append({"role": "user", "content": user_question})
                
                # Initialize a new inner list for this turn's query attempts
                current_state["generated_query"].append([])
                
                # Reset transient fields for the new turn
                current_state["retry_count"] = 0
                current_state["error_message"] = [] 

                config = {
                    "configurable": {
                        "thread_id": "session_01", 
                        "model_name": 'open_router',
                        "prompt_template_dir": PROMPT_TEMPLATE_DIR,
                        "data_dir": DATA_DIR,
                        "docs_dir": DOCS_DIR
                    }
                }

                # Invoke the graph, passing the PERSISTENT state
                start_time = time.perf_counter()
                current_state = app.invoke(current_state, config=config)
                end_time = time.perf_counter()
                
                # Process results as before
                duration = end_time - start_time
                
                # Capture the assistant's final response for the chat history
                # (Assuming format_final_response updated the 'messages' list)
                
                question_data = {
                    "turn": i,
                    "question": user_question,
                    "query_results": current_state.get("query_results"),
                    "time_spent_seconds": round(duration, 4)
                }

                # Save output for this turn
                with open(os.path.join(BASE_DIR, 'output', output_folder, f'turn_{i}.json'), 'w', encoding="utf-8") as f:
                    json.dump(current_state, f, ensure_ascii=False, indent=2)

            print("--- Chat Session Complete ---")


if __name__ == "__main__":
    main()