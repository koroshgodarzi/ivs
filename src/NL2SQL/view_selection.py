from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from NL2SQL.schema import GraphState
from utils import get_llm, extract_json_from_text, ommiting_think_block, count_chat_tokens

import os
from pathlib import Path
import json


def schema_retriever(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Use LLM to identify relevant view names based on the user's prompt.
    Saves only the names to the state, omitting file loading.
    """
    model_name = config.get("configurable", {}).get("model_name", "gpt")
    llm = get_llm(model_id=model_name)
    
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    if not user_messages:
        state["retrieved_schema"] = "No user question found."
        return state
    
    view_descriptions_path = Path(__file__).parent.parent.parent / "data" / "view_descriptions.json"

    with open(view_descriptions_path, "r", encoding="utf-8") as f:
        view_descriptions = json.load(f)  
    
    with open(os.path.join('..', 'prompt_template', 'schema_retriever_system_prompt.txt')) as f:
        system_prompt = f.read()

    # with open(os.path.join('..', 'prompt_template', 'schema_retriever_shot.txt')) as f:
    #     system_prompt += f.read()

    with open(os.path.join('..', 'prompt_template', 'schema_retriever_user_prompt.txt')) as f:
        user_prompt = f.read().format(view_descriptions, user_messages)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # num_tokens_msg = count_chat_tokens(messages)
    # print(num_tokens_msg)

    response = llm.invoke(messages)
    
    try:
        schema_names = response.content.strip()
        schema_names = ommiting_think_block(schema_names)
        schema_names = extract_json_from_text(schema_names)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse schema_names as JSON: {e}")

    state["retrieved_schema"] = schema_names
    print(f"retrieved schemas are: {schema_names}")

    return state


if __name__ == "__main__":

    # test_prompts = ['Is there relation between location and date info of projects.']
    test_prompts = ['لیست پروژه های جاری «ناصر اسدی» را بده',
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
#     test_prompts = [
#     "How many active projects do I have?",
#     "How many projects are in the 'under negotiation' status?",
#     "How many projects is Naser Asadi managing that are currently in execution?",
#     "How many active projects is Naser Asadi managing?",
#     "List the projects that have a supervisor or consultant.",
#     "Which of my EPC projects have greater actual progress?",
#     "Provide the total value of completed public contracts, broken down by year.",
#     "List the delayed contracts, ordered from the worst situation.",
#     "A new project is coming up. Based on the previous project managers, who do you think I should assign it to? Consider both the number of projects each person is handling and delays in their previous projects.",
#     "Is there a relationship between the project location and the likelihood of delays?",
#     "Provide a list of terminated (cancelled) contracts.",
#     "Provide the total amount and count of ongoing contracts.",
#     "Provide the total value of contracts for each year from 2011 onward.",
#     "Provide a list of contracts that have addendums.",
#     "Contracts that have had payments but no progress statements issued.",
#     "The total amount that has been invoiced but not yet paid for ongoing contracts.",
#     "The average percentage of addendums relative to the original contract value, broken down by year.",
#     "Which of my contracts is in very bad condition? You can use criteria such as physical progress compared to the amount paid, as well as the original contract value.",
#     "Which project manager has managed their contracts better? You can define a metric using the number of completed contracts as a positive score, terminated contracts as a negative score, and deviation of the final paid amount from the original contract value as a negative score, then judge based on that."
# ]


    for tp in test_prompts:
        print(tp)
        state = GraphState(messages=[{"role": "user", "content": tp}], generated_query=None, query_results=None, error_message=None, summary_context=None, retry_count=0, validation_result=None, retrieved_schema=['vw_Contracts'])
        schema_retriever(state, config={"configurable": {"model_name": 'qwen-api'}})
        print()