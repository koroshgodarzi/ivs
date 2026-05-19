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
    test_prompts = [
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