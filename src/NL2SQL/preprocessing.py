from NL2SQL.schema import GraphState
import os
from langchain_core.runnables import RunnableConfig
import json
from utils import get_llm, extract_json_from_text, ommiting_think_block
from pathlib import Path


def keyword_view_extraction(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Extracts keywords, attributes, and actual DB view names from the user's question.
    """
    # 1. Retrieve the user's question
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    user_question = user_messages[-1]["content"] if user_messages else ""

    # 2. Load the view descriptions
    # Adjust 'view_descriptions.json' path if it is located in a different directory
    view_desc_path = Path(__file__).parent.parent.parent / "data" / "view_descriptions.json"
    try:
        with open(view_desc_path, 'r', encoding='utf-8') as f:
            view_descriptions = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find view descriptions at {view_desc_path}")
        view_descriptions = {}

    # Format descriptions as a pretty-printed JSON string to insert into the prompt
    view_descriptions_str = json.dumps(view_descriptions, indent=4, ensure_ascii=False)

    # 3. Initialize the LLM
    model_name = config.get("configurable", {}).get("model_name", "gpt")
    llm = get_llm(model_name)

    # 4. Load prompt template and replace placeholders
    prompt_template_path = os.path.join('..', 'prompt_template', 'keyword_view_extraction_prompt.txt')
    try:
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template not found at {prompt_template_path}")
        return {"keywords": None}

    prompt = prompt.replace("[VIEW_DESCRIPTIONS_HERE]", view_descriptions_str)
    prompt = prompt.replace("[USER_QUESTION_HERE]", user_question)
    # print(prompt)
    
    # 5. Invoke LLM and parse the output
    response = llm.invoke(prompt)
    
    keywords = None
    try:
        keywords_text = response.content.strip()
        keywords_text = ommiting_think_block(keywords_text)
        keywords = extract_json_from_text(keywords_text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse keywords as JSON: {e}")

    print(f"---Extracted Keywords---\n{keywords}")

    return {"keywords": keywords}

def keyword_extraction(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Extracts keywords from the user's question using an LLM.
    """
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    user_question = user_messages[-1]["content"] if user_messages else ""

    model_name = config.get("configurable", {}).get("model_name", "gpt")
    llm = get_llm(model_name)  # Assumes you have a function to get your LLM instance

    with open(os.path.join('..', 'prompt_template', 'keyword_extraction_prompt.txt')) as f:
        prompt = f.read().replace("[USER_QUESTION_HERE]", user_question)
    
    # This is a simplified call; you might need to adapt it based on your LLM's API
    response = llm.invoke(prompt)
    
    try:
        keywords = response.content.strip()
        keywords = ommiting_think_block(keywords)
        keywords = extract_json_from_text(keywords)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse keywords as JSON: {e}")

    print(f"---Extracted Keywords---\n{keywords}")

    return {"keywords": keywords}


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
        keyword_view_extraction(state, config={"configurable": {"model_name": 'qwen-api'}})
        print()