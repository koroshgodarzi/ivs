from NL2SQL.schema import GraphState
from typing import Optional, List
import json
import os
import numpy as np
from pathlib import Path
import ollama 
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv
load_dotenv()

def schema_retriever(state: GraphState, config: RunnableConfig,
    top_k_schemas: int = 3,
) -> GraphState:
    """
    Initial node to identify relevant view names based on the user's prompt.
    Saves only the names to the state, omitting file loading.
    """
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    user_prompt = user_messages[-1]["content"] if user_messages else ""

    relevant_views = get_relevant_view_names(user_prompt=user_prompt, k=top_k_schemas)

    state["retrieved_schema"] = relevant_views

    print(f"Relevant views identified and stored: {relevant_views}")
    return state


def get_relevant_view_names(
    user_prompt: Optional[str] = None,
    k: int = 5,
    schema_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Determines which database views are relevant.
    """

    view_mapping_path = Path(__file__).parent.parent.parent / "data" / "index_disc.json"

    if user_prompt:
        try:
            top_k_indices = find_most_similar_embeddings(user_prompt, k=k)

            if not view_mapping_path.exists():
                raise FileNotFoundError(
                    f"View index mapping file not found at: {view_mapping_path}"
                )

            with open(view_mapping_path, "r", encoding="utf-8") as f:
                index_to_view = json.load(f)  

            schema_names = []
            for idx in top_k_indices:
                key = str(int(idx))  
                if key in index_to_view:
                    view = index_to_view[key]
                    if view not in schema_names:
                        schema_names.append(view)

            if not schema_names:
                raise ValueError(
                    f"No matching views found for indices: {top_k_indices}"
                )

        except Exception as e:
            raise RuntimeError(
                f"Failed to identify relevant schemas: {str(e)}"
            ) from e

    elif not schema_names:
        raise ValueError("Either user_prompt or schema_names must be provided")

    return schema_names


def find_most_similar_embeddings(user_prompt: str, k: int = 5, embeddings_path: str = None) -> np.ndarray:
    """
    Embed a user prompt using Ollama (embeddinggemma) and find the k most similar
    embeddings from a pre-computed numpy array.
    """

    if embeddings_path is None:
        embeddings_path = Path(__file__).parent.parent.parent / "data" / "embeddings_disc.npy"
    else:
        embeddings_path = Path(embeddings_path)

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")

    embeddings_array = np.load(str(embeddings_path))

    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    resp = ollama.embed(model=embed_model, input=user_prompt)
    user_embedding = np.array(resp["embeddings"][0], dtype=np.float32) 

    user_norm = user_embedding / np.linalg.norm(user_embedding)
    array_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)

    similarities = np.dot(array_norm, user_norm)

    top_k_indices = np.argsort(similarities)[::-1][:k]
    return top_k_indices


if __name__ == "__main__":

    # test_prompts = ['Is there relation between location and date info of projects.']
    # test_prompts = [
    #     "تعداد پروژه های فعال من چند تاست؟",
    #     "چند تا پروژه در حالت 'در حال مذاکره' دارم؟",
    #     "ناصر اسدی مدیر چند تا پروژه در وضعیت در حال اجراست؟",
    #     "ناصر اسدی مدیر چند تا پروژه فعاله؟",
    #     "لیست پروژه هایی که ناظر یا مشاور دارن",
    #     "کدوم یکی از پروژه های EPC من پیشرفت واقعی بیشتری دارن؟",
    #     "جمع رقم قراردادهای خاتمه یافته عمومی رو به تفکیک سال بده.",
    #     "لیست قراردادهای تاخیر دار رو به ترتیب از بدترین وضعیت بده",
    #     "یه پروژه جدید داره میاد. به نظرت بین مدیر پروژه های قبلی، به کدوم یکی بدمش؟ هم بحث تعداد پروژه هایی که نفر دستشه رو در نظر بگیر هم بحث تاخیر پروژه های قبلی",
    #     "آیا ارتباطی بین محل اجرای پروژه با احتمال تاخیرش دیده میشه؟",
    #     "لیست قراردادهای فسخ شده رو بده",
    #     "جمع مبلغ و تعداد قراردادهای جاری رو بده",
    #     "جمع قراردادهای هر سال از 90 به اینور رو بده",
    #     "لیست قراردادهایی که الحاقیه دارن رو بده",
    #     "قراردادهایی که صورت وضعیت نخوردن ولی پرداخت داشتن",
    #     "جمع مبالغی که صورت وضعیت شده اما هنوز پرداخت نشده برای قراردادهای جاری",
    #     "میانگین درصد الحاقیه ها نسبت به رقم قرارداد به تفکیک سال",
    #     "وضعیت کدوم قراردادم خیلی خرابه؟ میتونی از میزان پیشرفت فیزیکی نسبت به مبلغ پرداخت شده و همچنین مبلغ اولیه قرارداد برای معیار استفاده کنی",
    #     "کدوم مدیر پروژه قراردادهاش رو بهتر مدیریت کرده؟ میتونی یه معیار از تعداد قراردادهای خاتمه یافته به عنوان امتیاز مثبت، فسخ شده به عنوان امتیاز منفی، و انحراف رقم پرداخت شده نهایی نسبت به رقم اولیه به عنوان امتیاز منفی شکل بدی و بر اون اساس قضاوت کنی"
    # ]
    test_prompts = [
    "How many active projects do I have?",
    "How many projects are in the 'under negotiation' status?",
    "How many projects is Naser Asadi managing that are currently in execution?",
    "How many active projects is Naser Asadi managing?",
    "List the projects that have a supervisor or consultant.",
    "Which of my EPC projects have greater actual progress?",
    "Provide the total value of completed public contracts, broken down by year.",
    "List the delayed contracts, ordered from the worst situation.",
    "A new project is coming up. Based on the previous project managers, who do you think I should assign it to? Consider both the number of projects each person is handling and delays in their previous projects.",
    "Is there a relationship between the project location and the likelihood of delays?",
    "Provide a list of terminated (cancelled) contracts.",
    "Provide the total amount and count of ongoing contracts.",
    "Provide the total value of contracts for each year from 2011 onward.",
    "Provide a list of contracts that have addendums.",
    "Contracts that have had payments but no progress statements issued.",
    "The total amount that has been invoiced but not yet paid for ongoing contracts.",
    "The average percentage of addendums relative to the original contract value, broken down by year.",
    "Which of my contracts is in very bad condition? You can use criteria such as physical progress compared to the amount paid, as well as the original contract value.",
    "Which project manager has managed their contracts better? You can define a metric using the number of completed contracts as a positive score, terminated contracts as a negative score, and deviation of the final paid amount from the original contract value as a negative score, then judge based on that."
]


    for tp in test_prompts:
        print(tp)
        state = GraphState(messages=[{"role": "user", "content": tp}], generated_query=None, query_results=None, error_message=None, summary_context=None, retry_count=0, validation_result=None, retrieved_schema=['vw_Contracts'])
        schema_retriever(state, 3)
        print()