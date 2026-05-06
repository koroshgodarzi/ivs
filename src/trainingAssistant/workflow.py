from langgraph.graph import StateGraph, END
from trainingAssistant.source_matching import source_matching_node, embedding_query, hallucinated_llm_embedding, re_embed_segment_node
from trainingAssistant.path_selection import path_selection_node
from trainingAssistant.chunk_retrieval import retrieve_by_path, retrieve_chunks_globally, retrieve_by_source
from trainingAssistant.response_generation import response_generation_node, active_response_generation_node, recover_best_attempt_node
from trainingAssistant.schema import AgentState
import json
import os
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
import sqlite3
from langchain_core.messages import HumanMessage, AIMessage
import chromadb
from functools import partial


def create_rag_graph_path():
    workflow = StateGraph(AgentState)

    client = chromadb.PersistentClient(path="../data/chroma_db")
    collection = client.get_collection(name="software_user_guide")

    workflow.add_node("source_matching", source_matching_node)
    workflow.add_node("path_selection", partial(path_selection_node, collection=collection))
    workflow.add_node("chunk_retrieval", partial(retrieve_by_path, collection=collection))
    workflow.add_node("response_generation", response_generation_node)

    workflow.set_entry_point("source_matching")
    workflow.add_edge("source_matching", "path_selection")
    workflow.add_edge("path_selection", "chunk_retrieval")
    workflow.add_edge("chunk_retrieval", "response_generation")
    workflow.add_edge("response_generation", END)

    return workflow.compile()


def decide_next_step(state: AgentState):
    # If max steps reached, always end (the node already picked the best_answer)
    if state.get("steps_taken", 0) >= 3:
        # If the best version we found is already finished, we end.
        if state.get("best_is_finished"):
            return "end_with_best_fallback"
        # If not finished, go to recovery node to reset and continue.
        return "recover_and_finish"

    # Condition 1: Low confidence -> Re-embed (Loop to Retrieval)
    if state.get("low_confidence"):
        return "re_embed"
    
    # Condition 2: High confidence but NOT finished -> Continue (Loop to Generation)
    if not state.get("is_finished"):
        return "continue"

    # Condition 3: High confidence AND finished
    return END

def finalize_best_answer(state: AgentState):
    """Ensure the final answer is the best one found before ending."""
    return {"answer": state["answer"] + state["best_answer"]}

def create_active_rag_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("initial_embedding", hallucinated_llm_embedding)
    workflow.add_node("chunk_retrieval", retrieve_chunks_globally)
    workflow.add_node("chunk_reranking", retrieve_by_source)
    workflow.add_node("active_generation", active_response_generation_node)
    workflow.add_node("re_embed_segment", re_embed_segment_node)
    workflow.add_node("recover_best_attempt", recover_best_attempt_node)
    workflow.add_node("finalize_best", finalize_best_answer)

    workflow.set_entry_point("initial_embedding")
    workflow.add_edge("initial_embedding", "chunk_retrieval")
    workflow.add_edge("chunk_retrieval", "chunk_reranking")
    workflow.add_edge("chunk_reranking", "active_generation")

    workflow.add_conditional_edges(
        "active_generation",
        decide_next_step,
        {
            "re_embed": "re_embed_segment",
            "continue": "active_generation",
            "recover_and_finish": "recover_best_attempt", # Reverts context & answer
            "end_with_best_fallback": "finalize_best",    # Sets answer to best and ends
            END: END
        }
    )
    
    # Logic for recovered path
    workflow.add_edge("recover_best_attempt", "active_generation")
    workflow.add_edge("finalize_best", END)
    
    # Logic for re-embedding path
    workflow.add_edge("re_embed_segment", "chunk_retrieval")

    return workflow.compile()

    
def create_rag_graph():
    # 1. Initialize Memory for persistence
    memory = MemorySaver()

    # Initialize ChromaDB client and collection once here
    client = chromadb.PersistentClient(path="../data/chroma_db")
    collection = client.get_collection(name="software_user_guide")

    workflow = StateGraph(AgentState)

    # Use functools.partial to bind the collection to the node functions
    workflow.add_node("embedding_query", hallucinated_llm_embedding)
    workflow.add_node("chunk_retrieval", partial(retrieve_chunks_globally, collection=collection))
    workflow.add_node("chunk_reranking", partial(retrieve_by_source, collection=collection))
    workflow.add_node("response_generation", response_generation_node)

    workflow.set_entry_point("embedding_query")
    workflow.add_edge("embedding_query", "chunk_retrieval")
    workflow.add_edge("chunk_retrieval", "chunk_reranking")
    workflow.add_edge("chunk_reranking", "response_generation")
    workflow.add_edge("response_generation", END)

    return workflow.compile(checkpointer=memory)


def message_to_dict(message):
    """Converts LangChain messages to a JSON-serializable format."""
    if isinstance(message, HumanMessage):
        return {"role": "human", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"role": "ai", "content": message.content}
    return {"role": "other", "content": str(message.content)}

def main():
    # 1. Compile the graph
    # Note: Ensure create_rag_graph() now uses SqliteSaver as discussed previously
    app = create_rag_graph()
    
    # 2. Define the questions (Follow-ups included)
#     primary_questions = [
#     "چگونه می توانم قراردادها مرتبط با پروژه را در سامانه ثبت نمایم؟",
#     "امکان ثبت مبلغ قرارداد با ارزی غیر از ارز پیش فرض پروژه وجود دارد؟",
#     "برای یک قرارداد مبالغ به ارزهای مختلف قابل ثبت است؟",
#     "برای هر پروژه می توانم دو قرارداد ثبت نمایم؟",
#     "امکان ثبت تقویم جلسات فردی در سامانه وجود دارد؟ چگونه؟",
#     "چطور می توانم اتاق جلسات را بصورت پیش فرض داشته باشم؟",
#     "چگونه می توانم جلسات را دسته بندی نمایم؟",
#     "می توانم در سامانه فرآیند یا گردش کاری تعریف کنم؟ چگونه؟",
#     "برای فرمهای طراحی شده توسط خودم می توانم چرخه کاری طراحی نمایم؟"
# ]
#     midlevel_questions = [
#     "تبدیل مبلغ ارزی قرارداد به ارز پایه پروژه در سامانه چگونه انجام می شود؟",
#     "چگونه می توانم قراردادها ثبت شده را به فعالیت های کاری متصل نمایم؟",
#     "چگونه می توانم پیشرفت مالی پروژه را ثبت نمایم؟",
#     "چطوری می توانم پرداختهای مربوط به پروژه را ثبت نمایم؟",
#     "چگونه می توانم از جلسات همکاران و مدیران باخبر بشم؟",
#     "چرخه های کاری برای چه فرآیندهایی قابل طراحی است؟",
#     "چرخه های کاری طراحی شده در ماژول مدیریت فرم ها چه تفاوت و شباهتی با چرخه های کاری هسته دارد؟", 
#     "قابلیت های ماژول قرارداد چیه؟", 
#     "من قراردادهای چند ارزی رو چطور باید ثبت کنم؟"
# ]
#     questions = primary_questions
#     questions.extend(midlevel_questions)
        
    questions = [
    "در ماژول مالی و قراردادها چه کارهایی می توانم انجام دهم؟",
    "چه مدل قراردادی را می توانم در این ماژول ثبت کنم؟",
    "انواع قرارداد در نرم افزار وجود دارد؟",
    "چه فیلدهایی را می توانم در قرارداد ثبت نمایم؟",
    "فایل های قراردادها را چه کار کنم؟",
    "قراردادهای پروژه ای و غیر پروژه ای را چکار کنم؟",
    "قراردادهای غیر پروژه را چگونه می توانم ثبت نمایم؟",
    "پروژه صوری را چگونه ثبت نمایم؟",
    "الحاقیه ها را کجا می توانم ثبت نمایم؟",
    "قرارداد را به ساختار شکست وصل کنم یعنی چه؟",
    "قرارداد را به ساختار شکست را چگونه وصل کنم؟",
    "چه فایده ای دارد؟",
    "CPM را خودم باید تعریف کنم یا اتوماتیک انجام می‌شود؟",
    "صورت وضعیت های پروژه و قرارداد را چطور می‌توانم ثبت کنم؟",
    "جزییات صورت وضعیت را هم می‌توانم ثبت کنم؟",
    "ردیف‌های صورت وضعیت چطور؟",
    "این صورت وضعیت را چطور می‌توانم بگردانماش؟",
    "دسترسی های نقش قرارداد کجا تعیین می‌شود؟",
    "من می‌توانم کاربری را مشخص کنم که فقط بتواند قراردادها را ببیند؟",
    "برای بقیه ی قسمت ها هم دسترسی چطور تعیین می‌شود؟",
    "در بررسی صورت وضعیت چطور دسترسی ها تعیین می‌شود؟"
    ]

    output_folder = 'assistant_agent_follow_up_first_try'
    output_dir = os.path.join('..', 'output', output_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"run_{timestamp}.json")

    # 3. Configuration with a consistent thread_id for history
    thread_id = f"test_session_{timestamp}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "model_name": "ollama",
            'n_chunks': 5
        }
    }

    report = []
    
    for i, q in enumerate(questions):
        print(f"\n--- Processing question {i+1}/{len(questions)} ---")
        print(f"Query: {q}")
        
        # In LangGraph with persistence, we only need to send the NEW message.
        # The checkpointer automatically merges it into existing history for this thread_id.
        initial_state = {
            "messages": [HumanMessage(content=q)],
            "query": q,
            "query_embedding": [],
            "selected_sources": [],
            "candidate_paths": [],
            "selected_paths": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "answer": "",
            "steps_taken": 0,
            "low_confidence": False, 
            "is_finished": False,
            "last_generated_segment": "",
            "best_answer": "",
            "best_is_finished": False,
            "best_min_logprob": -1000
        }

        try:
            # 4. Invoke the graph with the persistent config
            final_state = app.invoke(initial_state, config=config)
            
            # 5. Prepare state for JSON serialization
            # We must convert BaseMessage objects to dicts and remove numpy arrays
            serializable_state = final_state.copy()
            
            serializable_state['messages'] = [message_to_dict(m) for m in final_state.get("messages", [])]
            serializable_state['query_embedding'] = [] # Remove large embedding vectors
            
            # Clean up chunks for readable JSON
            for key in ['retrieved_chunks', 'reranked_chunks']:
                serializable_state[key] = [
                    {k: v for k, v in chunk.items() if k in ["text", "metadata", "score"]} 
                    for chunk in final_state.get(key, [])
                ]
                        
            report.append(serializable_state)
            print(f"Response: {final_state['answer']}...")
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            report.append({"index": i, "query": q, "error": str(e)})

    # 6. Save the full report
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nRun completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()