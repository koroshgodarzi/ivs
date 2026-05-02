from langgraph.graph import StateGraph, END
from trainingAssistant.source_matching import source_matching_node, embedding_query, hallucinated_llm_embedding, re_embed_segment_node
from trainingAssistant.path_selection import path_selection_node
from trainingAssistant.chunk_retrieval import retrieve_by_path, retrieve_chunks_globally, retrieve_by_source
from trainingAssistant.response_generation import response_generation_node, active_response_generation_node, recover_best_attempt_node
from trainingAssistant.schema import AgentState
import json
import os
from datetime import datetime


def create_rag_graph_path():
    workflow = StateGraph(AgentState)

    workflow.add_node("source_matching", source_matching_node)
    workflow.add_node("path_selection", path_selection_node)
    workflow.add_node("chunk_retrieval", retrieve_by_path)
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
    workflow = StateGraph(AgentState)
    
    workflow.add_node("embedding_query", hallucinated_llm_embedding)
    workflow.add_node("chunk_retrieval", retrieve_chunks_globally)
    workflow.add_node("chunk_reranking", retrieve_by_source)
    workflow.add_node("response_generation", response_generation_node)

    workflow.set_entry_point("embedding_query")
    workflow.add_edge("embedding_query", "chunk_retrieval")
    workflow.add_edge("chunk_retrieval", "chunk_reranking")
    workflow.add_edge("chunk_reranking", "response_generation")
    workflow.add_edge("response_generation", END)

    return workflow.compile()


def main():
    # 1. Compile the graph
    output_folder = 'assistant_agent_test'
    app = create_rag_graph_path()

    primary_questions = [
    "چگونه می توانم قراردادها مرتبط با پروژه را در سامانه ثبت نمایم؟",
    "امکان ثبت مبلغ قرارداد با ارزی غیر از ارز پیش فرض پروژه وجود دارد؟",
    "برای یک قرارداد مبالغ به ارزهای مختلف قابل ثبت است؟",
    "برای هر پروژه می توانم دو قرارداد ثبت نمایم؟",
    "امکان ثبت تقویم جلسات فردی در سامانه وجود دارد؟ چگونه؟",
    "چطور می توانم اتاق جلسات را بصورت پیش فرض داشته باشم؟",
    "چگونه می توانم جلسات را دسته بندی نمایم؟",
    "می توانم در سامانه فرآیند یا گردش کاری تعریف کنم؟ چگونه؟",
    "برای فرمهای طراحی شده توسط خودم می توانم چرخه کاری طراحی نمایم؟"
]
    midlevel_questions = [
    "تبدیل مبلغ ارزی قرارداد به ارز پایه پروژه در سامانه چگونه انجام می شود؟",
    "چگونه می توانم قراردادها ثبت شده را به فعالیت های کاری متصل نمایم؟",
    "چگونه می توانم پیشرفت مالی پروژه را ثبت نمایم؟",
    "چطوری می توانم پرداختهای مربوط به پروژه را ثبت نمایم؟",
    "چگونه می توانم از جلسات همکاران و مدیران باخبر بشم؟",
    "چرخه های کاری برای چه فرآیندهایی قابل طراحی است؟",
    "چرخه های کاری طراحی شده در ماژول مدیریت فرم ها چه تفاوت و شباهتی با چرخه های کاری هسته دارد؟", 
    "قابلیت های ماژول قرارداد چیه؟", 
    "من قراردادهای چند ارزی رو چطور باید ثبت کنم؟"
]
    questions = primary_questions
    questions.extend(midlevel_questions)
        
    # Create directory if it doesn't exist
    output_dir = os.path.join('..', 'output', output_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # Unique filename for this specific run session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"run_{timestamp}.json")

    report = []
    
    for i, q in enumerate(questions):
        if i != 0: continue
        print(f"Processing question {i+1}/{len(questions)}...")
        
        initial_state = {
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
            final_state = app.invoke(initial_state)
            
            final_state['query_embedding'] = []
            final_state['retrieved_chunks'] = [
                {k: v for k, v in chunk.items() if k in ["text", "metadata", "score"]} 
                for chunk in final_state.get("retrieved_chunks", [])
            ]
            final_state['reranked_chunks'] = [
                {k: v for k, v in chunk.items() if k in ["text", "metadata", "score"]} 
                for chunk in final_state.get("reranked_chunks", [])
            ]
            

            # # Keys to EXCLUDE from the JSON (they contain large numpy arrays/vectors)
            # exclude_keys = {"query_embedding", "retrieved_chunks", "reranked_chunks"}
            
            # # Prepare data for saving
            # run_data = {
            #     "index": i,
            #     "timestamp": datetime.now().isoformat(),
            #     "data": {k: v for k, v in final_state.items() if k not in exclude_keys}
            # }
            
            report.append(final_state)
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            report.append({"index": i, "query": q, "error": str(e)})

    # Save the full report for this session
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Run completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()