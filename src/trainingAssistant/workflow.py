from langgraph.graph import StateGraph, END
from trainingAssistant.source_matching import source_matching_node, embedding_query, hallucinated_llm_embedding
from trainingAssistant.path_selection import path_selection_node
from trainingAssistant.chunk_retrieval import retrieve_by_path, retrieve_chunks_globally, retrieve_by_source
from trainingAssistant.response_generation import response_generation_node
from trainingAssistant.schema import AgentState
import json
import os
from datetime import datetime


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
    output_folder = 'assistant_agent_third_try'
    app = create_rag_graph()

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
        print(f"Processing question {i+1}/{len(questions)}...")
        
        initial_state = {
            "query": q,
            "query_embedding": [],
            "selected_sources": [],
            "candidate_paths": [],
            "selected_paths": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "answer": ""
        }

        try:
            final_state = app.invoke(initial_state)
            
            # Keys to EXCLUDE from the JSON (they contain large numpy arrays/vectors)
            exclude_keys = {"query_embedding", "retrieved_chunks", "reranked_chunks"}
            
            # Prepare data for saving
            run_data = {
                "index": i,
                "timestamp": datetime.now().isoformat(),
                "data": {k: v for k, v in final_state.items() if k not in exclude_keys}
            }
            
            report.append(run_data)
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            report.append({"index": i, "query": q, "error": str(e)})

    # Save the full report for this session
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Run completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()