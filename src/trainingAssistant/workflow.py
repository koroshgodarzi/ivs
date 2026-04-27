from langgraph.graph import StateGraph, END
from trainingAssistant.source_matching import source_matching_node, embedding_query
from trainingAssistant.path_selection import path_selection_node
from trainingAssistant.chunk_retrieval import retrieve_by_path, retrieve_chunks_globally, retrieve_by_source
from trainingAssistant.response_generation import response_generation_node
from trainingAssistant.schema import AgentState
import json
import os


def create_rag_graph():
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


def create_rag_graph_2():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("embedding_query", embedding_query)
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
    app = create_rag_graph_2()

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
    report = []
    for i, q in enumerate(questions):
        if i <= 16: continue
        print(i)
        # 2. Define the initial state
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

        # 3. Run the graph
        # Using .stream or .invoke
        # print("--- Starting Workflow ---\n")
        # for output in app.stream(initial_state):
        #     # This prints which node just finished and its output
        #     for key, value in output.items():
        #         print(f"Node '{key}' finished.")
        #         if "selected_tag" in value:
        #             print(f"  Tag Found: {value['selected_tag']}")
        #         if "selected_paths" in value:
        #             print(f"  Paths Selected: {value['selected_paths']}")
        
        # 4. Get the final result
        final_state = app.invoke(initial_state)
        exclude_keys = {"query_embedding", "retrieved_chunks", "reranked_chunks"}
        part_data = {k: v for k, v in final_state.items() if k not in exclude_keys}
        report.append(part_data)
        # print("\n--- Final Answer ---")
        print(f"query: {final_state['query']}")
        # print(f"selected_tag: {final_state['selected_tag']}")
        # print(f"candidate_paths: {final_state['candidate_paths']}")
        # print(f"selected_paths: {final_state['selected_paths']}")
        # print(f"retrieved_chunks: {final_state['retrieved_chunks']}")
        print(f"answer: {final_state['answer']}")
        # print("Question:")
        # print(q)
        # print("Answer")
        # print(final_state["answer"])

    with open(os.path.join('..', 'output', 'assistant_agent_second_try', 'result.json'), "w", encoding="utf-8") as f:
        json.dump(part_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()    