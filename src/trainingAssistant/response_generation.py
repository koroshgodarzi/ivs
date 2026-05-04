from utils import get_llm, get_truncated_history
from trainingAssistant.schema import AgentState
import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

def active_response_generation_node(state: AgentState, n_chunks=5):
    query = state["query"]
    chunks = state.get("reranked_chunks", [])[:n_chunks]
    print(chunks.get("score"))
    verified_answer = state.get("answer", "")  # Only high-confidence text goes here
    steps = state.get("steps_taken", 0)
    
    # Track the best complete or partial attempt for fallback
    best_min_lp = state.get("best_min_logprob", -999.0)
    best_answer_so_far = state.get("best_answer", "")

    context = "\n\n".join([f"Source ({c['metadata']['path']}): {c['text']}" for c in chunks])
    
    llm = get_llm("ollama", max_tokens=250) 
    
    # We always continue from the last VERIFIED part of the answer
    prefix_instruction = ""
    if verified_answer:
        prefix_instruction = f"Existing high-confidence response: {verified_answer}\nCONTINUE the response naturally."

    prompt = f"""
    Answer the user's question based strictly on the context. 
    Context:
    {context}
    
    Question: {query}
    {prefix_instruction}
    """
    
    response = llm.invoke(prompt, logprobs=True)
    content = response.content
    logprobs_data = response.response_metadata.get("logprobs", {}).get("content", [])
    finish_reason = response.response_metadata.get("finish_reason", "stop")
    print(finish_reason)

    # 1. Calculate confidence for THIS specific segment
    current_segment_logprobs = [token_info.get("logprob", 0) for token_info in logprobs_data]
    min_logprob = min(current_segment_logprobs) if current_segment_logprobs else 0
    
    THRESHOLD = -0.7 
    low_confidence = min_logprob < THRESHOLD
    is_finished = (finish_reason == "stop")

    # 2. Update Fallback tracking
    # Even if this segment is low confidence, it might be "less bad" than others.

    if min_logprob > best_min_lp:
        best_min_lp = min_logprob
        best_answer_so_far = content

    # 3. DECISION LOGIC
    if low_confidence:
        # LOW CONFIDENCE: Discard content from the final answer.
        # But return it as 'last_generated_segment' so 're_embed_segment_node' can use it for search.
        return {
            "answer": verified_answer, # Do NOT append
            "last_generated_segment": content, # Use for re-embedding
            "low_confidence": True,
            "is_finished": False,
            "steps_taken": steps + 1,
            "best_min_logprob": best_min_lp,
            "best_answer": best_answer_so_far,
            "best_is_finished": is_finished
        }
    else:
        # HIGH CONFIDENCE: This segment is good! Append it to the verified answer.
        new_verified_answer = verified_answer + content
        
        # If we are hitting the step limit now, check if we actually finished.
        # If we reached step 3 but haven't finished the sentence, the next loop will
        # trigger the 'steps >= 3' check in the router and we'll end.
        return {
            "answer": new_verified_answer, # APPENDED
            "last_generated_segment": content,
            "low_confidence": False,
            "is_finished": is_finished,
            "steps_taken": 0,
            "best_min_logprob": best_min_lp,
            "best_answer": best_answer_so_far,
            "best_is_finished": False
        }


def recover_best_attempt_node(state: AgentState):
    """
    Node that resets the state to the best attempt found during the search.
    """
    return {
        "answer": state["answer"] + state["best_answer"],
        # "reranked_chunks": state["best_chunks"], # Use the context that worked best
        "is_finished": state["best_is_finished"],
        "steps_taken": 0, # Reset to allow finishing
        "low_confidence": False # Assume we proceed to finish regardless
    }


def response_generation_node(state: AgentState, config: RunnableConfig):
    # 1. Configuration
    latest_query = state["messages"][-1].content
    model_name = config["configurable"].get("model_name", "ollama")
    n_chunks = config["configurable"].get("n_chunks", 5)
    llm = get_llm(model_name)

    # 2. Prepare Context from RAG
    chunks = state.get("reranked_chunks", [])[:n_chunks]
    context_str = "\n\n".join([f"Source ({c['metadata'].get('path', 'unknown')}): {c['text']}" for c in chunks])
    
    # 3. Prepare History (truncated to leave room for context and answer)
    history = get_truncated_history(state.get("messages", []), 1500)

    prompt = f"""
    Answer the user's question based strictly on the context provided. 
    Use the chat history to understand references (like 'he', 'it', 'previous step').
    If the question is a "how" question, go into detail. If "can", answer shortly.

    History:
    {history}

    Context:
    {context_str}
    
    Current Question: {latest_query}
    """
    
    response = llm.invoke(prompt)
    
    # Return the message so LangGraph appends it to history, 
    # and also update the 'answer' string for the UI.
    return {
        "messages": [response], 
        "answer": response.content
    }