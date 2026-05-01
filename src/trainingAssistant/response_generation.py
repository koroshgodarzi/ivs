from utils import get_llm
from trainingAssistant.schema import AgentState
import numpy as np

def active_response_generation_node(state: AgentState):
    query = state["query"]
    chunks = state.get("reranked_chunks", [])
    current_answer = state.get("answer", "")
    steps = state.get("steps_taken", 0)
    
    # Context from retrieved chunks
    context = "\n\n".join([f"Source ({c['metadata']['path']}): {c['text']}" for c in chunks])
    
    # We use a slightly higher max_token for the 'segment' to find low-confidence areas
    llm = get_llm("qwen_api", max_tokens=150) 
    
    # If we already have a partial answer, we tell the LLM to continue
    prefix = f"Existing partial response: {current_answer}" if current_answer else ""
    
    prompt = f"""
    Answer the user's question based strictly on the context. 
    {prefix}
    
    Context:
    {context}
    
    Question: {query}
    
    Continue the response naturally. If the existing partial response is empty, start from the beginning.
    """
    
    # Request logprobs from the API
    response = llm.invoke(prompt, logprobs=True)
    
    logprobs_data = response.response_metadata.get("logprobs", {}).get("content", [])
    
    # Define a confidence threshold (log probability)
    # -0.5 to -1.0 is a common range for 'low confidence'
    THRESHOLD = -0.7 
    
    content = response.content
    low_confidence_found = False
    cut_index = len(content)

    # Check tokens for low probability
    for i, token_info in enumerate(logprobs_data):
        if token_info.get("logprob", 0) < THRESHOLD:
            # We found a low confidence token!
            # We allow it to generate a bit more (already in content) then stop
            low_confidence_found = True
            # Optional: cut the content at the point of low confidence to re-verify
            # Or just flag it to trigger a new search for the NEXT segment
            break

    new_answer = current_answer + " " + content
    
    # If we've looped too many times, stop anyway
    if steps >= 3:
        low_confidence_found = False

    return {
        "answer": new_answer,
        "last_generated_segment": content, # Used for the next embedding query
        "low_confidence": low_confidence_found,
        "steps_taken": steps + 1
    }

def response_generation_node(state: AgentState, n_chunks=5):
    query = state["query"]
    chunks = state["reranked_chunks"][:n_chunks]
    
    context = "\n\n".join([f"Source ({c['metadata']['path']}): {c['text']}" for c in chunks])
    
    llm = get_llm("qwen_api")
    prompt = f"""
    Answer the user's question based strictly on the context provided. 
    If you cannot answer the question say you are not able to. 
    Do not mention about the provided context.
    If the question is a "how" question, i.e. the user wants to the procedure to do something go to details. If it is a "can" question, i.e. the user wants to know if a feature exists or they have the ability/access to do something, etc. answer in short.
    
    Context:
    {context}
    
    Question: {query}
    """
    # print(prompt)
    
    response = llm.invoke(prompt)
    return {"answer": response.content}