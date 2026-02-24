from utils import get_llm, ommiting_think_block
from langchain_core.messages import HumanMessage

def response_generation_node(state):
    query = state["query"]
    chunks = state["retrieved_chunks"]
    
    # Format the context for the LLM
    context_text = "\n\n".join([f"Source ({c['label']}): {c['text']}" for c in chunks])
    
    llm = get_llm("gpt-4o")
    
    prompt = f"""
    You are a helpful company assistant. Use the provided context to answer the user query accurately.
    If the context doesn't contain the answer, say you don't know.
    
    Context:
    {context_text}
    
    Query: {query}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    final_answer = ommiting_think_block(response.content)
    
    return {"answer": final_answer}