from utils import get_llm
from trainingAssistant.schema import AgentState


def response_generation_node(state: AgentState):
    query = state["query"]
    chunks = state["retrieved_chunks"]
    
    context = "\n\n".join([f"Source ({c['metadata']['path']}): {c['text']}" for c in chunks])
    
    llm = get_llm("qwen_api")
    prompt = f"""
    Answer the user's question based strictly on the context provided.
    
    Context:
    {context}
    
    Question: {query}
    """
    # print(prompt)
    
    response = llm.invoke(prompt)
    return {"answer": response.content}