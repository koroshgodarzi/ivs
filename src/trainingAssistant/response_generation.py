from utils import get_llm
from trainingAssistant.schema import AgentState


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