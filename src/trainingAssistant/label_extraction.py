from utils import get_llm, extract_json_from_text, ommiting_think_block
from langchain_core.messages import HumanMessage

def label_extraction_node(state):
    query = state["query"]
    llm = get_llm("gpt-4o") # or your preferred model id

    prompt = f"""
    You are an expert classifier. Given a user query, identify the relevant categories/labels 
    needed to answer the question.
    Available Labels: ["Label: A", "Label: B", "Label: C"]
    
    User Query: {query}
    
    Respond strictly in JSON format:
    {{"labels": ["Label: A", "Label: B"]}}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = ommiting_think_block(response.content)
    parsed = extract_json_from_text(content)
    
    return {"labels": parsed.get("labels", [])}