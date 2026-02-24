from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from trainingAssistant.label_extraction import label_extraction_node
from trainingAssistant.label_search import label_search_node
from trainingAssistant.response_generation import response_generation_node

# Define the state object
class AgentState(TypedDict):
    query: str
    labels: List[str]
    retrieved_chunks: List[dict]
    answer: str

def create_rag_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("label_extraction", label_extraction_node)
    workflow.add_node("label_search", label_search_node)
    workflow.add_node("response_generation", response_generation_node)

    # Define Edges
    workflow.set_entry_point("label_extraction")
    workflow.add_edge("label_extraction", "label_search")
    workflow.add_edge("label_search", "response_generation")
    workflow.add_edge("response_generation", END)

    return workflow.compile()