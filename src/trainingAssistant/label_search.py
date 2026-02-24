# Mock Data - In a real app, this would be your vector store or JSON database
MOCK_CHUNKS = [
    {"label": "Label: A", "index": 45, "text": "Company policy on remote work..."},
    {"label": "Label: B", "index": 102, "text": "Health insurance benefits details..."},
    {"label": "Label: C", "index": 87, "text": "IT security protocols for passwords..."}
]

def label_search_node(state):
    extracted_labels = state["labels"]
    
    # Filter: Only keep chunks that match the extracted labels
    filtered_chunks = [
        chunk for chunk in MOCK_CHUNKS 
        if chunk["label"] in extracted_labels
    ]
    
    # (Optional) Implement Ranking logic here if you have many chunks
    # For now, we take all relevant chunks as 'Top Relevant Chunks'
    return {"retrieved_chunks": filtered_chunks}