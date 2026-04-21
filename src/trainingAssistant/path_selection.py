from utils import get_llm, extract_json_from_text
from langchain_core.messages import HumanMessage
from trainingAssistant.schema import AgentState
import chromadb

# 1. Initialize the client to point to your local storage
client = chromadb.PersistentClient(path="../data/chroma_db")

# 2. Get the collection (Replace "your_collection_name" with the name you used when creating it)
collection = client.get_collection(name="software_user_guide")


def fetch_chunks_by_source(selected_sources, collection):
    """
    selected_sources: list of filenames, e.g. ["Catalog.md", "PM_concepts.jsonl"]
    """
    if not selected_sources:
        return []

    # If searching for one source: {"source": "Catalog.md"}
    # If searching for multiple: {"source": {"$in": ["Catalog.md", "Docs.md"]}}
    if len(selected_sources) == 1:
        where_clause = {"source": selected_sources[0]}
    else:
        where_clause = {"source": {"$in": selected_sources}}

    results = collection.get(
        where=where_clause,
        include=["documents", "metadatas", "embeddings"]
    )

    # Note: No json.loads() needed anymore!
    chunks = []
    for i in range(len(results["ids"])):
        chunks.append(results["metadatas"][i]["path"])
    return chunks


def path_selection_node(state: AgentState):
    query = state["query"]
    tag = state["selected_tag"]
    candidate_paths = fetch_chunks_by_source(tag, collection)
    candidate_paths = list(set(candidate_paths))
    
    llm = get_llm("qwen_api")
    prompt = f"""
    User Query: {query}

    Available Document Paths:
    {candidate_paths}
    
    Based on the query, return a JSON list of the indices of the paths that are relevant.
    Example: {{"selected_indices": [0, 2]}}
    """  #     Category: {tag}

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = extract_json_from_text(response.content)
    indices = parsed.get("selected_indices", [])
    
    selected_paths = [candidate_paths[i] for i in indices if i < len(candidate_paths)]

    # print(selected_paths)    
    return {"selected_paths": selected_paths, "candidate_paths": candidate_paths}