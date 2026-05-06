from utils import get_llm, extract_json_from_text
from langchain_core.messages import HumanMessage
from trainingAssistant.schema import AgentState


def fetch_chunks_by_source(selected_sources, collection):
    """
    selected_sources: list of filenames, e.g. ["Catalog.md", "PM_concepts.jsonl"]
    """
    if not selected_sources:
        return []

    if len(selected_sources) == 1:
        where_clause = {"source": selected_sources[0]}
    else:
        where_clause = {"source": {"$in": selected_sources}}

    results = collection.get(
        where=where_clause,
        include=["documents", "metadatas", "embeddings"]
    )

    chunks = []
    for i in range(len(results["ids"])):
        chunks.append(results["metadatas"][i]["path"])
    return chunks


def path_selection_node(state: AgentState, collection):
    query = state["query"]
    tag = state["selected_sources"]
    candidate_paths = fetch_chunks_by_source(tag, collection)
    candidate_paths = list(set(candidate_paths))

    llm = get_llm("ollama")
    prompt = f"""
    User Query: {query}

    Available Document Paths:
    {candidate_paths}

    Based on the query, return a JSON list of the indices of the paths that are relevant.
    Example: {{"selected_indices": [0, 2]}}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = extract_json_from_text(response.content)
    indices = parsed.get("selected_indices", [])

    selected_paths = [candidate_paths[i] for i in indices if i < len(candidate_paths)]

    return {"selected_paths": selected_paths, "candidate_paths": candidate_paths}
