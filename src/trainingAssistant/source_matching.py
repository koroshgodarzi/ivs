import numpy as np
from trainingAssistant.schema import AgentState
import os
import ollama
from pathlib import Path
import json
from utils import get_llm

def cosine_similarity(v1, v2):
    """Calculates the cosine similarity between two vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

def re_embed_segment_node(state: AgentState):
    # Take the last 250 tokens (or the last segment)
    segment_to_embed = state["last_generated_segment"]
    
    # Use your existing embedding logic
    user_embedding = get_query_embedding(segment_to_embed)
    
    return {"query_embedding": user_embedding}

def get_query_embedding(query: str):
    """Encapsulates the logic for fetching embeddings from Ollama."""
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    resp = ollama.embed(model=embed_model, input=query)
    return np.array(resp["embeddings"][0], dtype=np.float32)

def hallucinated_llm_embedding(state: AgentState):
    query = state["query"]
    
    llm = get_llm("ollama", max_tokens=250)
    prompt = f"""
    Answer the user's question. Imagine that you have the knowledge.
    
    Question: {query}
    """
    
    response = llm.invoke(prompt, logprobs=True)
    logprobs_data = response.response_metadata.get("logprobs", {}).get("content", [])
    current_segment_logprobs = [token_info.get("logprob", 0) for token_info in logprobs_data]
    min_logprob = min(current_segment_logprobs) if current_segment_logprobs else 0
    print(min_logprob)
    print(current_segment_logprobs)
    user_embedding = get_query_embedding(response.content)
    return {"query_embedding": user_embedding}

def embedding_query(state: AgentState):
    query = state["query"]

    user_embedding = get_query_embedding(query)
    return {"query_embedding": user_embedding}
    

def source_matching_node(state: AgentState, k=3):
    query = state["query"]
    
    # 1. Use the new helper function to get the embedding
    user_embedding = get_query_embedding(query)

    # 2. Setup paths
    data_dir = Path(__file__).parent.parent.parent / "data"
    embeddings_path = data_dir / "embeddings_md_disc.npy"
    source_mapping_path = data_dir / "index_md_disc.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")
    if not source_mapping_path.exists():
        raise FileNotFoundError(f"Source index mapping file not found at: {source_mapping_path}")

    # 3. Load stored data
    embeddings_array = np.load(str(embeddings_path))
    with open(source_mapping_path, "r", encoding="utf-8") as f:
        index_to_source = json.load(f)

    # 4. Calculate similarities using the cosine_similarity function
    # Note: We apply the function across the rows of the stored embeddings
    similarities = [cosine_similarity(user_embedding, row) for row in embeddings_array]
    similarities = np.array(similarities)

    # 5. Get top K indices
    top_k_indices = np.argsort(similarities)[::-1][:k]

    # 6. Map indices to source names
    sources = []
    for idx in top_k_indices:
        key = str(int(idx))
        if key in index_to_source:
            source_name = index_to_source[key]
            if source_name not in sources:
                sources.append(source_name)

    # Return updated state with 'source' naming convention
    return {"selected_sources": sources, "query_embedding": user_embedding}