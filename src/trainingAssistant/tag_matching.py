import numpy as np
from trainingAssistant.schema import AgentState
import os
import ollama
from pathlib import Path
import json


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def tag_matching_node(state: AgentState, k=3):
    query = state["query"]

    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    resp = ollama.embed(model=embed_model, input=query)
    user_embedding = np.array(resp["embeddings"][0], dtype=np.float32) 

    embeddings_path = Path(__file__).parent.parent.parent / "data" / "embeddings_md_disc.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")

    embeddings_array = np.load(str(embeddings_path))
    
    user_norm = user_embedding / np.linalg.norm(user_embedding)
    array_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)

    similarities = np.dot(array_norm, user_norm)

    top_k_indices = np.argsort(similarities)[::-1][:k]

    md_mapping_path = Path(__file__).parent.parent.parent / "data" / "index_md_disc.json"

    if not md_mapping_path.exists():
        raise FileNotFoundError(
            f"View index mapping file not found at: {md_mapping_path}"
        )

    with open(md_mapping_path, "r", encoding="utf-8") as f:
        index_to_md = json.load(f)  

    tags = []
    for idx in top_k_indices:
        key = str(int(idx))  
        if key in index_to_md:
            md = index_to_md[key]
            if md not in tags:
                tags.append(md)

    # print(tags)
    return {"selected_tag": tags, "query_embedding": user_embedding}