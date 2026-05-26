import chromadb
from NL2SQL.schema import GraphState
from langchain_core.runnables import RunnableConfig
import numpy as np
import ollama
import os
import pickle
from datasketch import MinHash


def identify_views_from_entities(
    entities: list, 
    get_query_embedding_func,
    npy_path: str = '../data/view_embeddings.npy', 
    names_path: str = '../data/view_names.pkl',
    top_k: int = 2,
    threshold: float = 0.4
) -> set:
    """
    Computes cosine similarity between entity embeddings and view descriptions 
    to identify the most relevant database views.
    """
    chosen_views = set()
    if not entities or not os.path.exists(npy_path) or not os.path.exists(names_path):
        return chosen_views
    
    try:
        view_embeddings = np.load(npy_path)
        with open(names_path, 'rb') as f:
            view_names = pickle.load(f)
        
        if len(view_embeddings) != len(view_names):
            print("Warning: View embeddings count does not match view names.")
            return chosen_views

        for entity in entities:
            entity_emb = np.array(get_query_embedding_func(entity))
            
            # Normalize vectors to calculate Cosine Similarity
            entity_norm = np.linalg.norm(entity_emb)
            if entity_norm > 0:
                entity_emb = entity_emb / entity_norm
                
            view_norms = np.linalg.norm(view_embeddings, axis=1)
            view_norms[view_norms == 0] = 1.0
            normalized_views = view_embeddings / view_norms[:, np.newaxis]
            
            similarities = np.dot(normalized_views, entity_emb)
            
            # Retrieve the top-K matches
            k = min(top_k, len(view_names))
            top_indices = np.argsort(similarities)[::-1][:k]
            
            for idx in top_indices:
                if similarities[idx] >= threshold:
                    chosen_views.add(view_names[idx])
                    
    except Exception as e:
        print(f"Warning: Failed to identify views from entities: {e}")
        
    return chosen_views


def retrieve_columns_by_attributes(
    attributes: list, 
    chosen_views: set, 
    collection, 
    get_query_embedding_func
) -> dict:
    """
    Queries ChromaDB to retrieve schema elements matching the attributes, 
    optionally filtering by the chosen views.
    """
    retrieved_columns = {}
    if not attributes:
        return retrieved_columns

    # Build metadata filter if relevant views are defined
    where_filter = None
    if chosen_views:
        if len(chosen_views) == 1:
            where_filter = {"view_name": list(chosen_views)[0]}
        else:
            where_filter = {"view_name": {"$in": list(chosen_views)}}

    for attr in attributes:
        query_embedding = get_query_embedding_func(attr)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10, 
            include=["documents", "metadatas"],
            where=where_filter
        )
        if results and results.get('metadatas'):
            retrieved_columns[attr] = results['metadatas'][0]
            
    return retrieved_columns

def retrieve_columns_by_user_question(
    user_question: str, 
    chosen_views: list, 
    collection, 
    get_query_embedding_func,
    n_results: int = 10
) -> list:
    """
    Queries ChromaDB to retrieve schema elements matching the user's question,
    optionally filtering by the chosen views.
    """
    if not user_question:
        return []

    # Build metadata filter if relevant views are defined
    where_filter = None
    if chosen_views:
        if len(chosen_views) == 1:
            where_filter = {"view_name": list(chosen_views)[0]}
        else:
            where_filter = {"view_name": {"$in": list(chosen_views)}}

    # Generate embedding for the full user question
    query_embedding = get_query_embedding_func(user_question)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results, 
        include=["documents", "metadatas"],
        where=where_filter
    )
    
    # Chroma returns a list of lists for metadatas (one list per query embedding)
    if results and results.get('metadatas') and len(results['metadatas']) > 0:
        return results['metadatas'][0]
        
    return []

def parse_lsh_key(key) -> tuple:
    """
    Parses a table and column name from an LSH index key.
    Handles formats like 'table.column == value', 'table.column', or tuples.
    """
    tbl, col = None, None
    if isinstance(key, str):
        if "|" in key:
            parts = key.split("|")
            tbl, col = parts[1], parts[2]
        elif "." in key:
            parts = key.split(".")
            if len(parts) == 2:
                tbl, col = parts[0], parts[1]
    elif isinstance(key, tuple) and len(key) >= 2:
        tbl, col = key[0], key[1]
    return tbl, col


def retrieve_values_by_lsh(
    values: list, 
    chosen_views: set, 
    retrieved_columns: dict, 
    lsh_path: str = '../data/lsh_index.pkl'
) -> dict:
    """
    Queries LSH index dictionary for approximate value matching,
    filtering matches to ensure they belong to the selected views.
    Updates retrieved_columns in-place for matches.
    """
    retrieved_values = {}
    if not values or not os.path.exists(lsh_path):
        return retrieved_values

    try:
        with open(lsh_path, 'rb') as f:
            lsh_data = pickle.load(f)
        
        lsh_index = lsh_data["lsh"]
        num_perm = lsh_data.get("num_perm", 128)
        shingle_size = lsh_data.get("shingle_size", 3)
        
        for value in values:
            val_normalized = value.lower()
            
            # Generate shingles matching the indexing format
            shingles = [
                val_normalized[i:i+shingle_size] 
                for i in range(len(val_normalized) - shingle_size + 1)
            ] if len(val_normalized) >= shingle_size else [val_normalized]
            
            m = MinHash(num_perm=num_perm)
            for shingle in shingles:
                m.update(shingle.encode('utf-8'))
            
            matched_keys = lsh_index.query(m)
            print(f"matched values: {matched_keys}")
            
            if matched_keys:
                valid_matches = []
                for key in matched_keys:
                    tbl, col = parse_lsh_key(key)
                    
                    if tbl and col:
                        # Enforce view match filtering
                        if not chosen_views or tbl in chosen_views:
                            valid_matches.append(key)
                            col_key = f"{tbl}.{col}"
                            if col_key not in retrieved_columns:
                                retrieved_columns[col_key] = [{"table": tbl, "column": col}]
                
                if valid_matches:
                    retrieved_values[value] = valid_matches
                    
    except Exception as e:
        print(f"Warning: Failed to execute LSH lookup: {e}")
        
    return retrieved_values


# --- Orchestrator Node ---

def querying(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Main node orchestrating view identification, schema querying, and LSH matching.
    """
    keywords = state.get("keywords", {"Views": [], "Attributes": [], "Values": []})
    
    # 1. Identify relevant views from extracted Entities
    # chosen_views = identify_views_from_entities(
    #     entities=keywords.get("Entities", []),
    #     get_query_embedding_func=get_query_embedding
    # )
    chosen_views = keywords.get("Views", [])

    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path='../data/schema_db')
    collection = client.get_collection("schema_collection")
    
    # 2. Query attributes and restrict retrieval to chosen views
    v = [view.split('.')[-1] for view in chosen_views]
    # retrieved_columns = retrieve_columns_by_attributes(
    #     attributes=keywords.get("Attributes", []),
    #     chosen_views=v,
    #     collection=collection,
    #     get_query_embedding_func=get_query_embedding
    # )
    
    user_messages = [msg for msg in state.get("messages", []) if msg.get("role") == "user"]
    user_question = user_messages[-1]["content"] if user_messages else ""
    retrieved_columns = retrieve_columns_by_user_question(
        user_question=user_question,
        chosen_views=v,
        collection=collection,
        get_query_embedding_func=get_query_embedding,
        n_results=10  # Adjust the number of retrieved schema items as needed
    )
    
    # 3. Query LSH for values, filtering matches by chosen views
    retrieved_values = retrieve_values_by_lsh(
        values=keywords.get("Values", []),
        chosen_views=v,
        retrieved_columns=retrieved_columns
    )
    
    # 4. Format outputs
    retrieved_schema_formatted = format_retrieved_schema(retrieved_columns)
    retrieved_values_formatted = format_retrieved_values(retrieved_values)

    print(f"---Chosen Views---\n{list(chosen_views)}")
    print(f"---Retrieved Schema---\n{retrieved_schema_formatted}")
    print(f"---Retrieved Values via LSH---\n{retrieved_values_formatted}")

    return {
        "retrieved_columns": retrieved_schema_formatted,
        "retrieved_values": retrieved_values_formatted
    }

def format_retrieved_schema(raw_schema: dict) -> dict:
    """
    Transforms the raw metadata into a dictionary of {view_name: [unique_columns]}.
    """
    formatted = {}

    # Iterate through each keyword's list of metadata
    for keyword, metadata_list in raw_schema.items():
        for entry in metadata_list:
            view_name = entry.get('view_name')
            view_column = entry.get('view_column')

            if view_name and view_column:
                # Initialize the list if the view_name is new
                if view_name not in formatted:
                    formatted[view_name] = set() # Use a set to prevent duplicates
                
                # Add the column name to the set
                formatted[view_name].add(view_column)

    # Convert sets back to lists for the final output
    return {view: list(columns) for view, columns in formatted.items()}

def format_retrieved_values(retrieved_values: dict) -> str:
    """Helper function to format retrieved database values into a clean text context."""
    if not retrieved_values:
        return "No exact or partial matching database values retrieved."
    
    formatted_lines = []
    for keyword, matches in retrieved_values.items():
        formatted_lines.append(f"- Keyword matches for '{keyword}':")
        for match in matches:
            parts = match.split('|')
            if len(parts) == 4:
                schema, table, column, value = parts
                formatted_lines.append(f"  * Table/View: {schema}.{table} -> Column: {column} contains matched value: N'{value}'")
            elif len(parts) == 3:
                schema, table, column = parts
                formatted_lines.append(f"  * Table/View: {schema}.{table} -> Column: {column} contains a partial match")
            else:
                formatted_lines.append(f"  * Match: {match}")
    return "\n".join(formatted_lines)


def get_query_embedding(query: str):
    """Encapsulates the logic for fetching embeddings from Ollama."""
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    resp = ollama.embed(model=embed_model, input=query)
    return np.array(resp["embeddings"][0], dtype=np.float32)