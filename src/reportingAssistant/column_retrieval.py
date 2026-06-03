import chromadb
from reportingAssistant.schema import GraphState
from langchain_core.runnables import RunnableConfig
import numpy as np
import ollama
import os
import pickle
from datasketch import MinHash


def retrieve_columns_by_attributes(
    attributes: list, 
    chosen_views: set, 
    collection, 
    get_query_embedding_func
) -> dict:
    """
    Queries ChromaDB to retrieve schema elements matching the attributes, 
    filtering by the fully qualified chosen views.
    """
    retrieved_columns = {}
    if not attributes:
        return retrieved_columns

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


def parse_lsh_key(key) -> tuple:
    """
    Parses a fully qualified table name and column name from an LSH index key.
    Handles formats like 'schema|table|column|value', 'schema.table.column', or tuples.
    Returns: (fully_qualified_table, column)
    """
    tbl, col = None, None
    if isinstance(key, str):
        if "|" in key:
            parts = key.split("|")
            # Format: 'schema|table|column|value' or 'schema|table|column'
            if len(parts) >= 3:
                schema, table, column = parts[0], parts[1], parts[2]
                tbl = f"{schema}.{table}"
                col = column
        elif "." in key:
            parts = key.split(".")
            if len(parts) >= 3:
                # Format: 'schema.table.column' (e.g. Report.vw_CBS.ProjectID)
                tbl = ".".join(parts[:-1])
                col = parts[-1]
            elif len(parts) == 2:
                tbl, col = parts[0], parts[1]
    elif isinstance(key, tuple) and len(key) >= 2:
        tbl, col = key[0], key[1]
    return tbl, col


def retrieve_values_by_lsh(
    values: list, 
    chosen_views: set, 
    retrieved_columns: dict, 
    lsh_path: str
) -> dict:
    """
    Queries LSH index dictionary for approximate value matching,
    filtering matches using fully qualified view paths.
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
                        # Direct fully qualified filter check
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


def querying(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Main node orchestrating view identification, schema querying, and LSH matching.
    """
    keywords = state.get("keywords", {"Views": [], "Attributes": [], "Values": []})
    chosen_views = keywords.get("Views", [])

    # Fetch configured dynamic paths
    configurable = config.get("configurable", {})
    data_dir = configurable.get("data_dir", os.path.join("..", "data", "noisy_inclusive"))

    client = chromadb.PersistentClient(path=os.path.join(data_dir, 'schema_db'))
    collection = client.get_collection("schema_collection")
    
    # Check attributes using unmodified, fully qualified chosen view names
    retrieved_columns = retrieve_columns_by_attributes(
        attributes=keywords.get("Attributes", []),
        chosen_views=set(chosen_views),
        collection=collection,
        get_query_embedding_func=get_query_embedding
    )
    
    # Query LSH filtering by the exact fully qualified views
    retrieved_values = retrieve_values_by_lsh(
        values=keywords.get("Values", []),
        chosen_views=set(chosen_views),
        retrieved_columns=retrieved_columns,
        lsh_path=os.path.join(data_dir, 'lsh_index.pkl')
    )
    
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
    Transforms raw metadata into {view_name: [unique_columns]}, compatible
    with both LSH output values and ChromaDB.
    """
    formatted = {}

    for keyword, metadata_list in raw_schema.items():
        for entry in metadata_list:
            # Handles both standard DB formatting and the LSH parsed fallback names
            view_name = entry.get('view_name') or entry.get('table')
            view_column = entry.get('view_column') or entry.get('column')

            if view_name and view_column:
                if view_name not in formatted:
                    formatted[view_name] = set()
                formatted[view_name].add(view_column)

    return {view: list(columns) for view, columns in formatted.items()}


def format_retrieved_values(retrieved_values: dict) -> str:
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
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    resp = ollama.embed(model=embed_model, input=query)
    return np.array(resp["embeddings"][0], dtype=np.float32)