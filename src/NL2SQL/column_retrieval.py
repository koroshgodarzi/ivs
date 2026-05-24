import chromadb
from NL2SQL.schema import GraphState
from langchain_core.runnables import RunnableConfig
import numpy as np
import ollama
import os
import pickle
from datasketch import MinHash

def querying(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Embeds keywords, searches ChromaDB to retrieve the necessary schema,
    and queries an LSH index dictionary to perform database value matching.
    """
    keywords = state.get("keywords", {"Attributes": [], "Values": []})
    
    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path='../data/schema_db')
    collection = client.get_collection("schema_collection")

    retrieved_columns = {}
    retrieved_values = {}  # Store matched cell values for prompt synthesis

    # --- 1. Retrieve Schema based on Attributes/Columns ---
    for attr in keywords.get("Attributes", []):
        query_embedding = get_query_embedding(attr)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10, 
            include=["documents", "metadatas"]
        )
        if results and results.get('metadatas'):
            retrieved_columns[attr] = results['metadatas'][0]

    # --- 2. Retrieve Schema & Values based on Cell Values (LSH Index) ---
    lsh_path = '../data/lsh_index.pkl'
    if os.path.exists(lsh_path) and keywords.get("Values"):
        try:
            with open(lsh_path, 'rb') as f:
                lsh_data = pickle.load(f)
            
            # Extract the actual LSH index and metadata
            lsh_index = lsh_data["lsh"]
            num_perm = lsh_data.get("num_perm", 128)
            shingle_size = lsh_data.get("shingle_size", 3)
            
            for value in keywords["Values"]:
                val_normalized = value.lower()
                
                # Create shingles matching the size used during indexing
                shingles = [
                    val_normalized[i:i+shingle_size] 
                    for i in range(len(val_normalized) - shingle_size + 1)
                ] if len(val_normalized) >= shingle_size else [val_normalized]
                
                # Calculate MinHash signature of the query value
                m = MinHash(num_perm=num_perm)
                for shingle in shingles:
                    m.update(shingle.encode('utf-8'))
                
                # Query LSH for approximate matches
                matched_keys = lsh_index.query(m)
                
                if matched_keys:
                    retrieved_values[value] = matched_keys
                    
                    # Extract tables/columns from matched keys to update retrieved_columns
                    for key in matched_keys:
                        tbl, col = None, None
                        
                        # Handle string keys like "table.column == value" or similar formats
                        if isinstance(key, str):
                            if "==" in key:
                                parts = key.split("==")[0].strip().split(".")
                                if len(parts) == 2:
                                    tbl, col = parts[0], parts[1]
                            elif "." in key:
                                parts = key.split(".")
                                if len(parts) == 2:
                                    tbl, col = parts[0], parts[1]
                                    
                        # Handle tuple keys like (table_name, column_name, value)
                        elif isinstance(key, tuple) and len(key) >= 2:
                            tbl, col = key[0], key[1]
                        
                        # If a valid table and column were found, add them to retrieved_columns
                        if tbl and col:
                            col_key = f"{tbl}.{col}"
                            if col_key not in retrieved_columns:
                                # Mimic the standard schema metadata format 
                                retrieved_columns[col_key] = [{"table": tbl, "column": col}]

        except Exception as e:
            print(f"Warning: Failed to execute LSH lookup: {e}")

    # Format the merged schema (including columns from attribute search and value search)
    retrieved_schema_formatted = format_retrieved_schema(retrieved_columns)
    retrieved_values_formatted = format_retrieved_values(retrieved_values)

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