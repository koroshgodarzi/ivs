import chromadb
from NL2SQL.schema import GraphState
from langchain_core.runnables import RunnableConfig
import numpy as np
import ollama
import os

def querying(state: GraphState, config: RunnableConfig) -> GraphState:
    """
    Embeds keywords and searches ChromaDB to retrieve the necessary schema.
    """
    keywords = state["keywords"]
    
    # Initialize your ChromaDB client and embedding model
    # This should be adapted to your specific setup
    client = chromadb.PersistentClient(path='../data/schema_db')
    collection = client.get_collection("schema_collection")

    retrieved_columns = {}

    # Embed the keywords
    for value in keywords["Attributes"]:
        query_embedding = get_query_embedding(value)

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,  # Retrieve the top 5 most relevant schema parts
            include=["documents", "metadatas"]
        )
        
        retrieved_columns[value] = results['metadatas'][0]

    retrieved_columns = format_retrieved_schema(retrieved_columns)

    print(f"---Retrieved Schema---\n{retrieved_columns}")

    return {"retrieved_columns": retrieved_columns}

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

def get_query_embedding(query: str):
    """Encapsulates the logic for fetching embeddings from Ollama."""
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")
    resp = ollama.embed(model=embed_model, input=query)
    return np.array(resp["embeddings"][0], dtype=np.float32)