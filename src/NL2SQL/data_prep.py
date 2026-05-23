import pandas as pd
import chromadb
import os
from chromadb.utils import embedding_functions
import requests
import csv
from collections import defaultdict

def get_join_relationships(needed_views_dict, csv_file_path='../docs/IDColumns.csv'):
    """
    Identifies joinable columns between a set of required views based on a schema CSV.
    
    Args:
        needed_views_dict (dict): The dictionary of views and columns the agent identified.
        csv_file_path (str): Path to the IDColumns.csv file.
        
    Returns:
        dict: A mapping of Column_Names to the list of views that contain them.
    """
    needed_table_names = set(needed_views_dict.keys())
    column_to_tables = defaultdict(list)

    # Read the CSV and map which requested tables share common ID columns
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            table = row['TABLE_NAME']
            column = row['Column_Name']
            
            # Only consider tables that the agent has flagged as "needed"
            if table in needed_table_names:
                column_to_tables[column].append(table)

    # Filter out columns that only appear in one of the selected tables 
    # (A column must exist in at least 2 tables to be used for a join)
    join_metadata = {
        col: tables for col, tables in column_to_tables.items() 
        if len(tables) > 1
    }

    return join_metadata

def create_chroma_db_from_csv(csv_file: str, db_path: str, collection_name: str, batch_size: int = 100):
    """
    Creates and populates a ChromaDB collection from a CSV file in batches.

    Args:
        csv_file (str): The path to the input CSV file.
        db_path (str): The directory where the ChromaDB database will be stored.
        collection_name (str): The name of the collection to be created.
        batch_size (int): The number of documents to add in each batch.
    """
    # 1. Load the data from the CSV file
    try:
        df = pd.read_csv(csv_file)
        # Handle potential empty rows or missing 'Description' values
        df.dropna(subset=['Description'], inplace=True)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # 2. Initialize ChromaDB client for persistent storage
    print(f"Initializing ChromaDB client at '{db_path}'...")
    client = chromadb.PersistentClient(path=db_path)

    # 3. Initialize the embedding model
    embedding_model = get_embedding_function()

    # 4. Create or get the collection
    print(f"Creating or getting collection: '{collection_name}'")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_model)

    # 5. Prepare documents, metadata, and IDs for batch insertion
    documents = []
    metadatas = []
    ids = []

    print("Processing data from CSV...")
    for index, row in df.iterrows():
        view_name = row['TABLE_NAME']
        column_name = row['Column_Name']
        description = row['Description']

        # The content to be embedded is the description
        documents.append(description)

        # Store view and column names as metadata
        metadatas.append({
            'view_name': view_name,
            'view_column': column_name
        })

        # Create a unique ID for each entry
        ids.append(f"id_{view_name}_{column_name}_{index}")

    # 6. Add the data to the collection in batches
    if documents:
        total_docs = len(documents)
        print(f"Adding {total_docs} documents to the collection in batches of {batch_size}...")

        # Create generators for chunking
        doc_chunks = chunk_list(documents, batch_size)
        meta_chunks = chunk_list(metadatas, batch_size)
        id_chunks = chunk_list(ids, batch_size)

        for i, (doc_batch, meta_batch, id_batch) in enumerate(zip(doc_chunks, meta_chunks, id_chunks)):
            print(f"Adding batch {i+1}...")
            collection.add(
                documents=doc_batch,
                metadatas=meta_batch,
                ids=id_batch
            )

        print("Database creation and population complete!")
        print(f"Total documents in collection: {collection.count()}")
    else:
        print("No valid documents found in the CSV to add to the collection.")


def get_embedding_function(ollama_model="embeddinggemma", hf_model="google/embeddinggemma-300m"):
    """
    Checks if Ollama is running locally. Returns OllamaEmbeddingFunction if available,
    otherwise returns HuggingFaceEmbeddingFunction.
    """
    ollama_url = "http://localhost:11434"
    
    try:
        # Check if Ollama is alive (pings the base API)
        response = requests.get(ollama_url, timeout=2)
        if response.status_code == 200:
            print(f"Ollama detected. Using model: {ollama_model}")
            return embedding_functions.OllamaEmbeddingFunction(
                url=f"{ollama_url}/api/embeddings",
                model_name=ollama_model
            )
    except requests.exceptions.ConnectionError:
        print("Ollama not found.")

    # Fallback to Hugging Face
    print(f"Falling back to Hugging Face API. Using model: {hf_model}")
    
    return embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=os.getenv("HF_TOKEN"),
        model_name=hf_model
    )

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    # Define the paths and names
    CSV_FILE_PATH = '../data/column_desc.csv'
    DB_DIRECTORY = '../data/schema_db'
    COLLECTION_NAME = 'schema_collection'

    # Create the database
    create_chroma_db_from_csv(CSV_FILE_PATH, DB_DIRECTORY, COLLECTION_NAME)

    # Example of how to test the database
    print("\n--- Testing the database with a query ---")
    client = chromadb.PersistentClient(path=DB_DIRECTORY)
    collection = client.get_collection(name=COLLECTION_NAME)

    # Query the collection
    query_results = collection.query(
        query_texts=["the money that has been paid"],
        n_results=5
    )

    print("Query results for 'the money that has been paid':")
    for i, doc in enumerate(query_results['documents'][0]):
        metadata = query_results['metadatas'][0][i]
        print(f"  - Result {i+1}:")
        print(f"    Document: {doc}")
        print(f"    Metadata: {metadata}")