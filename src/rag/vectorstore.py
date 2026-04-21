import chromadb
import requests
from chromadb.utils import embedding_functions
import os

from dotenv import load_dotenv
load_dotenv()


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

def get_vector_collection(path, collection_name):
    client = chromadb.PersistentClient(path=path)
    
    embedding_fn = get_embedding_function()

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

# Usage
# collection = get_vector_collection()