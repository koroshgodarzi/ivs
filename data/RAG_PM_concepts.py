import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer
import tiktoken
import chromadb
from chromadb.utils import embedding_functions

from dotenv import load_dotenv
load_dotenv()

def count_tokens(text: str) -> int:
    """
    Counts tokens for LangChain chat messages.
    Uses a best-effort tokenizer based on the configured model.
    """
    backend = os.getenv("LLM_BACKEND", "openai")

    if backend == "openai":
        model = os.getenv("LLM_MODEL", "qwen2.5-coder-7b-instruct")

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

    elif backend == "ollama":
        encoding = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            trust_remote_code=True,
        )

    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend}")

    return len(encoding.encode(text))


def load_data(file_path: str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            for line in f:
                data.append(json.loads(line))
        else:
            data = json.load(f)
    return data

def process_entries(entries, max_tokens=250, overlap=50):
    processed_docs = []
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for entry in entries:
        title = entry.get("title", "")
        content = entry.get("data", "")
        related = ", ".join(entry.get("related_titles", []))
        
        token_count = count_tokens(content)
        
        if token_count <= max_tokens:
            full_text = f"Title: {title}\nContent: {content}"
            processed_docs.append({
                "text": full_text,
                "metadata": {"title": title, "related": related}
            })
        else:
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                full_text = f"Title: {title} (Part {i+1})\nContent: {chunk}"
                processed_docs.append({
                    "text": full_text,
                    "metadata": {"title": title, "related": related, "chunk_id": i}
                })
                
    return processed_docs


def save_to_chroma(processed_docs, path="./chroma_db"):
    client = chromadb.PersistentClient(path=path)
    
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="embeddinggemma"
    )

    collection = client.get_or_create_collection(
        name="project_management_rag",
        embedding_function=ollama_ef
    )

    ids = [f"id_{i}" for i in range(len(processed_docs))]
    documents = [doc["text"] for doc in processed_docs]

    metadatas = [doc["metadata"] for doc in processed_docs]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Successfully vectorized and added {len(documents)} chunks using Ollama (embeddinggemma).")


def run_pipeline(file_path):
    raw_data = load_data(file_path)
    
    docs = process_entries(raw_data, max_tokens=250, overlap=50)
    
    save_to_chroma(docs)

run_pipeline("PM_concepts.jsonl")