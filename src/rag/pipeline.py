import json
from rag.chunkers import chunk_jsonl_recursive, chunk_markdown_sections
from rag.vectorstore import get_vector_collection
import os

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            return [json.loads(line) for line in f]
        elif file_path.endswith('.json'):
            return json.load(f)
        else:
            return [{"data": f.read()}]

def run_ingestion(file_path_list, strategy="recursive", db_path="../data/chroma_db", collection_name="project_management_rag"):
    collection = get_vector_collection(db_path, collection_name)
    total_chunks = 0
    
    for file_path in file_path_list:
        source = os.path.basename(file_path)
        raw_data = load_file(file_path)
    
        if strategy == "markdown":
            docs = chunk_markdown_sections(raw_data, source)
        else:
            docs = chunk_jsonl_recursive(raw_data)
            
        collection.add(
            ids=[f"{source}_{i}" for i in range(len(docs))],
            documents=[doc["text"] for doc in docs],
            metadatas=[doc["metadata"] for doc in docs]
        )
        total_chunks += len(docs)
        print(f"Successfully added {len(docs)} chunks from: {file_path})")
    
    print(f"\nIngestion Complete. Total chunks added: {total_chunks}")

if __name__ == "__main__":
    # Example usage:
    # run_ingestion("data/PM_concepts.jsonl", strategy="recursive")
    folder = "../data/MarkDown/"
    files = [
        "CreateContract.md",
        "MittingManagement.md",
        "MittingManagementSetting.md",
        "MittingRoom.md",
        "MittingTypeManagement.md",
        "WorkFLow.md",
        "WorkFlowManagement.md"
    ]

    file_paths = [folder + file for file in files]

    run_ingestion(file_paths, strategy="markdown", collection_name="software_user_guide")
