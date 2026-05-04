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

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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
        
        batch_size = 20 
        # We need a counter for the IDs that doesn't reset every batch
        chunk_idx = 0 
        
        for i, batch in enumerate(chunk_list(docs, batch_size)):
            # 1. Correctly prepare IDs (incrementing globally per file)
            batch_ids = [f"{source}_{chunk_idx + j}" for j in range(len(batch))]
            
            # 2. Correctly extract Text and Metadata
            batch_texts = [doc["text"] for doc in batch]
            batch_metadatas = [doc["metadata"] for doc in batch]
            
            # 3. Add to collection with correct keyword arguments
            collection.add(
                ids=batch_ids,          # Unique string IDs
                documents=batch_texts,  # The actual content to be embedded
                metadatas=batch_metadatas
            )
            
            chunk_idx += len(batch) # Increment the counter
            print(f"Ingested batch {i+1} for {source}")

        total_chunks += len(docs)
        print(f"Successfully added {len(docs)} chunks from: {file_path}")
    
    print(f"\nIngestion Complete. Total chunks added: {total_chunks}")

# Make sure this helper function is defined in your file!
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    # Example usage:
    # run_ingestion("data/PM_concepts.jsonl", strategy="recursive")
    folder = "../data/MD-1/"
    files = [
        # "CreateContract.md",
        # "MittingManagement.md",
        # "MittingManagementSetting.md",
        # "MittingRoom.md",
        # "MittingTypeManagement.md",
        # "WorkFlow.md",
        # "WorkFlowManagement.md",
        # "FinancialManagement.md",
        # "RolesAndSecurity.md",
        # "DynamicFields.md",
        # "WBS-Program.md",
        # "ProjectTypeManagement.md",
        # "ContractTypeManagement.md", 
        # "IPMPAdvantages.md", 
        # "EPC.md",
                                            "IPMP-Vs-EPM.md",
        # "MadulePerformanceDetail.md", 
        # "BasicInfoProjectManagement.md",
        # "IPMP-ProjectManagementSoftware.md",
        # "IPMP-ProjectManagementMethods.md",
        # "EVMSetting.md",
        # "EVMDashboard.md",
        # "CreateProject.md",
        # "FinancialStatement.md"
    ]

    file_paths = [folder + file for file in files]

    run_ingestion(file_paths, strategy="markdown", collection_name="software_user_guide")
