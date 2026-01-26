from graph.schema import GraphState
from typing import Optional, List, Dict, Any
import json
import os
import numpy as np
from pathlib import Path
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

def schema_retriever(state: GraphState) -> GraphState:
    """
    Initial node to identify relevant view names based on the user's prompt.
    Saves only the names to the state, omitting file loading.
    """
    user_messages = [msg for msg in state.get("messages", []) if msg["role"] == "user"]
    user_prompt = user_messages[-1]["content"] if user_messages else ""
    
    # Now returns a list of names instead of a large string
    relevant_views = get_relevant_view_names(user_prompt=user_prompt)
    
    # Store the list of names in state
    state["retrieved_schema"] = relevant_views
    
    print(f"Relevant views identified and stored: {relevant_views}")
    return state


def get_relevant_view_names(user_prompt: str = None, k: int = 1, schema_names: Optional[List[str]] = None) -> List[str]:
    """
    Determines which database views are relevant. 
    File loading logic has been removed.
    """
    view_mapping_path = Path(__file__).parent.parent.parent / "data" / "view_index_mapping.json"
    
    if user_prompt:
        try:
            # Find most similar embeddings
            top_k_indices = find_most_similar_embeddings(user_prompt, k=k)
            
            if not view_mapping_path.exists():
                raise FileNotFoundError(f"View index mapping file not found at: {view_mapping_path}")
            
            with open(view_mapping_path, "r", encoding="utf-8") as f:
                view_mapping = json.load(f)
            
            index_to_view = {v: k for k, v in view_mapping.items()}
            
            # Extract just the names
            schema_names = [index_to_view[int(idx)] for idx in top_k_indices if int(idx) in index_to_view]
            
            if not schema_names:
                raise ValueError(f"No matching views found for indices: {top_k_indices}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to identify relevant schemas: {str(e)}") from e
    elif not schema_names:
        raise ValueError("Either user_prompt or schema_names must be provided")
    
    # Return the list of names directly; no file reading occurs.
    return schema_names


def find_most_similar_embeddings(user_prompt: str, k: int = 5, embeddings_path: str = None) -> np.ndarray:
    """
    Embed a user prompt using Gemini API and find the k most similar 
    embeddings from a pre-computed numpy array.
    """
    # 1. Setup API Configuration
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    genai.configure(api_key=api_key)
    
    # 2. Handle File Path
    if embeddings_path is None:
        # Default path logic
        embeddings_path = Path(__file__).parent.parent.parent / "data" / "embeddings.npy"
    else:
        embeddings_path = Path(embeddings_path)
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")
    
    # 3. Load existing embeddings
    embeddings_array = np.load(str(embeddings_path))
    
    # 4. Generate Embedding for User Prompt
    # Using 'retrieval_query' task type which is optimized for search queries
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=user_prompt,
        task_type="retrieval_query"
    )
    
    # Extract the vector and convert to numpy
    user_embedding = np.array(result['embedding'])
    
    # 5. Compute Cosine Similarity
    # Normalize vectors to unit length
    user_norm = user_embedding / np.linalg.norm(user_embedding)
    # Normalize the entire matrix (axis 1)
    array_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    
    # Dot product of normalized vectors gives cosine similarity
    similarities = np.dot(array_norm, user_norm)
    
    # 6. Return top K indices
    # argsort returns indices in ascending order, so we reverse it [::-1]
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    return top_k_indices


if __name__ == "__main__":
    import os
    from pathlib import Path

    # 1. Setup Mock Environment Variables (Replace with your actual key for testing)
    # os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

    # 2. Define a Mock GraphState
    # This mimics the structure LangGraph uses
    mock_state: GraphState = {
        "messages": [
            {
                "role": "user", 
                "content": "I need to see the latest contract values and project statuses"
            }
        ],
        "retrieved_schema": []
    }

    print("--- Starting Schema Retrieval Test ---")
    
    try:
        # 3. Test the Node function
        # This will internally call get_relevant_view_names and find_most_similar_embeddings
        updated_state = schema_retriever(mock_state)
        
        # 4. Validate Results
        retrieved_names = updated_state.get("retrieved_schema", [])
        
        print("\nSUCCESS!")
        print(f"User Prompt: {mock_state['messages'][-1]['content']}")
        print(f"Identified View Names: {retrieved_names}")
        
        if isinstance(retrieved_names, list) and len(retrieved_names) > 0:
            print("Verification: State correctly contains a list of names.")
        else:
            print("Verification: No views were found. Check if your .npy and .json files are populated.")

    except FileNotFoundError as e:
        print(f"\nFILE ERROR: {e}")
        print("Ensure your 'table_view' directory contains 'embeddings.npy' and 'view_index_mapping.json'.")
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {type(e).__name__}: {e}")

    print("\n--- Test Complete ---")