import os
import json
import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

# Import components from trainingAssistant.workflow.py
# Make sure your project structure allows for this import.
# For example, if this API file is at the root and 'trainingAssistant' is a sibling directory,
# this import path should work correctly.
from trainingAssistant.workflow import create_rag_graph
from trainingAssistant.schema import AgentState # Assuming AgentState is a TypedDict or similar structure
from langchain_core.messages import HumanMessage, AIMessage # For constructing LangChain messages


# --- FastAPI Setup ---
app = FastAPI()

# --- Initialize Workflow Graph ---
# The checkpointer inside create_rag_graph uses MemorySaver as per your workflow.py.
# For a production API requiring persistent chat history across server restarts,
# you would typically configure create_rag_graph to use a SqliteSaver or similar.
rag_workflow_graph = create_rag_graph()


# --- Pydantic Models for API Request/Response ---
# These models define the expected structure of incoming chat requests
# and the structure of messages within those requests, mimicking OpenAI's API.
class Message(BaseModel):
    role: str # e.g., "user", "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str # The model name to be used, e.g., "ollama"
    messages: List[Message] # A list of message objects, typically conversation history
    user: Optional[str] = "default_user" # Optional user identifier

# --- API Endpoints ---

# Health check endpoint
@app.get("/ping")
def ping():
    """Returns a simple health check response."""
    return {"ok": True}

# Endpoint to list available models (for compatibility with OpenAI clients)
@app.get("/v1/models")
def get_models():
    """Lists available models, including 'ollama' as used in your workflow."""
    return {
        "object": "list",
        "data": [
            {"id": "ollama", "object": "model", "owned_by": "local"},
            {"id": "gpt-4o-api", "object": "model", "owned_by": "openai"},
            {"id": "qwen-api", "object": "model", "owned_by": "alibaba"},
            {"id": "ollama:qwen2.5-coder:14b", "object": "model", "owned_by": "local"} # Example local model
        ]
    }

# Main chat completion endpoint
@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    """
    Processes a chat request using the trainingAssistant RAG workflow.
    Expects the latest user message and invokes the graph to generate a response.
    """
    # 1. Prepare User Input for the Graph
    if not req.messages:
        # Return an error if no messages are provided in the request
        return {"error": "No messages provided in the request."}, 400

    # The trainingAssistant workflow, as shown in its main() function,
    # processes only the latest HumanMessage and uses its content for the 'query' field.
    user_question = req.messages[-1].content
    
    # Initialize AgentState as expected by your create_rag_graph workflow.
    # This dictionary mirrors the `initial_state` creation in the `main()` function
    # of your `trainingAssistant.workflow.py`.
    initial_state_for_graph = {
        "messages": [HumanMessage(content=user_question)], # Only the latest user message as a LangChain HumanMessage
        "query": user_question,
        "query_embedding": [],
        "selected_sources": [],
        "candidate_paths": [],
        "selected_paths": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "steps_taken": 0,
        "low_confidence": False,
        "is_finished": False,
        "last_generated_segment": "",
        "best_answer": "",
        "best_is_finished": False,
        "best_min_logprob": -1000
    }

    # 2. Execution Configuration for the Graph
    # Each API call generates a new unique `thread_id`. This means each chat request
    # is treated as a fresh, independent interaction, similar to how your `main()`
    # function processes each question separately with a new timestamp-based `thread_id`.
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()), # Generate a unique ID for this chat session
            "model_name": req.model, # Use the model specified in the request
            'n_chunks': 5 # Configurable parameter, as seen in your workflow's main function
        }
    }

    # 3. Run the Workflow
    answer = "I'm sorry, an unexpected error occurred while generating a response."
    try:
        # Use `ainvoke` for asynchronous execution, suitable for FastAPI
        final_state = await rag_workflow_graph.ainvoke(initial_state_for_graph, config=config)
        
        # 4. Extract the final answer from the graph's output state
        # Your trainingAssistant workflow stores the final generated response in the 'answer' field.
        answer = final_state.get("answer", answer) # Use the default error message if 'answer' is not found

    except Exception as e:
        print(f"Error invoking trainingAssistant graph: {e}")
        answer = f"An error occurred while processing your request: {str(e)}"

    # 5. Return OpenAI Compatible JSON Response
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer # The generated answer from your RAG workflow
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0, # Placeholder; actual token counts would require LLM integration
            "completion_tokens": 0, # Placeholder
            "total_tokens": 0 # Placeholder
        }
    }

# --- Uvicorn Server Startup ---
# This block allows you to run the API using 'python your_api_file.py'
if __name__ == "__main__":
    import uvicorn
    print("✅ Training Assistant Workflow API LOADED")
    # You can change the host and port as needed
    uvicorn.run(app, host="0.0.0.0", port=8000)