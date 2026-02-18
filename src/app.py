from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time
import uuid
from graph.workflow import build_graph
from typing import List, Optional, Dict, Any, TypedDict

app = FastAPI()
workflow_graph = build_graph()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    user: Optional[str] = "default_user"

@app.get("/ping")
def ping():
    return {"ok": True}

@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o-api", "object": "model", "owned_by": "openai"},
            {"id": "qwen-api", "object": "model", "owned_by": "alibaba"},
            {"id": "ollama:qwen2.5-coder:14b", "object": "model", "owned_by": "local"}
        ]
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    # 1. Prepare User Input for the Graph
    # We pass the full history or just the latest question based on your GraphState needs
    user_question = req.messages[-1].content
    
    initial_state = {
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "retry_count": 0,
        "summary_context": None,
        "generated_query": None,
        "error_message": None,
        "query_results": None,
        "validation_result": None,
        "query_explanation": None,
    }

    # 2. Execution Config
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "model_name": req.model
        }
    }

    # 3. Run the Workflow
    # The graph will run format_final_response which appends a new message to the list
    final_state = workflow_graph.invoke(initial_state, config=config)
    
    # 4. EXTRACT THE LAST MESSAGE CONTENT
    # Since your format_final_response node does: state["messages"].append(...)
    # The response is now the last item in the messages list.
    if final_state.get("messages") and len(final_state["messages"]) > 0:
        last_message = final_state["messages"][-1]
        # Safety check: ensure the last message is from the assistant
        if last_message.get("role") == "assistant":
            answer = last_message.get("content")
        else:
            answer = "I processed the request but the final message was not from the assistant."
    else:
        answer = "I'm sorry, the workflow failed to generate a response message."

    # 5. Return OpenAI Compatible JSON
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
                    "content": answer # This now contains your Markdown table/explanation
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

    
if __name__ == "__main__":
    import uvicorn
    print("✅ Text-to-SQL Workflow API LOADED")
    uvicorn.run(app, host="0.0.0.0", port=8000)