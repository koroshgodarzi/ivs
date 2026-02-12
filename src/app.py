import os
import time
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
load_dotenv()

# --- Pydantic Models for OpenAI Compatibility ---

class ModelCard(BaseModel):
    """A model card, as returned by the /v1/models endpoint."""
    id: str
    """The ID of the model."""
    object: str = "model"
    """The object type, which is always 'model'."""
    created: int = Field(default_factory=lambda: int(time.time()))
    """The timestamp of when the model was created."""
    owned_by: str = "user"
    """The owner of the model, typically 'user' for custom models."""

class ModelList(BaseModel):
    """A list of models, as returned by the /v1/models endpoint."""
    object: str = "list"
    """The object type, which is always 'list'."""
    data: List[ModelCard]
    """A list of the available model cards."""


# --- FastAPI Application ---

app = FastAPI(
    title="SQL Generator - OpenAI Compatible API",
    description="An OpenAI-compatible API for the SQL Generator agent.",
)

# --- API Endpoints ---

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    An endpoint to list the available models.
    This is required for Open WebUI to show your agent in the model list.
    """
    # Get the model name from an environment variable, with a default
    model_name = os.getenv("LLM_MODEL", "sql-generator")

    return ModelList(
        data=[
            ModelCard(id=model_name)
        ]
    )

if __name__ == "__main__":
    import uvicorn
    # To run this file, you would use the command:
    # uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)