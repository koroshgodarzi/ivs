import streamlit as st
import requests
import json
from typing import List, Dict, Any

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Training Assistant UI", page_icon="🤖")

# --- API Details ---
# Replace with the actual URL of your running FastAPI application
FASTAPI_API_URL = "http://localhost:8000" # Default, assuming it runs locally

# --- Helper Functions ---

def get_models():
    """Fetches the list of available models from the FastAPI backend."""
    try:
        response = requests.get(f"{FASTAPI_API_URL}/v1/models")
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {e}")
        return []

def send_chat_request(messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    """
    Sends a chat request to the FastAPI backend and returns the response.
    Mimics the structure expected by the /v1/chat/completions endpoint.
    """
    # The FastAPI endpoint expects a ChatRequest object, which includes 'model' and 'messages'.
    # We'll construct this payload.
    payload = {
        "model": model,
        "messages": messages,
        # 'user' is optional and not strictly needed for this basic UI,
        # but could be added if your FastAPI handled user context.
    }

    try:
        response = requests.post(f"{FASTAPI_API_URL}/v1/chat/completions", json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending message: {e}")
        return {"error": str(e)}
    except json.JSONDecodeError:
        st.error("Error decoding JSON response from API.")
        return {"error": "Invalid JSON response"}


# --- Streamlit UI ---

st.title("💬 Training Assistant Chatbot")
st.caption("🚀 A Streamlit UI for your FastAPI Chatbot API")

# --- Model Selection ---
available_models = get_models()
model_names = [m["id"] for m in available_models] if available_models else ["ollama"] # Fallback to default

if not model_names:
    st.warning("Could not retrieve model list from API. Using default model 'ollama'.")
    selected_model = "ollama"
else:
    selected_model = st.selectbox("Choose a model:", model_names, index=model_names.index("ollama") if "ollama" in model_names else 0)

st.write(f"Selected model: `{selected_model}`")


# --- Chat History Management ---
# Use Streamlit's session state to persist chat history
if "messages" not in st.session_state:
    st.session_state.messages = [] # Initialize with an empty list

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
user_input = st.chat_input("Ask me anything about your training data...")

if user_input:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare messages for the API call (send the full history)
    # The FastAPI endpoint expects the last message from the user, but sending history
    # allows for potential future stateful implementations in the API.
    # For now, we send the latest user message as per your API's logic.
    api_messages_payload = []
    if st.session_state.messages:
        # Add messages up to the latest user input.
        # Your API processes the last message in the list.
        api_messages_payload = st.session_state.messages

    # Show placeholder for assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

    # Call the FastAPI backend
    response_data = send_chat_request(api_messages_payload, selected_model)

    if "error" in response_data:
        # Display error message if API call failed
        message_placeholder.markdown(f"Error: {response_data['error']}")
        # Add error to session state as well
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {response_data['error']}"})
    elif "choices" in response_data and response_data["choices"]:
        assistant_response = response_data["choices"][0]["message"]["content"]
        message_placeholder.markdown(assistant_response)
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    else:
        # Handle unexpected response structure
        message_placeholder.markdown("Received an unexpected response format from the API.")
        st.session_state.messages.append({"role": "assistant", "content": "Received an unexpected response format from the API."})


# --- Optional: Clear Chat Button ---
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.header("About")
st.sidebar.info(
    "This UI connects to a FastAPI backend that powers a RAG chatbot. "
    "Enter your questions in the chat input below."
)
st.sidebar.info(f"API Endpoint: `{FASTAPI_API_URL}`")