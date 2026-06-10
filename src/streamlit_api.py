import streamlit as st
import time
import os
from reportingAssistant.workflow import column_based_graph # Import your compiled graph
# from utils import format_chat_history # Your custom helper

# --- Page Config ---
st.set_page_config(page_title="Database Assistant", layout="wide")

app = column_based_graph()

# --- Initialize Session State ---
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {
        "messages": [],
        "master_messages": [], 
        "retry_count": 0,
        "generated_query": [],
        "error_message": [],
        "query_results": None,
        "query_explanation": [],
        "retrieved_columns": None,
        "keywords": None,
        "query_generation_user_prompt": None
    }

st.title("📊 Natural Language to SQL Assistant")

# --- Display Chat History ---
for msg in st.session_state.graph_state["master_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
if user_input := st.chat_input("Ask a question about the database..."):
    # 1. Update state with user message
    st.session_state.graph_state["master_messages"].append({"role": "user", "content": user_input})
    st.session_state.graph_state["generated_query"].append([]) # Start new turn list
    st.session_state.graph_state["retry_count"] = 0
    st.session_state.graph_state["final_response"] = ""
    st.session_state.graph_state["retry_count"] = 0

    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Call the graph
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            try:
                # Setup configuration
                config = {
                    "configurable": {
                        "thread_id": "streamlit_session",
                        "model_name": "open_router",
                        "prompt_template_dir": "../prompt_template",
                        "data_dir": "../data/clean",
                        "docs_dir": "../data/clean"
                    }
                }

                # Execute graph
                final_state = app.invoke(st.session_state.graph_state, config=config)

                # Update persistent state
                st.session_state.graph_state = final_state

                # Extract the last assistant message)
                assistant_response = final_state["response"][-1]
                st.markdown(assistant_response)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")