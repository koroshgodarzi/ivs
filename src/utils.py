from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
import os
import json
import re
import os
import tiktoken
from typing import List, Dict
from langchain_core.messages import BaseMessage
from transformers import AutoTokenizer
from openai import OpenAI
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import requests
import csv
from collections import defaultdict
# import chromadb
# from chromadb.utils import embedding_functions

from dotenv import load_dotenv
load_dotenv()


import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

def get_llm(model_id: str, max_tokens: int = 4096, reasoning=False):
    """
    model_id comes from the UI/API request.
    Example IDs: 'gpt-4o', 'qwen-72b-api', 'ollama-qwen'
    """
    if model_id.startswith("hugging_face"):
        client = ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN"),
            model="Qwen/Qwen2.5-14B:featherless-ai",
            temperature=0,
            max_tokens=max_tokens,
        )
        return client

    # 1. OLLAMA Logic
    elif model_id.startswith("ollama"):
        # You can extract the specific version if you send 'ollama:qwen2.5'
        model_name =  "qwen2.5-coder:32b"
        # model_name = model_name[0] + ':' + model_name[1]
        return ChatOllama(
            model=model_name,
            temperature=0,
            num_predict=max_tokens,
            format="json",
            reasoning=False
        )

    # 2. QWEN API Logic (OpenAI Compatible)
    elif "qwen" in model_id.lower() and "api" in model_id.lower():
        return ChatOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL"), # e.g. DashScope or your proxy
            model='Qwen3-30B-A3B-lbu2r',
            temperature=0,
            max_tokens=max_tokens,
        )

    elif "open_router" in model_id.lower():
        reasoning_config = {"effort": "medium"} if reasoning else {"effort": "none"}
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY_REPORTER"),
            base_url="https://openrouter.ai/api/v1", # e.g. DashScope or your proxy
            model='qwen/qwen3.5-9b',
            temperature=0,
            max_tokens=max_tokens,
            reasoning=reasoning_config
        )

    # 3. GPT Logic (OpenAI)
    elif "gpt" in model_id.lower():
        return ChatOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("GPT_BASE_URL"),
            model='GPT-5-Nano-xudvw',
            temperature=0,
            max_tokens=max_tokens,
        )

    # Fallback / Default
    else:
        # Use your original environment variable logic as a fallback
        return ChatOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            temperature=0,
            max_tokens=max_tokens,
        )


def get_truncated_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Returns the most recent messages that fit within max_tokens."""
    truncated = []
    # Work backwards from the most recent message
    for msg in reversed(messages):
        # Temporary list to check token count
        test_list = [msg] + truncated
        if count_chat_tokens(test_list) > max_tokens:
            break
        truncated = [msg] + truncated
    return truncated


def count_chat_tokens(messages: list[BaseMessage]) -> int:
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

    total_tokens = 0

    for message in messages:
        total_tokens += len(encoding.encode(message.content))

        # Small overhead per message (role + separators)
        total_tokens += 4

    # Extra tokens for assistant reply priming
    total_tokens += 2

    return total_tokens


def extract_json_from_text(text: str) -> dict:
    """
    Extract JSON from text that may contain markdown code blocks.
    
    Args:
        text: Text that may contain JSON wrapped in markdown code blocks (```json ... ```)
        
    Returns:
        Parsed JSON as a dictionary
        
    Raises:
        json.JSONDecodeError: If no valid JSON can be extracted or parsed
    """
    # Remove markdown code blocks if present
    # Match ```json ... ``` or ``` ... ```
    text = re.sub(r'```json\s*\n?', '', text)
    text = re.sub(r'```\s*\n?', '', text)
    text = text.strip()
    
    # Try to find JSON object boundaries
    # Look for first { and last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx + 1]
        return json.loads(json_str)
    
    # If no braces found, try parsing the whole text
    return json.loads(text)


def create_column_names_for_schemas(schema_categories: dict) -> list:
    """
    schema_categories: dict where
        key   -> schema name
        value -> iterable of needed category IDs for that schema
    """

    all_schemas_metadata = []

    for schema, needed_categories in schema_categories.items():
        needed_cat_ids = {int(n) for n in needed_categories}

        # 1. Load the category mapping for this specific schema
        cat_path = os.path.join('..', 'data', 'short_schema', f'{schema}.json')
        if not os.path.exists(cat_path):
            continue  # or raise an error

        with open(cat_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)

        table_name = schema_data['table']
        all_schemas_metadata.append({'table_name': table_name})

        needed_columns = []
        for category in schema_data['categories']:
            if category['category_name'] in needed_cat_ids:
                needed_columns.extend(category['columns'])

        # 2. Load the actual metadata for this specific schema
        meta_path = os.path.join('..', 'data', 'metadata', f'{schema}_column_meta.json')
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, 'r', encoding='utf-8') as f:
            master_metadata = json.load(f)

        # 3. Filter and annotate with schema source
        for col_info in master_metadata:
            if col_info['name'] in needed_columns:
                col_info = col_info.copy()  # avoid mutating original metadata
                col_info['table_source'] = schema
                all_schemas_metadata.append(col_info)

    return all_schemas_metadata


def create_ddl_for_schemas(needed_columns_dict: dict, data_dir: str) -> str:
    all_schemas_text = ""

    for table_name, columns in needed_columns_dict.items():
        # Attempt to load metadata using fully qualified name first, falling back to name split
        [path, short_name] = table_name.split(".")
        meta_path = os.path.join(data_dir, 'metadata', f'{path + "_" + short_name}.json')
        if not os.path.exists(meta_path):
            meta_path = os.path.join(data_dir, 'metadata', f'{short_name}.json')
        
        if not os.path.exists(meta_path):
            print(f"Warning: Metadata file for {table_name} not found.")
            continue

        with open(meta_path, 'r', encoding='utf-8') as f:
            table_metadata = json.load(f)

        requested_cols_upper = [c.upper() for c in columns]
        filtered_cols = [
            col for col in table_metadata 
            if col['name'].upper() in requested_cols_upper
        ]

        all_schemas_text += f"CREATE TABLE {table_name} (\n"
        col_definitions = []
        for col in filtered_cols:
            col_def = f"  {col['name']} {col.get('type', 'VARCHAR')}"
            if col.get('description'):
                col_def += f" -- {col['description']}"
            col_definitions.append(col_def)
        
        all_schemas_text += ",\n".join(col_definitions)
        all_schemas_text += "\n);\n\n"

    return all_schemas_text



def create_data_context_for_schemas(needed_columns_dict: dict, data_dir: str) -> str:
    data_context_text = ""

    for table_name, columns in needed_columns_dict.items():
        [path, short_name] = table_name.split(".")
        meta_path = os.path.join(data_dir, 'metadata', f'{path + "_" + short_name}.json')
        if not os.path.exists(meta_path):
            meta_path = os.path.join(data_dir, 'metadata', f'{short_name}_column_meta.json')
        
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, 'r', encoding='utf-8') as f:
            table_metadata = json.load(f)

        requested_cols_upper = [c.upper() for c in columns]
        filtered_cols = [
            col for col in table_metadata 
            if col['name'].upper() in requested_cols_upper
        ]

        table_context_lines = []
        
        for col in filtered_cols:
            col_name = col['name']
            
            if col.get('unique_values'):
                vals_str = json.dumps(col['unique_values'], ensure_ascii=False)
                table_context_lines.append(f"  - {col_name} (Unique Values): {vals_str}")
            elif col.get('example_values'):
                vals_str = json.dumps(col['example_values'], ensure_ascii=False)
                table_context_lines.append(f"  - {col_name} (Example Values): {vals_str}")

        if table_context_lines:
            data_context_text += f"Table Content Summary for {table_name}:\n"
            data_context_text += "\n".join(table_context_lines)
            data_context_text += "\n\n"

    return data_context_text.strip()


def get_join_relationships(needed_views_dict, csv_file_path):
    """
    Identifies joinable columns between a set of required views based on a schema CSV.
    Supports comma-separated metadata lists with or without standard headers.
    """
    needed_table_names = set(needed_views_dict.keys())
    column_to_tables = defaultdict(list)

    if not os.path.exists(csv_file_path):
        print(f"Warning: Join relationships metadata file not found at {csv_file_path}")
        return {}

    with open(csv_file_path, mode='r', encoding='utf-8-sig') as f:
        # Detect if file contains specific column labels/headers
        sample = f.read(2048)
        f.seek(0)
        
        has_header = False
        if sample:
            first_line = sample.splitlines()[0].upper()
            if 'TABLE_NAME' in first_line or 'COLUMN_NAME' in first_line or 'VIEW_NAME' in first_line:
                has_header = True

        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                table = None
                column = None
                for k, v in row.items():
                    if k and k.upper() in ['TABLE_NAME', 'VIEW_NAME']:
                        table = v
                    elif k and k.upper() in ['COLUMN_NAME', 'COLUMN_NAME']:
                        column = v
                
                if not table or not column:
                    keys = list(row.keys())
                    if len(keys) >= 2:
                        table = row[keys[0]]
                        column = row[keys[1]]
                
                if table and column and table in needed_table_names:
                    column_to_tables[column].append(table)
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    table, column = row[0].strip(), row[1].strip()
                    if table in needed_table_names:
                        column_to_tables[column].append(table)

    join_metadata = {
        col: tables for col, tables in column_to_tables.items() 
        if len(tables) > 1
    }

    return join_metadata


def format_join_info_for_llm(join_info):
    output = ""
    for column, tables in join_info.items():
        table_list = ", ".join(tables)
        output += f"- {column}: Links {table_list}\n"
    return output


def ommiting_think_block(text: str) -> str:
    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string")

    import re
    cleaned = re.sub(
        r'(?is)<think>.*?</think>\s*',
        '',
        text
    ).strip()

    return cleaned


def fix_sql_wildcards(sql):
    import re
    pattern = r"LIKE\s+N'%([^']+)%'"
    
    def replace_spaces(match):
        content = match.group(1)
        modified_content = content.replace(" ", "%")
        return f"LIKE N'%{modified_content}%'"

    return re.sub(pattern, replace_spaces, sql)

def get_rag_context(user_question: str, n_results: int = 2, chroma_db_path: str = "../data/chroma_db"):
    # Initialize the same client and embedding function used in storage
    client = chromadb.PersistentClient(path=chroma_db_path)
    
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="embeddinggemma"
    )
    
    collection = client.get_collection(
        name="project_management_rag", 
        embedding_function=ollama_ef
    )

    results = collection.query(
        query_texts=[user_question],
        n_results=n_results
    )

    context_list = results.get("documents", [[]])[0]
    return "\n---\n".join(context_list)


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == "__main__":
    import pprint

    # Parameters as requested
    test_schemas = ['vw_Contracts']
    test_categories = [2] 

    # Execute function
    try:
        results = create_column_names_for_schemas(test_schemas, test_categories)
        
        # Display results
        print(f"--- Metadata Results (Found {len(results)} columns) ---")
        pprint.pprint(results)
        
    except Exception as e:
        print(f"An error occurred during testing: {e}")