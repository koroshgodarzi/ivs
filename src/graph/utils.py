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

from dotenv import load_dotenv
load_dotenv()


import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

def get_llm(model_id: str, max_tokens: int = 4096):
    """
    model_id comes from the UI/API request.
    Example IDs: 'gpt-4o', 'qwen-72b-api', 'ollama-qwen'
    """
    # 1. OLLAMA Logic
    if model_id.startswith("ollama"):
        # You can extract the specific version if you send 'ollama:qwen2.5'
        model_name = model_id.split(":")[1::] if ":" in model_id else "qwen2.5-coder:14b"
        model_name = model_name[0] + ':' + model_name[1]
        return ChatOllama(
            model=model_name,
            temperature=0,
            num_predict=max_tokens,
            format="json",
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


def ommiting_think_block(text: str) -> str:
    """
    Extract a SQL query from model output.
    - Removes any <think>...</think> blocks (if present)
    - Extracts SQL starting from first SELECT or WITH
    - Validates basic SQL-only constraints
    """

    if not text or not isinstance(text, str):
        raise ValueError("Input must be a non-empty string")

    cleaned = re.sub(
        r'(?is)<think>.*?</think>\s*',
        '',
        text
    ).strip()

    return cleaned


def fix_sql_wildcards(sql):
    # This regex finds N'%...%' and captures the content inside the wildcards
    pattern = r"LIKE\s+N'%([^']+)%'"
    
    def replace_spaces(match):
        content = match.group(1)
        # Replace spaces with %
        modified_content = content.replace(" ", "%")
        return f"LIKE N'%{modified_content}%'"

    return re.sub(pattern, replace_spaces, sql)


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