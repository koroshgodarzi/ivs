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


def get_llm(max_tokens: int = 1024):
    backend = os.getenv("LLM_BACKEND", "openai")

    if backend == "openai":
        return ChatOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            model=os.getenv("LLM_MODEL"),
            temperature=0,
            max_tokens=max_tokens,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    elif backend == "ollama":
        return ChatOllama(
            model=os.getenv("LLM_MODEL", "qwen2.5-coder:14b"),
            temperature=0,
            num_predict=max_tokens,
            format="json",
        )

    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend}")


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