from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
import json
import re

from dotenv import load_dotenv
load_dotenv()


def get_llm():
    """
    Connects to Hugging Face's OpenAI-compatible API.
    This fixes 'StopIteration' and 'Task Support' errors.
    """
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set")
    
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto", 
    )
    return ChatHuggingFace(llm=llm)


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


def create_column_names_for_schemas(schema_list: list, needed_categories: str) -> str:
    
    needed_cat_ids = [int(n) for n in needed_categories]
    all_schemas_metadata = []

    # Iterate through every schema name in the list
    for schema in schema_list:
        # 1. Load the category mapping for this specific schema
        cat_path = os.path.join('..', 'data', 'short_schema', f'{schema}_column_cat.json')
        if not os.path.exists(cat_path):
            continue # Or handle error: schema file missing
            
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

        # 3. Filter and add a 'table' identifier so the LLM knows which table is which
        for col_info in master_metadata:
            if col_info['name'] in needed_columns:
                # Optional: Add the schema/table name to the dict so the LLM knows context
                col_info['table_source'] = schema 
                all_schemas_metadata.append(col_info)

    return all_schemas_metadata


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