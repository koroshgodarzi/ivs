import re
# from langchain_text_splitter import RecursiveCharacterTextSplitter
from utils import count_chat_tokens
import os
import tiktoken
from transformers import AutoTokenizer


def chunk_jsonl_recursive(entries, max_tokens=250, overlap=50):
    """Strategy 1: Recursive splitting for JSONL/General text."""
    processed_docs = []
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for entry in entries:
        title = entry.get("title", "")
        content = entry.get("data", "")
        related = ", ".join(entry.get("related_titles", []))
        
        if count_chat_tokens(content) <= max_tokens:
            processed_docs.append({
                "text": f"Title: {title}\nContent: {content}",
                "metadata": {"title": title, "related": related}
            })
        else:
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                processed_docs.append({
                    "text": f"Title: {title} (Part {i+1})\nContent: {chunk}",
                    "metadata": {"title": title, "related": related, "chunk_id": i}
                })
    return processed_docs

def chunk_markdown_sections(entries, source, max_tokens=250, overlap=50):
    """
    Splits Markdown by bold headers, then further chunks by tokens 
    if sections are too long.
    """
    tokenizer = get_tokenizer()
    processed_docs = []
    
    # Check if tokenizer is tiktoken or HuggingFace
    is_tiktoken = isinstance(tokenizer, tiktoken.Encoding)

    for entry in entries:
        content = entry.get("data", "")
        header_match = re.search(r'^#\s*(.*)', content, re.MULTILINE)
        main_title = header_match.group(1).strip() if header_match else "Untitled"

        # Split by bold sections **Title**
        sections = re.findall(r'\*\*(.*?)\*\*\s*(.*?)(?=\n\*\*|\Z)', content, re.DOTALL | re.MULTILINE)
        
        for title_part, data_part in sections:
            title_part = title_part.strip()
            data_part = data_part.strip()
            
            # Skip chunks with media/images (as per original logic)
            if re.search(r'!\[.*?\]\(.*?\)', data_part):
                continue

            full_text = f"**{title_part}**\n{data_part}"
            
            # Encode text to tokens
            if is_tiktoken:
                tokens = tokenizer.encode(full_text)
            else:
                tokens = tokenizer.encode(full_text, add_special_tokens=False)

            # If the section fits in one chunk
            if len(tokens) <= max_tokens:
                processed_docs.append({
                    "text": full_text,
                    "metadata": {
                        "source": source,
                        "path": f"{main_title} > {title_part}",
                        "chunk_id": 0
                    }
                })
            else:
                # Sliding window chunking
                chunk_idx = 0
                start_idx = 0
                
                while start_idx < len(tokens):
                    end_idx = start_idx + max_tokens
                    chunk_tokens = tokens[start_idx:end_idx]
                    
                    # Decode tokens back to text
                    if is_tiktoken:
                        chunk_text = tokenizer.decode(chunk_tokens)
                    else:
                        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    
                    processed_docs.append({
                        "text": chunk_text,
                        "metadata": {
                            "source": source,
                            "path": f"{main_title} > {title_part}",
                            "chunk_id": chunk_idx
                        }
                    })
                    
                    chunk_idx += 1
                    # Move window forward by (max - overlap)
                    start_idx += (max_tokens - overlap)
                    
                    # Prevent infinite loop if overlap >= max_tokens
                    if overlap >= max_tokens:
                        break

    return processed_docs
    

def get_tokenizer():
    """Helper to get the tokenizer based on environment variables."""
    backend = os.getenv("LLM_BACKEND", "openai")
    if backend == "openai":
        model = os.getenv("LLM_MODEL", "qwen2.5-coder-7b-instruct")
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    elif backend == "ollama":
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {backend}")