import os
import json
import numpy as np
import ollama


def embed_txt_folder(
    folder_path: str,
    output_array_path: str = "embeddings_disc.npy",
    output_index_path: str = "index_disc.json",
    model: str = "embeddinggemma",
):
    """
    Reads all .txt files in a folder, embeds their contents using Ollama,
    stores embeddings in a NumPy array, and saves an index JSON mapping
    row -> filename.
    """

    txt_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(".txt")
    )

    embeddings = []
    index_map = {}

    for idx, filename in enumerate(txt_files):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        response = ollama.embeddings(
            model=model,
            prompt=text
        )

        embedding = response["embedding"]
        embeddings.append(embedding)
        index_map[idx] = filename.split('_')[0]+'_'+filename.split('_')[1]

    # Convert to NumPy array (shape: [num_files, embedding_dim])
    embedding_array = np.array(embeddings, dtype=np.float32)

    # Save outputs
    np.save(output_array_path, embedding_array)

    with open(output_index_path, "w", encoding="utf-8") as f:
        json.dump(index_map, f, indent=2)

    return embedding_array, index_map


def embed_descriptions_from_json(
    json_input_path: str,
    output_array_path: str = "embeddings_md_disc.npy",
    output_index_path: str = "index_md_disc.json",
    model: str = "embeddinggemma",
):
    """
    Reads a JSON file containing file_names and descriptions, 
    embeds the descriptions using Ollama, stores them in a NumPy array, 
    and saves an index mapping row -> filename.
    """

    # 1. Read the JSON file created in the previous step
    if not os.path.exists(json_input_path):
        print(f"Error: File {json_input_path} not found.")
        return None, None

    with open(json_input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = []
    index_map = {}

    print(f"Starting embedding process using model: {model}...")

    for idx, entry in enumerate(data):
        filename = entry.get("file_name", f"unknown_{idx}")
        description = entry.get("description", "")

        if not description:
            print(f"Warning: No description for {filename}. Skipping.")
            continue

        # 2. Generate embedding for the Farsi description
        try:
            response = ollama.embeddings(
                model=model,
                prompt=description
            )
            
            embedding = response["embedding"]
            embeddings.append(embedding)
            
            # 3. Map the current index to the original filename
            index_map[idx] = filename
            print(f"[{idx+1}/{len(data)}] Embedded description for: {filename}")

        except Exception as e:
            print(f"Error embedding {filename}: {e}")

    if not embeddings:
        print("No embeddings were generated.")
        return None, None

    # 4. Convert to NumPy array (shape: [num_entries, embedding_dim])
    embedding_array = np.array(embeddings, dtype=np.float32)

    # 5. Save the .npy file
    np.save(output_array_path, embedding_array)

    # 6. Save the index mapping (ensure_ascii=False for potential Farsi filenames)
    with open(output_index_path, "w", encoding="utf-8") as f:
        json.dump(index_map, f, indent=2, ensure_ascii=False)

    print(f"\nSuccess!")
    print(f"Embeddings saved to: {output_array_path}")
    print(f"Index map saved to: {output_index_path}")

    return embedding_array, index_map


if __name__ == "__main__":
    # embed_txt_folder('short_schema')
    embed_descriptions_from_json('md_description.json')