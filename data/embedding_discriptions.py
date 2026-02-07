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


if __name__ == "__main__":
    embed_txt_folder('short_schema')