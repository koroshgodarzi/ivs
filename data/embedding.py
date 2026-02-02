from __future__ import annotations

import numpy as np
import pandas as pd
import ollama
import os
import json
from typing import Optional, Tuple


def build_bilingual_view_embeddings(
    english_xlsx: str = "BI Views description_.xlsx",
    persian_xlsx: str = "BI_Views_description_Persian.xlsx",
    *,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "gemmaembedding",  # kept for API compatibility
    sheet_name: Optional[str] = 0,
    timeout_s: int = 120,
    save_npy_path: Optional[str] = "embeddings.npy",
    view_index_mapping_path: str = "view_index_mapping.json",
    require_all_matches: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      embeddings: float32 array of shape (2*N, D)
      view_names: string array of shape (2*N,) aligned with embeddings
    """

    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma")

    df_en = pd.read_excel(english_xlsx, sheet_name=sheet_name, engine="openpyxl")
    df_fa = pd.read_excel(persian_xlsx, sheet_name=sheet_name, engine="openpyxl")

    if df_en.shape[1] < 2 or df_fa.shape[1] < 2:
        raise ValueError("Each file must have at least 2 columns: name and description.")

    # Normalize names
    en_map = {
        str(name).strip(): ("" if pd.isna(desc) else str(desc).strip())
        for name, desc in zip(df_en.iloc[:, 0], df_en.iloc[:, 1])
    }
    fa_map = {
        str(name).strip(): ("" if pd.isna(desc) else str(desc).strip())
        for name, desc in zip(df_fa.iloc[:, 0], df_fa.iloc[:, 1])
    }

    common = sorted(set(en_map) & set(fa_map))

    missing_in_fa = sorted(set(en_map) - set(fa_map))
    missing_in_en = sorted(set(fa_map) - set(en_map))

    if require_all_matches and (missing_in_fa or missing_in_en):
        msg = []
        if missing_in_fa:
            msg.append(f"Missing in Persian file: {missing_in_fa[:10]}{'...' if len(missing_in_fa) > 10 else ''}")
        if missing_in_en:
            msg.append(f"Missing in English file: {missing_in_en[:10]}{'...' if len(missing_in_en) > 10 else ''}")
        raise ValueError("View-name mismatch between files. " + " | ".join(msg))

    if not common:
        raise ValueError("No matching view names found between the two files.")

    rows_emb: list[np.ndarray] = []
    rows_names: list[str] = []
    index_to_view: dict[int, str] = {}

    for view in common:
        en_desc = en_map[view]
        fa_desc = fa_map[view]

        if not en_desc:
            raise ValueError(f"Empty English description for view {view!r}")
        if not fa_desc:
            raise ValueError(f"Empty Persian description for view {view!r}")

        # English embedding
        resp_en = ollama.embed(model=embed_model, input=en_desc)
        emb_en = np.array(resp_en["embeddings"][0], dtype=np.float32)

        idx = len(rows_emb)
        rows_emb.append(emb_en)
        rows_names.append(view)
        index_to_view[idx] = view.split()[-1]

        # Persian embedding
        resp_fa = ollama.embed(model=embed_model, input=fa_desc)
        emb_fa = np.array(resp_fa["embeddings"][0], dtype=np.float32)

        idx = len(rows_emb)
        rows_emb.append(emb_fa)
        rows_names.append(view)
        index_to_view[idx] = view.split()[-1]

    embeddings = np.asarray(rows_emb, dtype=np.float32)
    view_names = np.asarray(rows_names, dtype=object)

    if save_npy_path:
        np.save(save_npy_path, embeddings)

    if view_index_mapping_path:
        with open(view_index_mapping_path, "w", encoding="utf-8") as f:
            json.dump(index_to_view, f, ensure_ascii=False, indent=2)

    return embeddings, view_names


if __name__ == "__main__":
    build_bilingual_view_embeddings()
