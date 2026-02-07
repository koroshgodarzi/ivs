from __future__ import annotations

import numpy as np
import pandas as pd
import os
import json
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer


def build_bilingual_view_embeddings(
    english_xlsx: str = "BI Views description_.xlsx",
    persian_xlsx: str = "BI_Views_description_Persian.xlsx",
    *,
    sheet_name: Optional[str] = 0,
    save_npy_path: Optional[str] = "embeddings_multilingual.npy",
    view_index_mapping_path: str = "view_index_mapping.json",
    require_all_matches: bool = True,
    device: Optional[str] = None,  # "cpu", "cuda", "mps"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      embeddings: float32 array of shape (2*N, D)
      view_names: string array of shape (2*N,) aligned with embeddings
    """

    # Load Multilingual-E5-large
    model = SentenceTransformer(
        "intfloat/multilingual-e5-large",
        device=device,
    )

    df_en = pd.read_excel(english_xlsx, sheet_name=sheet_name, engine="openpyxl")
    df_fa = pd.read_excel(persian_xlsx, sheet_name=sheet_name, engine="openpyxl")

    if df_en.shape[1] < 2 or df_fa.shape[1] < 2:
        raise ValueError("Each file must have at least 2 columns: name and description.")

    # Normalize names → description
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

    texts: list[str] = []
    rows_names: list[str] = []
    index_to_view: dict[int, str] = {}

    for view in common:
        en_desc = en_map[view]
        fa_desc = fa_map[view]

        if not en_desc:
            raise ValueError(f"Empty English description for view {view!r}")
        if not fa_desc:
            raise ValueError(f"Empty Persian description for view {view!r}")

        # E5 requires prefixes
        texts.append(f"passage: {en_desc}")
        rows_names.append(view)
        index_to_view[len(texts) - 1] = view.split()[-1]

        texts.append(f"passage: {fa_desc}")
        rows_names.append(view)
        index_to_view[len(texts) - 1] = view.split()[-1]

    # Compute embeddings (batched + normalized)
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    view_names = np.asarray(rows_names, dtype=object)

    if save_npy_path:
        np.save(save_npy_path, embeddings)

    if view_index_mapping_path:
        with open(view_index_mapping_path, "w", encoding="utf-8") as f:
            json.dump(index_to_view, f, ensure_ascii=False, indent=2)

    return embeddings, view_names


if __name__ == "__main__":
    build_bilingual_view_embeddings('BI_Views_description_English_Detailed.xlsx', 'BI_Views_description_Persian_Refined.xlsx')
