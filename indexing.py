import os
from config import CHROMA_DIR
from data_loader import load_dataset
from preprocessor import preprocess
from embedder import build_index, load_index
from supplemental_store import load_supplemental_embed_records


def _merged_records_for_index(log=print) -> list[dict]:
    """Core Excel (preprocessed) plus persisted supplemental FAQs."""
    core_raw = load_dataset()
    core = preprocess(core_raw)
    extra = load_supplemental_embed_records()
    if extra:
        log(f"[indexing] Merging {len(extra)} supplemental FAQ row(s) with core dataset.")
    return core + extra


def setup(embed_model, log=print):
    index_exists = os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR)
    if index_exists:
        log("\n[indexing] Existing index found — skipping re-embedding.")
        return load_index(embed_model)
    log("\n[indexing] No index found — building from scratch...")
    records = _merged_records_for_index(log=log)
    return build_index(records, embed_model)


def reindex(embed_model, log=print):
    log("\n[indexing] Re-indexing dataset (core workbook + supplemental FAQs)...")
    records = _merged_records_for_index(log=log)
    collection = build_index(records, embed_model)
    log("[indexing] Re-indexing complete.\n")
    return collection
