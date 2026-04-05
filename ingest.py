"""
Ingest new FAQ rows into Chroma (immediate search) and supplemental JSONL (survives reindex).
"""
from preprocessor import preprocess
from embedder import append_to_collection
from supplemental_store import append_cleaned_records


def ingest_faq_records(
    raw_records: list[dict],
    collection,
    embed_model,
    default_sheet: str = "Imported FAQ",
) -> tuple[int, str]:
    """
    Preprocess, vector-append, and persist. Returns (count_added, user_message).
    """
    if not raw_records:
        return 0, "No records to ingest."

    normalized = []
    for r in raw_records:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        sheet = (r.get("sheet") or default_sheet).strip() or default_sheet
        if q and a:
            normalized.append({"question": q, "answer": a, "sheet": sheet})

    if not normalized:
        return 0, "No valid question/answer pairs."

    cleaned = preprocess(normalized)
    if not cleaned:
        return 0, "All rows were dropped during text cleaning."

    n = append_to_collection(collection, cleaned, embed_model)
    append_cleaned_records(cleaned)
    msg = (
        f"Added **{n}** FAQ entr{'y' if n == 1 else 'ies'} to the live index and saved them "
        "for the next full rebuild."
    )
    return n, msg
